import gc
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Subset

from methods._trainer import _Trainer
from utils.memory import MemoryBatchSampler
from utils.train_utils import select_scheduler


class LaPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(LaPrompt, self).__init__(*args, **kwargs)
        self.laprompt_use_ca = kwargs.get("laprompt_use_ca", False)
        self.laprompt_ca_lr = kwargs.get("laprompt_ca_lr", 0.01)
        self.laprompt_ca_epochs = kwargs.get("laprompt_ca_epochs", 5)
        self.laprompt_ca_storage = kwargs.get("laprompt_ca_storage", "variance")
        self.laprompt_ema_decay = kwargs.get("laprompt_ema_decay", 0.0)
        self.laprompt_tuned_epoch = kwargs.get("laprompt_tuned_epoch", 1)
        self.laprompt_n_centroids = kwargs.get("laprompt_n_centroids", 5)
        self.laprompt_add_num = kwargs.get("laprompt_add_num", 8)
        
        # 新增: 在线学习优化策略开关
        self.use_dynamic_logit_mask = kwargs.get("use_dynamic_logit_mask", True)
        self.use_forgetting_aware_init = kwargs.get("use_forgetting_aware_init", True)
        self.logit_mask_temp = kwargs.get("logit_mask_temp", 1.0)
        
        # 遗忘感知初始化策略
        self.init_strategy = kwargs.get("init_strategy", "variance_perturb")
        self.init_scale = kwargs.get("init_scale", 0.01)

        if kwargs.get("tuned_epoch") is not None:
            self.laprompt_tuned_epoch = kwargs.get("tuned_epoch")
        if kwargs.get("ca_lr") is not None:
            self.laprompt_ca_lr = kwargs.get("ca_lr")
        if kwargs.get("crct_epochs") is not None:
            self.laprompt_ca_epochs = kwargs.get("crct_epochs")
        if kwargs.get("ca_storage_efficient_method") is not None:
            self.laprompt_ca_storage = kwargs.get("ca_storage_efficient_method")
        if kwargs.get("n_centroids") is not None:
            self.laprompt_n_centroids = kwargs.get("n_centroids")
        if kwargs.get("add_num") is not None:
            self.laprompt_add_num = kwargs.get("add_num")
        if kwargs.get("ema_decay") is not None:
            self.laprompt_ema_decay = kwargs.get("ema_decay")
        if kwargs.get("logit_mask_temp") is not None:
            self.logit_mask_temp = kwargs.get("logit_mask_temp")

        self.current_task = 0
        self.cls_mean: Dict[int, torch.Tensor] = {}
        self.cls_cov: Dict[int, torch.Tensor] = {}
        self.cls_multi_means: Dict[int, torch.Tensor] = {}
        self.cls_multi_covs: Dict[int, torch.Tensor] = {}
        self.cls2task: Dict[int, int] = {}
        self._task_seen_labels: List[int] = []
        self._task_index_to_label: Dict[int, int] = {}
        self._prev_task_class_count = 0

    def setup_distributed_model(self):
        super().setup_distributed_model()
        if not hasattr(self.model_without_ddp, "forward_features"):
            raise ValueError(
                "laprompt mode requires a model with `forward_features`; "
                "set --model_name laprompt"
            )

    def online_before_task(self, task_id):
        self.current_task = task_id
        self._task_seen_labels = []
        self._task_index_to_label = {}
        self._prev_task_class_count = len(self.exposed_classes)
        
        # 遗忘感知初始化: 基于历史统计信息初始化新任务提示
        if self.use_forgetting_aware_init and task_id > 0:
            self._initialize_prompts_forgetting_aware(task_id)

    def online_after_task(self, task_id):
        if self.laprompt_use_ca:
            self._compute_task_statistics(task_id)
            if task_id > 0:
                self.classifier_align()

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        for j in range(len(labels)):
            raw_label = labels[j].item()
            labels[j] = self.exposed_classes.index(raw_label)
            self._task_seen_labels.append(raw_label)
            self._task_index_to_label[int(idx[j].item())] = int(labels[j].item())

        self.memory_sampler = MemoryBatchSampler(
            self.memory,
            self.memory_batchsize,
            self.temp_batchsize * self.online_iter * self.world_size,
        )
        self.memory_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.memory_batchsize,
            sampler=self.memory_sampler,
            num_workers=4,
        )
        self.memory_provider = iter(self.memory_dataloader)

        total_loss, total_acc, total_iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            total_loss += loss
            total_acc += acc
            total_iter += 1

        self.update_memory(idx, labels)
        del images, labels
        gc.collect()
        return total_loss / total_iter, total_acc / total_iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        has_memory_batch = False

        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in range(len(memory_labels)):
                memory_labels[i] = self.exposed_classes.index(memory_labels[i].item())
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)
            has_memory_batch = True

        x = x.to(self.device)
        y = y.to(self.device)
        x = self.train_transform(x)

        self.optimizer.zero_grad()
        logit, loss = self.model_forward(x, y, has_memory_batch=has_memory_batch)
        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.laprompt_ema_decay > 0 and hasattr(self.model_without_ddp.backbone, "lapromptPool"):
            self.model_without_ddp.backbone.lapromptPool.update_attn_ema()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)
        return total_loss, total_correct / total_num_data

    def model_forward(self, x, y, has_memory_batch=False):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            taskids = [self.current_task] * x.size(0)
            output = self.model(x, train=True, taskids=taskids)

            if isinstance(output, dict):
                logit = output["logits"]
            else:
                logit = output

            logit = logit + self.mask
            
            # 动态Logit掩码: 增强版非参数掩码策略
            if self.use_dynamic_logit_mask:
                logit = self._apply_dynamic_logit_mask(logit, y, has_memory_batch)
            else:
                # 原始基础掩码逻辑
                if (
                    self.current_task > 0
                    and self._prev_task_class_count > 0
                    and not has_memory_batch
                ):
                    current_task_rows = y >= self._prev_task_class_count
                    if torch.any(current_task_rows):
                        logit[current_task_rows, : self._prev_task_class_count] = float("-inf")
            
            loss = F.cross_entropy(logit, y.to(torch.int64))
        return logit, loss
    
    def _apply_dynamic_logit_mask(self, logit, y, has_memory_batch):
        """
        动态Logit掩码策略v3: 基于样本难度的自适应mask
        思路: 对容易样本（高置信度）应用更强的旧类别抑制
        """
        if self.current_task == 0:
            return logit
        
        if self._prev_task_class_count > 0 and not has_memory_batch:
            current_task_rows = y >= self._prev_task_class_count
            if torch.any(current_task_rows):
                # 获取当前任务样本的logits
                current_logits = logit[current_task_rows]
                
                # 计算每个样本的置信度（当前任务类别的概率）
                probs = F.softmax(current_logits, dim=1)
                current_task_probs = probs[:, self._prev_task_class_count:].sum(dim=1)
                
                # 自适应mask: 对高置信度样本（>0.6）应用更强抑制
                easy_samples = current_task_probs > 0.6
                hard_samples = ~easy_samples
                
                # 对容易样本: 强抑制旧类别 (mask=0.1)
                if easy_samples.any():
                    logit[current_task_rows][easy_samples, :self._prev_task_class_count] *= 0.1
                
                # 对困难样本: 弱抑制旧类别 (mask=0.5)
                if hard_samples.any():
                    logit[current_task_rows][hard_samples, :self._prev_task_class_count] *= 0.5
        
        return logit

    def online_evaluate(self, test_loader, samples_cnt=None):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)

        self.model.eval()
        if self.laprompt_ema_decay > 0 and hasattr(self.model_without_ddp.backbone, "lapromptPool"):
            self.model_without_ddp.backbone.lapromptPool.use_attn_ema_params()
        with torch.no_grad():
            for data in test_loader:
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    output_nomask = self.model(x, train=False, taskids=None)
                    logits_nomask = output_nomask["logits"] + self.mask
                    pred_classes = torch.argmax(logits_nomask, dim=1)

                task_ids = []
                for cls_idx in pred_classes.tolist():
                    mapped_task = self.cls2task.get(int(cls_idx), self.current_task)
                    task_ids.append(int(mapped_task))

                output = self.model(x, train=False, taskids=task_ids)

                if isinstance(output, dict):
                    logit = output["logits"]
                else:
                    logit = output

                logit = logit + self.mask
                loss = F.cross_entropy(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        return {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

    def _compute_task_statistics(self, task_id):
        if not hasattr(self.model_without_ddp, "forward_features"):
            return
        if not self._task_index_to_label:
            return
        class_ids = sorted(set(self._task_index_to_label.values()))
        if not class_ids:
            return

        sample_ids = sorted(self._task_index_to_label.keys())
        pseudo_labels = torch.tensor(
            [self._task_index_to_label[sid] for sid in sample_ids],
            dtype=torch.long,
        )
        subset = Subset(self.train_dataset, sample_ids)
        stats_loader = DataLoader(
            subset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=4,
        )

        feature_bank = []
        label_bank = []
        with torch.no_grad():
            ptr = 0
            for images, _ in stats_loader:
                images = images.to(self.device)
                taskids = [task_id] * images.size(0)
                features = self.model_without_ddp.forward_features(images, taskids=taskids)
                if isinstance(features, tuple):
                    features = features[0]
                feature_bank.append(features.detach().cpu())

                bs = images.size(0)
                label_bank.append(pseudo_labels[ptr:ptr + bs].cpu())
                ptr += bs

        if not feature_bank:
            return

        all_features = torch.cat(feature_bank, dim=0)
        all_labels = torch.cat(label_bank, dim=0)
        for cls in class_ids:
            idx = (all_labels == cls).nonzero().squeeze(-1)
            if idx.numel() <= 1:
                continue
            cls_feat = all_features[idx]
            mean = cls_feat.mean(dim=0)
            if self.laprompt_ca_storage == "variance":
                cov = self._safe_variance(cls_feat)
            elif self.laprompt_ca_storage == "multi-centroid":
                self._compute_multi_centroid_for_class(cls, cls_feat)
                self.cls2task[cls] = task_id
                continue
            elif self.laprompt_ca_storage != "covariance":
                raise NotImplementedError(
                    f"laprompt_ca_storage={self.laprompt_ca_storage} is not implemented"
                )
            else:
                cov = self._safe_covariance(cls_feat)
            self.cls_mean[cls] = mean.to(self.device)
            self.cls_cov[cls] = cov.to(self.device)
            self.cls2task[cls] = task_id

    def _safe_variance(self, cls_feat):
        centered = cls_feat - cls_feat.mean(dim=0, keepdim=True)
        denom = max(1, cls_feat.size(0) - 1)
        var = (centered.pow(2).sum(dim=0) / denom).float()
        var = torch.nan_to_num(var, nan=1e-4, posinf=1.0, neginf=1e-4)
        return var + 1e-4

    def _safe_covariance(self, cls_feat):
        centered = cls_feat - cls_feat.mean(dim=0, keepdim=True)
        denom = max(1, cls_feat.size(0) - 1)
        cov = (centered.T @ centered) / denom
        cov = cov.float()
        cov = 0.5 * (cov + cov.T)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = cov + torch.eye(cov.shape[0], device=cov.device) * 1e-4
        min_eig = torch.linalg.eigvalsh(cov).min()
        if min_eig <= 0:
            cov = cov + torch.eye(cov.shape[0], device=cov.device) * (-min_eig + 1e-5)
        return cov

    def _compute_multi_centroid_for_class(self, cls, cls_feat):
        if cls_feat.size(0) < 2:
            return

        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError("multi-centroid CA requires scikit-learn") from exc

        k = int(max(1, min(self.laprompt_n_centroids, cls_feat.size(0))))
        x = cls_feat.float()
        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=self.rnd_seed,
        )
        assigns = torch.from_numpy(kmeans.fit_predict(x.cpu().numpy())).to(x.device)

        means = []
        covs = []
        for c in range(k):
            members = x[assigns == c]
            if members.size(0) <= 1:
                continue
            c_mean = members.mean(dim=0)
            c_cov = self._safe_covariance(members)
            means.append(c_mean)
            covs.append(c_cov)

        if means:
            self.cls_multi_means[cls] = torch.stack(means).to(self.device)
            self.cls_multi_covs[cls] = torch.stack(covs).to(self.device)
        else:
            self.cls_multi_means[cls] = cls_feat.mean(dim=0, keepdim=True).to(self.device)
            self.cls_multi_covs[cls] = self._safe_covariance(cls_feat).unsqueeze(0).to(self.device)

    def classifier_align(self):
        if self.laprompt_ca_storage == "multi-centroid":
            class_ids = sorted(self.cls_multi_means.keys())
        else:
            class_ids = sorted(self.cls_mean.keys())
        if not class_ids:
            return

        was_training = self.model.training
        self.model.train()

        head = self.model_without_ddp.backbone.backbone.head
        optimizer = optim.SGD(
            head.parameters(),
            lr=self.laprompt_ca_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.laprompt_ca_epochs),
        )

        num_sampled_per_cls = max(int(self.laprompt_add_num), int(self.batchsize) * 2)
        for _ in range(self.laprompt_ca_epochs):
            sampled_data = []
            sampled_label = []
            for class_id in class_ids:
                if self.laprompt_ca_storage == "multi-centroid":
                    means = self.cls_multi_means[class_id]
                    covs = self.cls_multi_covs[class_id]
                    per_component = max(1, num_sampled_per_cls // means.size(0))
                    for comp in range(means.size(0)):
                        mean = means[comp]
                        cov = covs[comp]
                        if torch.isnan(mean).any() or torch.isnan(cov).any():
                            continue
                        if torch.isinf(mean).any() or torch.isinf(cov).any():
                            continue
                        cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
                        dist_obj = MultivariateNormal(mean.float(), cov.float())
                        feats = dist_obj.sample((per_component,))
                        sampled_data.append(feats)
                        sampled_label.extend([class_id] * per_component)
                else:
                    mean = self.cls_mean[class_id]
                    cov = self.cls_cov[class_id]
                    if self.laprompt_ca_storage == "variance":
                        cov = torch.diag(cov)
                    if cov.ndim != 2:
                        continue
                    if torch.isnan(mean).any() or torch.isnan(cov).any():
                        continue
                    if torch.isinf(mean).any() or torch.isinf(cov).any():
                        continue

                    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
                    dist_obj = MultivariateNormal(mean.float(), cov.float())
                    feats = dist_obj.sample((num_sampled_per_cls,))
                    sampled_data.append(feats)
                    sampled_label.extend([class_id] * num_sampled_per_cls)

            if not sampled_data:
                break

            inputs = torch.cat(sampled_data, dim=0).to(self.device)
            targets = torch.tensor(sampled_label, dtype=torch.long, device=self.device)
            shuffled = torch.randperm(inputs.size(0), device=inputs.device)
            inputs = inputs[shuffled]
            targets = targets[shuffled]

            batch = max(32, self.batchsize)
            for start in range(0, inputs.size(0), batch):
                end = start + batch
                inp = inputs[start:end]
                tgt = targets[start:end]
                logits = self.model_without_ddp(inp, fc_only=True)["logits"]
                loss = F.cross_entropy(logits, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        if not was_training:
            self.model.eval()

    def _initialize_prompts_forgetting_aware(self, task_id):
        """
        遗忘感知初始化策略选择器
        根据 init_strategy 参数选择不同的初始化方法
        """
        init_strategy = getattr(self, 'init_strategy', 'variance_perturb')
        
        # 调试信息
        has_stats = hasattr(self, 'cls_mean') and len(self.cls_mean) > 0
        print(f"[Init] Task {task_id}: strategy={init_strategy}, has_stats={has_stats}, cls_mean_len={len(self.cls_mean) if hasattr(self, 'cls_mean') else 0}")
        
        if init_strategy == 'variance_perturb':
            self._init_variance_perturbation(task_id)
        elif init_strategy == 'knn_prototype':
            self._init_knn_prototype(task_id)
        elif init_strategy == 'gaussian_sample':
            self._init_gaussian_sample(task_id)
        elif init_strategy == 'task_orthogonal':
            self._init_task_orthogonal(task_id)
        elif init_strategy == 'mean_centered':
            self._init_mean_centered(task_id)
        elif init_strategy == 'contrastive_init':
            self._init_contrastive(task_id)
        elif init_strategy == 'zero_init':
            self._init_zero(task_id)
        elif init_strategy == 'random_scaled':
            self._init_random_scaled(task_id)
    
    def _init_variance_perturbation(self, task_id):
        """策略1: 基于历史方差的扰动初始化"""
        if not self.cls_mean or not self.cls_cov:
            return
        
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if not hasattr(prompt_pool, 'layer_embedding'):
            return
        
        with torch.no_grad():
            avg_var = torch.stack(list(self.cls_cov.values())).mean()
            std = torch.sqrt(avg_var).item()
            
            for layer_idx in range(prompt_pool.layer_embedding.size(0)):
                layer_emb = prompt_pool.layer_embedding[layer_idx]
                perturbation = torch.randn_like(layer_emb) * std * self.init_scale
                prompt_pool.layer_embedding[layer_idx] += perturbation
    
    def _init_knn_prototype(self, task_id):
        """策略2: 基于K近邻原型的初始化"""
        if not self.cls_mean:
            return
        
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if not hasattr(prompt_pool, 'prompt'):
            return
        
        with torch.no_grad():
            # 使用历史类别均值作为原型
            prototypes = torch.stack(list(self.cls_mean.values()))
            # 新任务提示 = 最近原型 + 扰动
            if task_id < prompt_pool.prompt.size(0):
                nearest_proto = prototypes.mean(dim=0)  # 简化：使用平均原型
                prompt_pool.prompt[task_id] += nearest_proto[:prompt_pool.prompt[task_id].size(0)] * self.init_scale
    
    def _init_gaussian_sample(self, task_id):
        """策略3: 从历史分布中采样初始化"""
        if not self.cls_mean or not self.cls_cov:
            return
        
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if not hasattr(prompt_pool, 'prompt'):
            return
        
        with torch.no_grad():
            # 计算历史分布的均值和协方差
            hist_mean = torch.stack(list(self.cls_mean.values())).mean(dim=0)
            hist_cov = torch.stack(list(self.cls_cov.values())).mean(dim=0)
            
            if task_id < prompt_pool.prompt.size(0):
                # 从历史分布采样
                sample = torch.randn_like(prompt_pool.prompt[task_id])
                sample = sample * torch.sqrt(hist_cov[:sample.size(0)]) + hist_mean[:sample.size(0)]
                prompt_pool.prompt[task_id] = sample * self.init_scale
    
    def _init_task_orthogonal(self, task_id):
        """策略4: 任务正交初始化 - 使新任务与历史任务正交"""
        if not self.cls_mean or task_id == 0:
            return
        
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if not hasattr(prompt_pool, 'prompt'):
            return
        
        with torch.no_grad():
            if task_id < prompt_pool.prompt.size(0):
                # 计算历史任务的平均方向
                hist_prompts = prompt_pool.prompt[:task_id].mean(dim=0)
                current_prompt = prompt_pool.prompt[task_id]
                
                # 正交化：减去在历史方向上的投影
                proj = (current_prompt * hist_prompts).sum() / (hist_prompts.norm()**2 + 1e-8)
                orthogonal = current_prompt - proj * hist_prompts
                prompt_pool.prompt[task_id] = orthogonal * self.init_scale
    
    def _init_mean_centered(self, task_id):
        """策略5: 均值中心化初始化"""
        if not self.cls_mean:
            return
        
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if not hasattr(prompt_pool, 'prompt'):
            return
        
        with torch.no_grad():
            hist_center = torch.stack(list(self.cls_mean.values())).mean(dim=0)
            if task_id < prompt_pool.prompt.size(0):
                # 新任务提示围绕历史中心
                prompt_pool.prompt[task_id] = hist_center[:prompt_pool.prompt[task_id].size(0)] * self.init_scale
    
    def _init_contrastive(self, task_id):
        """策略6: 对比式初始化 - 拉开与负样本的距离"""
        if not self.cls_mean or task_id == 0:
            return
        
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if not hasattr(prompt_pool, 'prompt'):
            return
        
        with torch.no_grad():
            # 获取负样本（历史类别）中心
            neg_center = torch.stack(list(self.cls_mean.values())).mean(dim=0)
            if task_id < prompt_pool.prompt.size(0):
                current = prompt_pool.prompt[task_id]
                # 远离负样本中心
                direction = current - neg_center[:current.size(0)]
                prompt_pool.prompt[task_id] = current + direction * self.init_scale
    
    def _init_zero(self, task_id):
        """策略7: 零初始化 - 作为对照"""
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if hasattr(prompt_pool, 'prompt') and task_id < prompt_pool.prompt.size(0):
            with torch.no_grad():
                prompt_pool.prompt[task_id].zero_()
    
    def _init_random_scaled(self, task_id):
        """策略8: 随机缩放初始化"""
        prompt_pool = self.model_without_ddp.backbone.lapromptPool
        if hasattr(prompt_pool, 'prompt') and task_id < prompt_pool.prompt.size(0):
            with torch.no_grad():
                prompt_pool.prompt[task_id] *= self.init_scale
    
    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, None)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def update_memory(self, sample, label):
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()

        idx = []
        if self.is_main_process():
            for _ in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    picked = torch.randint(0, self.seen, (1,)).item()
                    if picked < self.memory_size:
                        idx.append(picked)
                    else:
                        idx.append(self.memory_size)

        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(), dtype=torch.long).to(self.device)
            dist.barrier()
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()

        for i, index in enumerate(idx):
            mem_label = self.exposed_classes[label[i].item()]
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], mem_label], index)
            else:
                self.memory.replace_data([sample[i], mem_label])
