import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
import sklearn.metrics
from tqdm import tqdm

try:
    from downstream.wesad.wesad_tools import Dataset_train, Dataset_multiclass, train_model, eval_model, get_logger
    from downstream.model_select import select_model
    from codebook.utils import cosine_scheduler
except ImportError as e:
    print(f"Warning: Import failed - {e}")
    print("请确保所有依赖模块在正确路径下")


class LabRAMAdapter(nn.Module):

    def __init__(self, model, patch_size=100):
        super(LabRAMAdapter, self).__init__()
        self.model = model
        self.patch_size = patch_size
        print(f"📦 LabRAMAdapter initialized with patch_size={patch_size}")

    def forward(self, x):
        # 确保输入是3D: (batch, channels, time)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)

        seq_len = x.shape[-1]

        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, pad_len))

        x = rearrange(x, 'b c (n t) -> b c n t', t=self.patch_size)

        try:
            return self.model(x, input_chans=[0, 1])
        except TypeError:
            return self.model(x)


class WESADTrainer:
    def __init__(self, args):
        self.args = args

        # ✨ NEW: Get train_data_ratio from args
        self.train_data_ratio = getattr(args, 'train_data_ratio', 1.0)

        # Validate train_data_ratio
        if not 0.0 < self.train_data_ratio <= 1.0:
            raise ValueError(f"train_data_ratio must be between 0.0 and 1.0, got {self.train_data_ratio}")

        # 从args获取task_name
        if not hasattr(args, 'task_name') or args.task_name is None:
            raise ValueError("必须指定 task_name 参数,可选值: 'binary' 或 'multiclass'")

        self.task_name = args.task_name

        # 验证task_name
        if self.task_name not in ['binary', 'multiclass']:
            raise ValueError(f"无效的 task_name: {self.task_name}. 必须是 'binary' 或 'multiclass'")

        # 根据task_name设置数据路径和类别数
        self.setup_task()

        self.device = torch.device(args.device if hasattr(args, 'device') else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))

        # 模型配置
        self.backbone = args.backbone
        self.IN_CHANS = 1

        # SOTA格式参数 (50Hz × 60s = 3000)
        self.WINDOW_SIZE = 3000
        self.SAMPLING_RATE = 50
        self.PATCH_SIZE = getattr(args, 'patch_size', 40)
        self.TARGET_PATCHES = self.WINDOW_SIZE // self.PATCH_SIZE  # ~23 patches

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs

        self.pretrained = getattr(args, 'pretrained', True)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)
        self.checkpoint_path = getattr(args, 'checkpoint_path', None)

        # 训练参数
        self.patience = getattr(args, 'patience', 15)
        self.clip_grad = getattr(args, 'clip_grad', 1.0)

        print(f"\n{'=' * 60}")
        print(f"WESAD Trainer Configuration:")
        print(f"  - Task: {self.task_name}")
        print(f"  - Num classes: {self.num_classes}")
        print(f"  - Backbone: {self.backbone}")
        print(f"  - Train data ratio: {self.train_data_ratio * 100:.1f}%")  # ✨ NEW
        print(f"  - Sampling rate: {self.SAMPLING_RATE}Hz")
        print(f"  - Window size: {self.WINDOW_SIZE} samples ({self.WINDOW_SIZE / self.SAMPLING_RATE:.1f}s)")
        print(f"  - Patch size: {self.PATCH_SIZE}")
        print(f"  - Target patches: {self.TARGET_PATCHES}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Device: {self.device}")
        print(f"  - Data path: {self.data_path}")
        print(f"{'=' * 60}\n")

    def setup_task(self):
        """根据task_name设置任务参数"""
        if self.task_name == 'binary':
            self.num_classes = 1  # BCEWithLogitsLoss
            self.data_path = os.path.join(self.args.raw_data_path, "out/binary")
        elif self.task_name == 'multiclass':
            self.num_classes = 4  # CrossEntropyLoss (4 classes)
            self.data_path = os.path.join(self.args.raw_data_path, "out/multiclass")

        # 验证数据路径存在
        train_path = os.path.join(self.data_path, "train")
        if not os.path.exists(train_path):
            raise ValueError(
                f"数据路径不存在: {train_path}\n"
                f"请先运行预处理:\n"
                f"python downstream_main.py --dataset_name wesad --stage preprocessing "
                f"--wesad_format sota --classification_type {self.task_name}"
            )

    def set_seed(self, seed=42):
        """设置随机种子"""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_data(self):
        """加载SOTA格式数据 (train/val/test目录)"""
        print(f"\n📂 Loading data from: {self.data_path}")

        data = {}
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.data_path, split)

            if not os.path.exists(split_path):
                raise ValueError(f"数据集 {split} 目录不存在: {split_path}")

            samples = np.load(os.path.join(split_path, "samples.npy"))
            labels = np.load(os.path.join(split_path, "labels.npy"))
            subjects = np.load(os.path.join(split_path, "subjects.npy"))

            # 转换为torch tensor
            data[split] = {
                'samples': torch.from_numpy(samples).float(),
                'labels': torch.from_numpy(labels).long(),
                'subjects': subjects
            }

            print(f"\n{split.upper()} Set:")
            print(f"  - Samples: {samples.shape}")
            print(f"  - Labels: {labels.shape}")
            print(f"  - Class distribution: {np.bincount(labels)}")
            print(f"  - Unique subjects: {len(np.unique(subjects))}")

        return data

    def subsample_train_data(self, data):
        """
        ✨ NEW: Subsample training data based on train_data_ratio

        Args:
            data: Dictionary containing 'train', 'val', 'test' splits

        Returns:
            Modified data dictionary with subsampled training set
        """
        if self.train_data_ratio >= 1.0:
            print(f"\n📊 Using 100% of training data")
            return data

        train_samples = data['train']['samples']
        train_labels = data['train']['labels']
        train_subjects = data['train']['subjects']

        total_samples = len(train_labels)
        n_samples = int(total_samples * self.train_data_ratio)

        # Ensure at least one sample per class (for balanced sampling)
        unique_labels = torch.unique(train_labels)
        min_samples_per_class = len(unique_labels)
        n_samples = max(n_samples, min_samples_per_class)

        print(f"\n📊 Subsampling training data:")
        print(f"   Original size: {total_samples} samples")
        print(f"   Target ratio: {self.train_data_ratio * 100:.1f}%")
        print(f"   New size: {n_samples} samples")

        # Stratified sampling to maintain class distribution
        indices_per_class = {}
        for label in unique_labels:
            indices_per_class[label.item()] = torch.where(train_labels == label)[0]

        # Calculate samples per class
        samples_per_class = {}
        for label, indices in indices_per_class.items():
            class_ratio = len(indices) / total_samples
            samples_per_class[label] = max(1, int(n_samples * class_ratio))

        # Adjust to match exact n_samples
        total_allocated = sum(samples_per_class.values())
        if total_allocated < n_samples:
            # Add remaining samples to largest class
            largest_class = max(samples_per_class, key=samples_per_class.get)
            samples_per_class[largest_class] += (n_samples - total_allocated)
        elif total_allocated > n_samples:
            # Remove excess from largest class
            largest_class = max(samples_per_class, key=samples_per_class.get)
            samples_per_class[largest_class] -= (total_allocated - n_samples)

        # Sample indices
        selected_indices = []
        np.random.seed(42)  # For reproducibility

        for label, n_class_samples in samples_per_class.items():
            available_indices = indices_per_class[label].numpy()
            if len(available_indices) >= n_class_samples:
                sampled = np.random.choice(available_indices, n_class_samples, replace=False)
            else:
                sampled = available_indices
            selected_indices.extend(sampled)

        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)

        # Create subsampled training data
        data['train']['samples'] = train_samples[selected_indices]
        data['train']['labels'] = train_labels[selected_indices]
        data['train']['subjects'] = train_subjects[selected_indices]

        # Print class distribution after subsampling
        new_labels = data['train']['labels'].numpy()
        print(f"   Class distribution after subsampling: {np.bincount(new_labels)}")
        print(f"   Unique subjects: {len(np.unique(data['train']['subjects']))}")

        return data

    def get_model(self):
        """获取模型"""
        num_classes = self.num_classes if self.task_name == 'multiclass' else 1

        model, use_patches = select_model(
            backbone=self.backbone,
            num_classes=num_classes,
            in_chans=self.IN_CHANS,
            pretrained=self.pretrained,
            checkpoint_path=self.checkpoint_path,
            freeze_backbone_flag=self.freeze_backbone,
            device=self.device,
            patch_size=self.PATCH_SIZE,
            input_size=self.WINDOW_SIZE
        )

        self.use_patches = use_patches

        if use_patches:
            model = LabRAMAdapter(model, patch_size=self.PATCH_SIZE)
            print(f"✅ Model wrapped with LabRAMAdapter")
        else:
            print(f"✅ Model used directly (expects 3D input)")

        return model

    def train_one_epoch(self, model, train_loader, optimizer, loss_fn, lr_schedule_values, epoch, num_steps_per_epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        for batch_idx, (signal, label) in enumerate(pbar):
            # 更新学习率
            global_step = batch_idx + (epoch - 1) * num_steps_per_epoch
            if global_step < len(lr_schedule_values):
                current_lr = lr_schedule_values[global_step]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            signal = signal.to(self.device)

            if self.task_name == 'binary':
                label = label.float().view(-1, 1).to(self.device)
            else:
                label = label.long().to(self.device)

            logits = model(signal)
            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)

            optimizer.step()

            total_loss += loss.item()

            # 收集预测
            if self.task_name == 'binary':
                all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            else:
                all_preds.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_targets.append(label.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return avg_loss, all_preds, all_targets

    def evaluate(self, model, test_loader, loss_fn):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for signal, label in test_loader:
                signal = signal.to(self.device)

                if self.task_name == 'binary':
                    label = label.float().view(-1, 1).to(self.device)
                else:
                    label = label.long().to(self.device)

                logits = model(signal)
                loss = loss_fn(logits, label)

                total_loss += loss.item()

                if self.task_name == 'binary':
                    all_preds.append(torch.sigmoid(logits).cpu().numpy())
                else:
                    all_preds.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_targets.append(label.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return avg_loss, all_preds, all_targets

    def calculate_metrics_binary(self, preds, targets):
        """计算二分类指标"""
        # preds shape: (N, 1) or (N,)
        preds = preds.ravel()
        targets = targets.ravel()

        # 找最佳阈值
        precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(targets, preds)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        preds_binary = (preds >= best_threshold).astype(int)

        acc = sklearn.metrics.accuracy_score(targets, preds_binary)
        f1 = sklearn.metrics.f1_score(targets, preds_binary)
        precision = sklearn.metrics.precision_score(targets, preds_binary, zero_division=0)
        recall = sklearn.metrics.recall_score(targets, preds_binary, zero_division=0)

        try:
            auc = sklearn.metrics.roc_auc_score(targets, preds)
        except ValueError:
            auc = 0.5

        # 混淆矩阵
        cm = sklearn.metrics.confusion_matrix(targets, preds_binary)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)

        return {
            'acc': acc,
            'f1': f1,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'best_threshold': best_threshold,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }

    def calculate_metrics_multiclass(self, preds, targets):
        """计算多分类指标"""
        # preds shape: (N, num_classes)
        targets = targets.ravel()
        preds_class = np.argmax(preds, axis=1)

        acc = sklearn.metrics.accuracy_score(targets, preds_class)
        f1_macro = sklearn.metrics.f1_score(targets, preds_class, average='macro')
        f1_weighted = sklearn.metrics.f1_score(targets, preds_class, average='weighted')

        # 统一使用 f1 命名(weighted)
        f1 = f1_weighted

        precision = sklearn.metrics.precision_score(targets, preds_class, average='weighted', zero_division=0)
        recall = sklearn.metrics.recall_score(targets, preds_class, average='weighted', zero_division=0)

        # 计算 AUC (one-vs-rest multiclass)
        try:
            auc = sklearn.metrics.roc_auc_score(targets, preds, multi_class='ovr', average='weighted')
        except (ValueError, IndexError):
            auc = 0.0

        # 计算每个类别的指标
        per_class_f1 = sklearn.metrics.f1_score(targets, preds_class, average=None)

        # 混淆矩阵
        cm = sklearn.metrics.confusion_matrix(targets, preds_class)

        return {
            'acc': acc,
            'f1': f1,
            'auc': auc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'per_class_f1': per_class_f1,
            'confusion_matrix': cm
        }

    def train(self):
        """主训练流程"""
        self.set_seed(42)

        # 加载数据
        data = self.load_data()

        # ✨ NEW: Subsample training data based on ratio
        data = self.subsample_train_data(data)

        # 检查数据长度
        sample_length = data['train']['samples'].shape[1]
        if sample_length != self.WINDOW_SIZE:
            print(f"⚠️ 警告: 数据长度({sample_length})与期望窗口大小({self.WINDOW_SIZE})不匹配")

            if sample_length < self.WINDOW_SIZE:
                # 填充
                for split in data:
                    pad_size = self.WINDOW_SIZE - sample_length
                    data[split]['samples'] = F.pad(
                        data[split]['samples'].unsqueeze(1),
                        (0, pad_size)
                    ).squeeze(1)
                print(f"   ✅ 已填充到: {self.WINDOW_SIZE}")
            else:
                # 截断
                for split in data:
                    data[split]['samples'] = data[split]['samples'][:, :self.WINDOW_SIZE]
                print(f"   ✅ 已截断到: {self.WINDOW_SIZE}")

        # 处理NaN值
        for split in data:
            data[split]['samples'] = torch.nan_to_num(data[split]['samples'], 0)

        # 创建数据集
        if self.task_name == 'binary':
            DatasetClass = Dataset_train
        else:
            DatasetClass = Dataset_multiclass

        train_dataset = DatasetClass(
            data['train']['samples'],
            data['train']['labels'],
            mode='train',
            target_len=self.WINDOW_SIZE,
            end_idx=self.WINDOW_SIZE
        )

        val_dataset = DatasetClass(
            data['val']['samples'],
            data['val']['labels'],
            mode='eval',
            target_len=self.WINDOW_SIZE,
            end_idx=self.WINDOW_SIZE
        )

        test_dataset = DatasetClass(
            data['test']['samples'],
            data['test']['labels'],
            mode='eval',
            target_len=self.WINDOW_SIZE,
            end_idx=self.WINDOW_SIZE
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=getattr(self.args, 'num_workers', 4),
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, 'num_workers', 4)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=getattr(self.args, 'num_workers', 4)
        )

        print(f"\n📊 Dataset sizes:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val:   {len(val_dataset)} samples")
        print(f"   Test:  {len(test_dataset)} samples\n")

        # 创建模型
        model = self.get_model()
        model.to(self.device)

        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=getattr(self.args, 'weight_decay', 0.05)
        )

        # 损失函数
        if self.task_name == 'binary':
            # 计算类别权重
            num_pos = torch.sum(data['train']['labels'] == 1).item()
            num_neg = torch.sum(data['train']['labels'] == 0).item()
            pos_weight = (num_neg / num_pos) * 0.5 if num_pos > 0 else 1.0

            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(self.device)
            )
            print(f"📊 Binary classification - pos_weight: {pos_weight:.4f}")
        else:
            # 计算类别权重
            class_counts = np.bincount(data['train']['labels'].numpy())
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_weights)

            loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            )
            print(f"📊 Multiclass classification - class weights: {class_weights}")

        # 学习率调度
        min_lr = getattr(self.args, 'min_lr', 1e-6)
        warmup_epochs = getattr(self.args, 'warmup_epochs', 5)
        num_steps_per_epoch = len(train_loader)

        lr_schedule_values = cosine_scheduler(
            self.lr,
            min_lr,
            self.epochs,
            num_steps_per_epoch,
            warmup_epochs=warmup_epochs,
        )

        # 训练循环
        best_val_metric = 0
        best_metrics = None
        patience_counter = 0

        print(f"\n{'=' * 60}")
        print(f"开始训练...")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_loss, train_preds, train_targets = self.train_one_epoch(
                model, train_loader, optimizer, loss_fn,
                lr_schedule_values, epoch, num_steps_per_epoch
            )

            # 验证
            val_loss, val_preds, val_targets = self.evaluate(model, val_loader, loss_fn)

            # 计算指标
            if self.task_name == 'binary':
                val_metrics = self.calculate_metrics_binary(val_preds, val_targets)
                current_metric = val_metrics['f1']
                metric_name = 'F1'
            else:
                val_metrics = self.calculate_metrics_multiclass(val_preds, val_targets)
                current_metric = val_metrics['f1']
                metric_name = 'F1'

            # 早停和最佳模型保存
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_metrics = val_metrics.copy()
                patience_counter = 0

                # 保存最佳模型
                if hasattr(self.args, 'model_save_path'):
                    os.makedirs(self.args.model_save_path, exist_ok=True)
                    save_path = os.path.join(
                        self.args.model_save_path,
                        f'wesad_{self.task_name}_best.pth'
                    )
                    torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            # 打印进度
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_metrics['acc'] * 100:.2f}% | "
                      f"Val {metric_name}: {current_metric:.4f} | "
                      f"Patience: {patience_counter}/{self.patience}")

            # 早停
            if patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
                break

        # 测试最佳模型
        print(f"\n{'=' * 60}")
        print(f"最佳验证结果 (Epoch {epoch - patience_counter}):")
        print(f"{'=' * 60}")

        if self.task_name == 'binary':
            print(f"   Accuracy: {best_metrics['acc'] * 100:.2f}%")
            print(f"   F1-Score: {best_metrics['f1']:.4f}")
            print(f"   AUC: {best_metrics['auc']:.4f}")
            print(f"   Precision: {best_metrics['precision']:.4f}")
            print(f"   Recall: {best_metrics['recall']:.4f}")
            print(f"   Sensitivity (TPR): {best_metrics['sensitivity']:.4f}")
            print(f"   Specificity (TNR): {best_metrics['specificity']:.4f}")
        else:
            print(f"   Accuracy: {best_metrics['acc'] * 100:.2f}%")
            print(f"   F1-Score: {best_metrics['f1']:.4f}")
            print(f"   AUC: {best_metrics['auc']:.4f}")

        # 测试集评估
        print(f"\n{'=' * 60}")
        print(f"测试集评估:")
        print(f"{'=' * 60}")

        test_loss, test_preds, test_targets = self.evaluate(model, test_loader, loss_fn)

        if self.task_name == 'binary':
            test_metrics = self.calculate_metrics_binary(test_preds, test_targets)
            print(f"   Accuracy: {test_metrics['acc'] * 100:.2f}%")
            print(f"   F1-Score: {test_metrics['f1']:.4f}")
            print(f"   AUC: {test_metrics['auc']:.4f}")
            print(f"   Precision: {test_metrics['precision']:.4f}")
            print(f"   Recall: {test_metrics['recall']:.4f}")
            print(f"   Sensitivity: {test_metrics['sensitivity']:.4f}")
            print(f"   Specificity: {test_metrics['specificity']:.4f}")
        else:
            test_metrics = self.calculate_metrics_multiclass(test_preds, test_targets)
            print(f"   Accuracy: {test_metrics['acc'] * 100:.2f}%")
            print(f"   F1-Score: {test_metrics['f1']:.4f}")
            print(f"   AUC: {test_metrics['auc']:.4f}")

        print(f"\n{'=' * 60}\n")

        return test_metrics