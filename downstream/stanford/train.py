import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from downstream.model_select import select_model

# 适配新版 PyTorch AMP API
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

try:
    from codebook import modeling_finetune
except ImportError:
    modeling_finetune = None

from codebook.utils import cosine_scheduler
from downstream.stanford.tools import Dataset_train, get_logger

torch.set_float32_matmul_precision('high')


class LabRAMAdapter(nn.Module):
    def __init__(self, model, target_len=1250, patch_size=50, use_patches=True):
        super(LabRAMAdapter, self).__init__()
        self.model = model
        self.target_len = target_len
        self.patch_size = patch_size
        self.use_patches = use_patches

        print(f"LabRAMAdapter: use_patches={use_patches}, target_len={target_len}, patch_size={patch_size}")

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 3 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)
        elif x.ndim == 4:
            print(f"Warning: Input is already 4D: {x.shape}. Flattening to 3D...")
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W)

        # 确保输入长度正确 (1250)
        current_len = x.shape[-1]
        if current_len != self.target_len:
            print(f"Warning: Input length mismatch! Expected {self.target_len}, got {current_len}. Fixing...")
            if current_len < self.target_len:
                x = F.pad(x, (0, self.target_len - current_len))
            else:
                x = x[..., :self.target_len]

        # 🔧 关键修复：只有需要 patches 的模型（如 Transformer）才 reshape 成 4D
        if self.use_patches:
            # Reshape for Transformer: [B, C, N, T]
            # 1250 / 50 = 25 patches
            x = rearrange(x, 'b c (n t) -> b c n t', t=self.patch_size)

            try:
                return self.model(x, input_chans=[0, 1])
            except TypeError:
                return self.model(x)
        else:
            # 🔧 对于 Conv1D 模型（如 AnyPPG），保持 3D 输入
            # x 已经是 [B, C, L] 的形状，直接传给模型
            return self.model(x)


class StanfordTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_dir = os.path.join(args.raw_data_path, "out/quality_50hz")

        self.batch_size = args.batch_size
        self.backbone = args.backbone
        self.task_name = 'quality'

        if self.backbone == 'gpt':
            self.target_len = 1240
        else:
            self.target_len = 1250

        self.patch_size = 50 # 40  # 1250 / 50 = 25 patches

        # Signal quality 分类任务：3个类别
        self.out_dim = 3
        self.IN_CHANS = 1
        self.pretrained = getattr(args, 'pretrained', True)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)

        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler('cuda')

        self.amp_dtype = torch.bfloat16 if (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        print(f"\n{'=' * 60}")
        print(f"Stanford Signal Quality Trainer Configuration:")
        print(f"  - Task: Signal Quality (3-class)")
        print(f"  - Backbone: {self.backbone}")
        print(f"  - Sampling rate: 50Hz")
        print(f"  - Window size: {self.target_len} samples (25s)")
        print(f"  - Patch size: {self.patch_size}")
        print(f"  - Target patches: {self.target_len // self.patch_size}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Device: {self.device}")
        print(f"  - AMP dtype: {self.amp_dtype}")
        print(f"{'=' * 60}\n")

    def get_model(self):
        num_classes = self.out_dim

        model, use_patches = select_model(
            backbone=self.args.backbone,
            num_classes=num_classes,
            in_chans=self.IN_CHANS,
            pretrained=self.pretrained,
            checkpoint_path=getattr(self.args, 'checkpoint_path', None),
            freeze_backbone_flag=self.freeze_backbone,
            device=self.device,
            patch_size=self.patch_size,
            input_size=self.target_len
        )

        self.use_patches = use_patches
        return model

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

    def calculate_metrics(self, preds, targets, probs):
        """计算评估指标：F1, Acc, AUROC"""
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='macro')

        # 计算 AUROC（多分类）
        try:
            # 将标签二值化
            targets_bin = label_binarize(targets, classes=[0, 1, 2])
            auroc = roc_auc_score(targets_bin, probs, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: AUROC calculation failed: {e}")
            auroc = 0.0

        return {
            'acc': acc,
            'f1': f1,
            'auroc': auroc
        }

    def train_one_epoch(self, model, train_loader, optimizer, criterion, lr_schedule_values,
                        epoch, num_steps_per_epoch):
        """训练一个 epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for b_idx, (signal, label) in enumerate(train_loader):
            # 更新学习率
            global_step = b_idx + (epoch - 1) * num_steps_per_epoch
            if global_step < len(lr_schedule_values):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule_values[global_step]

            signal = signal.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast('cuda', dtype=self.amp_dtype):
                    logits = model(signal)
                    loss = criterion(logits, label)

                if torch.isnan(loss):
                    print(f"WARNING: Loss is NaN at step {global_step}. Skipping.")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = model(signal)
                loss = criterion(logits, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            # 收集预测结果
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(label.cpu())

        avg_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        return avg_loss, all_preds, all_targets

    def evaluate(self, model, data_loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for signal, label in data_loader:
                signal = signal.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast('cuda', dtype=self.amp_dtype):
                        logits = model(signal)
                        loss = criterion(logits, label)
                else:
                    logits = model(signal)
                    loss = criterion(logits, label)

                total_loss += loss.item()

                # 获取预测和概率
                probs = F.softmax(logits.float(), dim=1)
                preds = torch.argmax(logits.float(), dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_targets.append(label.cpu())

        avg_loss = total_loss / len(data_loader)
        all_probs = torch.cat(all_probs).float().numpy()
        all_preds = torch.cat(all_preds).float().numpy()
        all_targets = torch.cat(all_targets).numpy()

        return avg_loss, all_preds, all_targets, all_probs

    def training(self):
        """五折交叉验证训练"""
        self.set_seed(42)

        print(f"\n{'=' * 60}")
        print(f"Loading data for 5-Fold Cross Validation...")
        print(f"{'=' * 60}\n")

        # 加载所有数据（train + val + test）
        all_x = []
        all_y = []

        for split in ['train', 'val', 'test']:
            path_x = os.path.join(self.base_dir, f"{split}_x.npy")
            path_y = os.path.join(self.base_dir, f"{split}_y_qa.npy")

            if os.path.exists(path_x) and os.path.exists(path_y):
                x = np.load(path_x)
                y = np.load(path_y)
                all_x.append(x)
                all_y.append(y)
                print(f"Loaded {split}: {x.shape}")
            else:
                print(f"Warning: {split} data not found")

        # 合并所有数据
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        print(f"\nTotal data shape: {all_x.shape}")
        print(f"Class distribution: {np.bincount(all_y)}")

        # 五折交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # 存储每折的结果
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(all_x), 1):
            print(f"\n{'=' * 60}")
            print(f"Fold {fold}/5")
            print(f"{'=' * 60}\n")

            # 分割数据
            train_x = all_x[train_idx]
            train_y = all_y[train_idx]
            test_x = all_x[test_idx]
            test_y = all_y[test_idx]

            print(f"Train: {train_x.shape}, Test: {test_x.shape}")
            print(f"Train class distribution: {np.bincount(train_y)}")
            print(f"Test class distribution: {np.bincount(test_y)}")

            # 计算类别权重
            class_counts = np.bincount(train_y)
            total_samples = len(train_y)
            weights = total_samples / (len(class_counts) * class_counts.astype(np.float32))
            weights = torch.tensor(weights / weights.mean(), dtype=torch.float32).to(self.device)
            print(f"Class weights: {weights}\n")

            # 创建数据集
            train_set = Dataset_train(train_x, train_y, mode='train')
            test_set = Dataset_train(test_x, test_y, mode='eval')

            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=32,
                drop_last=True,
                pin_memory=True
            )

            test_loader = DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=32,
                pin_memory=True
            )

            # 创建模型
            model = self.get_model()

            # 🔧 关键修复：将 use_patches 传递给 LabRAMAdapter
            model = LabRAMAdapter(
                model,
                target_len=self.target_len,
                patch_size=self.patch_size,
                use_patches=self.use_patches  # 🔧 新增
            )
            model.to(self.device)

            # 优化器和损失函数
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.args.lr,
                weight_decay=getattr(self.args, 'weight_decay', 0.05)
            )

            criterion = nn.CrossEntropyLoss(weight=weights)

            # 学习率调度
            epochs = self.args.epochs
            num_steps_per_epoch = len(train_loader)
            min_lr = getattr(self.args, 'min_lr', 1e-6)
            warmup_epochs = getattr(self.args, 'warmup_epochs', 5)

            lr_schedule_values = cosine_scheduler(
                self.args.lr, min_lr, epochs, num_steps_per_epoch, warmup_epochs=warmup_epochs
            )

            # 训练循环
            best_auroc = 0.0
            best_metrics = None
            patience_counter = 0
            patience = getattr(self.args, 'patience', 10)

            print(f"Starting training for Fold {fold}...")

            for epoch in range(1, epochs + 1):
                # 训练
                train_loss, train_preds, train_targets = self.train_one_epoch(
                    model, train_loader, optimizer, criterion,
                    lr_schedule_values, epoch, num_steps_per_epoch
                )

                # 测试
                test_loss, test_preds, test_targets, test_probs = self.evaluate(
                    model, test_loader, criterion
                )

                # 计算指标
                test_metrics = self.calculate_metrics(test_preds, test_targets, test_probs)

                # 保存最佳模型（基于 AUROC）
                if test_metrics['auroc'] > best_auroc:
                    best_auroc = test_metrics['auroc']
                    best_metrics = test_metrics.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 打印进度
                if epoch % 5 == 0 or epoch == 1:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Test Loss: {test_loss:.4f} | "
                          f"Test Acc: {test_metrics['acc'] * 100:.2f}% | "
                          f"Test F1: {test_metrics['f1']:.4f} | "
                          f"Test AUROC: {test_metrics['auroc']:.4f} | "
                          f"Best AUROC: {best_auroc:.4f}")

                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            # 记录本折最佳结果
            fold_results.append(best_metrics)

            print(f"\n{'=' * 60}")
            print(f"Fold {fold} Best Results (based on AUROC):")
            print(f"{'=' * 60}")
            print(f"  Accuracy: {best_metrics['acc'] * 100:.2f}%")
            print(f"  F1-Score: {best_metrics['f1']:.4f}")
            print(f"  AUROC: {best_metrics['auroc']:.4f}")
            print(f"{'=' * 60}\n")

        # 计算并打印五折平均结果
        print(f"\n{'=' * 60}")
        print(f"5-Fold Cross Validation Results:")
        print(f"{'=' * 60}\n")

        avg_acc = np.mean([r['acc'] for r in fold_results])
        std_acc = np.std([r['acc'] for r in fold_results])
        avg_f1 = np.mean([r['f1'] for r in fold_results])
        std_f1 = np.std([r['f1'] for r in fold_results])
        avg_auroc = np.mean([r['auroc'] for r in fold_results])
        std_auroc = np.std([r['auroc'] for r in fold_results])

        print(f"Average Accuracy: {avg_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
        print(f"Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Average AUROC: {avg_auroc:.4f} ± {std_auroc:.4f}")

        print(f"\n{'=' * 60}")
        print(f"Individual Fold Results:")
        print(f"{'=' * 60}")
        for i, metrics in enumerate(fold_results, 1):
            print(f"Fold {i}: Acc={metrics['acc'] * 100:.2f}%, "
                  f"F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}")
        print(f"{'=' * 60}\n")

        return {
            'avg_acc': avg_acc,
            'std_acc': std_acc,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'avg_auroc': avg_auroc,
            'std_auroc': std_auroc,
            'fold_results': fold_results
        }