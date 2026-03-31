import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from sklearn.model_selection import StratifiedKFold

from codebook.optim_factory import create_optimizer
from codebook.utils import NativeScalerWithGradNormCount, cosine_scheduler
from downstream.model_select import select_model
from einops import rearrange

from downstream.real_world.rw_tools import (
    load_humanid_data,
    HumanIDDataset,
    calculate_classification_metrics,
    print_classification_report,
    AugmentationConfigs
)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        num_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        
        # One-hot encoding
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        
        # Label smoothing
        targets = (1 - self.smoothing) * targets + self.smoothing / num_classes
        
        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss


def mixup_data(x, y, alpha=0.2, num_classes=35, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失计算（分类版本）"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SimpleOptimizer:
    """简化的优化器工厂（如果没有codebook模块）"""
    @staticmethod
    def create(args, model):
        if args.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")


class SimpleScaler:
    """简化的梯度缩放器（如果没有codebook模块）"""
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
    
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None):
        self.scaler.scale(loss).backward()
        
        if clip_grad is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        
        self.scaler.step(optimizer)
        self.scaler.update()


def cosine_scheduler_simple(base_value, final_value, epochs, niter_per_ep, warmup_epochs=5):
    """简化的余弦学习率调度器"""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs * niter_per_ep)
    
    iters = np.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


class HumanIDTrainer:
    """
    人体识别训练器
    支持5折交叉验证和完整的分类评估
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cuda')
        self.output_dir = args.model_save_path
        self.data_dir = os.path.join(args.seg_save_path, "humanid_processed")
        os.makedirs(self.output_dir, exist_ok=True)
        self.backbone = args.backbone
        
        # 分类任务参数
        self.num_classes = 35  # 35位受试者
        
        # Trick配置
        self.use_tta = getattr(args, 'use_tta', True)
        self.use_mixup = getattr(args, 'use_mixup', True)
        self.use_swa = getattr(args, 'use_swa', True)
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.2)
        self.n_tta = getattr(args, 'n_tta', 5)
        self.k_folds = 5
        
        # 数据增强配置
        aug_level = getattr(args, 'augmentation_level', 'medium')
        if aug_level == 'light':
            self.aug_config = AugmentationConfigs.LIGHT
        elif aug_level == 'strong':
            self.aug_config = AugmentationConfigs.STRONG
        else:
            self.aug_config = AugmentationConfigs.MEDIUM
        
        print(f"\n{'='*70}")
        print(f"Human Identification Trainer - 5-Fold Cross Validation")
        print(f"{'='*70}")
        print(f"Task: 35-Class Classification (Human ID)")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"\n🎯 Active Features:")
        print(f"  ✓ Label Smoothing Cross Entropy")
        print(f"  ✓ Data Augmentation: {aug_level.upper()}")
        if self.use_tta:
            print(f"  ✓ Test-Time Augmentation (n={self.n_tta})")
        if self.use_mixup:
            print(f"  ✓ MixUp (alpha={self.mixup_alpha})")
        if self.use_swa:
            print(f"  ✓ Stochastic Weight Averaging")
        print(f"  ✓ 5-Fold Stratified Cross Validation")
        print(f"{'='*70}\n")
        
        # 数据配置
        self.IN_CHANS = 1

        if self.backbone == 'gpt':
            self.TARGET_LEN = 280  # Crop input to 1248 for GPT
        else:
            self.TARGET_LEN = 300

        self.PATCH_SIZE = getattr(args, 'patch_size', 100)  # 可调整
        self.TARGET_PATCHES = self.TARGET_LEN // self.PATCH_SIZE
    
    def get_model(self):
        """
        获取模型
        这里需要根据实际的模型选择模块调整
        """
        # 如果有model_select模块
        try:
            from downstream.model_select import select_model
            model, use_patches = select_model(
                backbone=self.args.backbone,
                num_classes=self.num_classes,  # 35类分类
                in_chans=self.IN_CHANS,
                pretrained=getattr(self.args, 'pretrained', False),
                checkpoint_path=getattr(self.args, 'checkpoint_path', None),
                freeze_backbone_flag=getattr(self.args, 'freeze_backbone', False),
                device=self.device,
                patch_size=self.PATCH_SIZE,
                input_size=self.TARGET_LEN
            )
            self.use_patches = use_patches
            return model
        except ImportError:
            print("Warning: model_select not found, using simple CNN model")
            return self.get_simple_cnn_model()

    def get_simple_cnn_model(self):
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=35):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
                self.bn1 = nn.BatchNorm1d(32)
                self.pool1 = nn.MaxPool1d(2)

                self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(64)
                self.pool2 = nn.MaxPool1d(2)

                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm1d(128)
                self.pool3 = nn.MaxPool1d(2)

                # Global average pooling
                self.gap = nn.AdaptiveAvgPool1d(1)

                # Classifier
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                # x: (B, 1, 300)
                if x.dim() == 4:  # (B, C, N, T) - patch based
                    B, C, N, T = x.shape
                    x = x.reshape(B, C, N * T)

                x = F.relu(self.bn1(self.conv1(x)))
                x = self.pool1(x)

                x = F.relu(self.bn2(self.conv2(x)))
                x = self.pool2(x)

                x = F.relu(self.bn3(self.conv3(x)))
                x = self.pool3(x)

                x = self.gap(x).squeeze(-1)
                x = self.fc(x)

                return x

        self.use_patches = False
        return SimpleCNN(num_classes=self.num_classes).to(self.device)
    
    def training(self):
        """主训练函数 - 5折交叉验证"""
        args = self.args
        
        print(f"\n{'='*70}")
        print(f"Starting 5-Fold Cross Validation: Human Identification")
        print(f"{'='*70}\n")
        
        # 1. 加载所有数据
        X_all, y_all, subject_info = load_humanid_data(self.data_dir)
        
        print(f"Total Samples for CV: {len(X_all)}")
        print(f"Class Distribution:")
        unique, counts = np.unique(y_all, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls}: {cnt} samples")

        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        # 存储每折的结果
        fold_results = []
        
        # 3. 开始 5 折循环
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            print(f"\n{'='*30} Fold {fold + 1}/{self.k_folds} {'='*30}")
            
            # 准备当前折的数据
            X_train_fold, y_train_fold = X_all[train_idx], y_all[train_idx]
            X_val_fold, y_val_fold = X_all[val_idx], y_all[val_idx]
            
            print(f"  Train: {len(X_train_fold)} samples")
            print(f"  Val:   {len(X_val_fold)} samples")
            
            # 创建数据集
            train_ds = HumanIDDataset(
                X_train_fold, y_train_fold, 
                mode='train', 
                aug_config=self.aug_config
            )
            val_ds = HumanIDDataset(
                X_val_fold, y_val_fold, 
                mode='test'
            )
            
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=getattr(args, 'num_workers', 8),
                pin_memory=True,
                drop_last=True
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=getattr(args, 'num_workers', 4)
            )
            
            # 初始化模型和训练组件
            model = self.get_model()
            
            # 损失函数：标签平滑交叉熵
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            
            # 优化器
            try:
                from codebook.optim_factory import create_optimizer
                optimizer = create_optimizer(args, model)
            except ImportError:
                optimizer = SimpleOptimizer.create(args, model)
            
            # 梯度缩放器
            try:
                from codebook.utils import NativeScalerWithGradNormCount
                loss_scaler = NativeScalerWithGradNormCount()
            except ImportError:
                loss_scaler = SimpleScaler()
            
            # SWA设置
            swa_model = None
            swa_scheduler = None
            swa_start = int(args.epochs * 0.75)
            if self.use_swa:
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=args.min_lr)
            
            # 学习率调度
            warmup_epochs = getattr(args, 'warmup_epochs', 5)
            try:
                from codebook.utils import cosine_scheduler
                lr_schedule_values = cosine_scheduler(
                    args.lr, args.min_lr, args.epochs, len(train_loader),
                    warmup_epochs=warmup_epochs,
                )
            except ImportError:
                lr_schedule_values = cosine_scheduler_simple(
                    args.lr, args.min_lr, args.epochs, len(train_loader),
                    warmup_epochs=warmup_epochs
                )
            
            best_accuracy = 0.0
            best_f1 = 0.0
            best_metrics = {}
            
            # 训练循环
            for epoch in range(args.epochs):
                # 训练
                train_loss, train_acc = self.train_one_epoch(
                    model, train_loader, criterion, optimizer,
                    loss_scaler, lr_schedule_values, epoch
                )
                
                # SWA更新
                if self.use_swa and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    if swa_scheduler is not None:
                        swa_scheduler.step()
                
                # 验证
                if self.use_swa and epoch >= swa_start:
                    eval_model = swa_model
                    model_type = "SWA"
                else:
                    eval_model = model
                    model_type = "Regular"
                
                if self.use_tta:
                    val_metrics = self.evaluate_with_tta(eval_model, val_loader)
                else:
                    val_metrics = self.evaluate(eval_model, val_loader)
                
                # 打印进度
                if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                    print(f"  [Fold {fold+1}] Epoch {epoch+1}/{args.epochs} ({model_type}) | "
                          f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_metrics['accuracy']:.4f} | "
                          f"F1: {val_metrics['f1_macro']:.4f} | "
                          f"AUC: {val_metrics['auc_macro']:.4f}")
                
                # 保存最佳模型
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    best_f1 = val_metrics['f1_macro']
                    best_metrics = val_metrics.copy()
                    
                    # 保存模型
                    if self.use_swa and epoch >= swa_start:
                        save_model = swa_model.module
                    else:
                        save_model = model
                    
                    save_dict = {
                        'model': save_model.state_dict(),
                        'metrics': best_metrics,
                        'epoch': epoch,
                        'fold': fold + 1,
                        'args': vars(args)
                    }
                    
                    save_path = os.path.join(
                        self.output_dir,
                        f"humanid_fold{fold+1}_best.pth"
                    )
                    torch.save(save_dict, save_path)
            
            print(f"\n✅ Fold {fold+1} Best Results:")
            print(f"   Accuracy: {best_accuracy:.4f}")
            print(f"   F1-Score: {best_f1:.4f}")
            print(f"   AUC:      {best_metrics['auc_macro']:.4f}")
            
            fold_results.append(best_metrics)
        
        # 汇总5折结果
        print(f"\n{'='*70}")
        print(f"5-Fold Cross Validation Complete!")
        print(f"{'='*70}")
        
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        std_acc = np.std([r['accuracy'] for r in fold_results])
        avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
        std_f1 = np.std([r['f1_macro'] for r in fold_results])
        avg_auc = np.mean([r['auc_macro'] for r in fold_results])
        std_auc = np.std([r['auc_macro'] for r in fold_results])
        
        print(f"\nAverage Results over {self.k_folds} folds:")
        print(f"  Accuracy:  {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"  F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"  AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
        
        print(f"\nPer-Fold Details:")
        print("-" * 70)
        for i, res in enumerate(fold_results):
            print(f"  Fold {i+1}: Acc={res['accuracy']:.4f}, "
                  f"F1={res['f1_macro']:.4f}, "
                  f"AUC={res['auc_macro']:.4f}")
        print(f"{'='*70}\n")
        
        # 保存汇总结果
        summary = {
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'fold_results': fold_results
        }
        
        summary_path = os.path.join(self.output_dir, 'cv_summary.npy')
        np.save(summary_path, summary)
        print(f"Summary saved to: {summary_path}")
    
    def train_one_epoch(self, model, loader, criterion, optimizer,
                       loss_scaler, lr_schedule_values, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for step, (x, y) in enumerate(loader):
            # 更新学习率
            it = step + epoch * len(loader)
            if it < len(lr_schedule_values):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule_values[it]
            
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            # MixUp
            use_mixup = self.use_mixup and np.random.rand() < 0.5
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(
                    x, y, alpha=self.mixup_alpha, 
                    num_classes=self.num_classes, 
                    device=self.device
                )

            if x.shape[-1] > self.TARGET_LEN:
                x = x[..., :self.TARGET_LEN]

            # Patch reshape（如果需要）
            if self.use_patches and x.shape[-1] == self.TARGET_LEN:
                try:
                    from einops import rearrange
                    x = rearrange(x, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)
                except ImportError:
                    # 手动reshape
                    B, C, L = x.shape
                    N = L // self.PATCH_SIZE
                    x = x[:, :, :N * self.PATCH_SIZE].reshape(B, C, N, self.PATCH_SIZE)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                pred = model(x)
                if use_mixup:
                    loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
                else:
                    loss = criterion(pred, y)
            
            # 反向传播
            optimizer.zero_grad()
            
            if hasattr(loss_scaler, '__call__'):
                loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters())
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率（仅在非MixUp时）
            if not use_mixup:
                _, predicted = torch.max(pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, model, loader):
        """标准评估"""
        model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        for x, y in loader:
            x = x.to(self.device)

            if x.shape[-1] > self.TARGET_LEN:
                x = x[..., :self.TARGET_LEN]
            
            # Patch reshape
            if self.use_patches and x.shape[-1] == self.TARGET_LEN:
                try:
                    from einops import rearrange
                    x = rearrange(x, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)
                except ImportError:
                    B, C, L = x.shape
                    N = L // self.PATCH_SIZE
                    x = x[:, :, :N * self.PATCH_SIZE].reshape(B, C, N, self.PATCH_SIZE)
            
            with torch.cuda.amp.autocast():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
            
            _, pred = torch.max(logits, 1)
            
            all_preds.append(pred.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.numpy())
        
        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        
        metrics = calculate_classification_metrics(
            targets, preds, probs, num_classes=self.num_classes
        )
        
        return metrics
    
    @torch.no_grad()
    def evaluate_with_tta(self, model, loader):
        """测试时增强评估"""
        model.eval()
        all_probs_tta = []
        all_targets = []
        
        for x, y in loader:
            x = x.to(self.device)
            
            # 多次预测并平均
            tta_probs = []
            for tta_idx in range(self.n_tta):
                x_aug = x.clone()
                
                # 轻微增强
                scale = 0.98 + torch.rand(1).to(self.device) * 0.04
                x_aug = x_aug * scale
                
                if torch.rand(1) > 0.5:
                    noise = torch.randn_like(x_aug) * 0.02
                    x_aug += noise

                if x_aug.shape[-1] > self.TARGET_LEN:
                    x_aug = x_aug[..., :self.TARGET_LEN]
                
                # Patch reshape
                if self.use_patches and x_aug.shape[-1] == self.TARGET_LEN:
                    try:
                        from einops import rearrange
                        x_aug = rearrange(x_aug, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)
                    except ImportError:
                        B, C, L = x_aug.shape
                        N = L // self.PATCH_SIZE
                        x_aug = x_aug[:, :, :N * self.PATCH_SIZE].reshape(B, C, N, self.PATCH_SIZE)
                
                with torch.cuda.amp.autocast():
                    logits = model(x_aug)
                    probs = F.softmax(logits, dim=-1)
                
                tta_probs.append(probs.cpu().numpy())
            
            # 平均概率
            avg_probs = np.mean(tta_probs, axis=0)
            all_probs_tta.append(avg_probs)
            all_targets.append(y.numpy())
        
        probs = np.concatenate(all_probs_tta)
        targets = np.concatenate(all_targets)
        preds = np.argmax(probs, axis=1)
        
        metrics = calculate_classification_metrics(
            targets, preds, probs, num_classes=self.num_classes
        )
        
        return metrics


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser('Human Identification Training')
    
    # 数据路径
    parser.add_argument('--seg_save_path', type=str, default='./data',
                       help='Path to processed data')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints',
                       help='Path to save models')
    
    # 模型配置
    parser.add_argument('--backbone', type=str, default='resnet1d',
                       help='Model backbone')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adamw')
    
    # 增强和tricks
    parser.add_argument('--augmentation_level', type=str, default='medium',
                       choices=['light', 'medium', 'strong'])
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--use_swa', action='store_true', default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--n_tta', type=int, default=5)
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=25)
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = HumanIDTrainer(args)
    trainer.training()
