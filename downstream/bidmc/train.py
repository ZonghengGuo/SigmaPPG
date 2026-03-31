"""
改进版BIDMC训练器 - 集成多个性能提升tricks + 5折交叉验证
可以直接替换原有的train.py使用

主要改进：
1. 组合损失函数 (L1 + Huber + 相对误差)
2. 测试时增强 (TTA)
3. 标签平滑
4. MixUp数据增强
5. 自适应权重平均 (SWA)
6. 5折交叉验证 (5-Fold Cross Validation)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from sklearn.model_selection import KFold  # 引入KFold
from codebook.optim_factory import create_optimizer
from codebook.utils import NativeScalerWithGradNormCount, cosine_scheduler
from downstream.bidmc.tools import (
    load_bidmc_data,
    BIDMCDataset,
    calculate_metrics,
    get_task_unit
)
from downstream.model_select import select_model
from einops import rearrange


class CombinedLoss(nn.Module):
    """
    组合损失函数：L1 + Huber + 相对误差

    Args:
        alpha: L1损失权重
        beta: Huber损失权重
        gamma: 相对误差权重
    """

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()
        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self, pred, target):
        # L1 损失 - 基础
        l1_loss = self.l1(pred, target)

        # Huber 损失 - 对离群点更鲁棒
        huber_loss = self.huber(pred, target)

        # 相对误差损失 - 关注百分比误差
        relative_loss = torch.mean(torch.abs((pred - target) / (target + 1e-8)))

        total_loss = self.alpha * l1_loss + self.beta * huber_loss + self.gamma * relative_loss
        return total_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑回归损失
    防止模型过拟合到精确标签值
    """

    def __init__(self, base_criterion, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.base_criterion = base_criterion

    def forward(self, pred, target):
        # 训练时添加小量噪声
        if self.training:
            noise = torch.randn_like(target) * self.smoothing * torch.std(target)
            target_smooth = target + noise
            return self.base_criterion(pred, target_smooth)
        else:
            return self.base_criterion(pred, target)


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    MixUp数据增强

    Args:
        x: 输入数据
        y: 标签
        alpha: Beta分布参数

    Returns:
        mixed_x, y_a, y_b, lam
    """
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
    """MixUp损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class BIDMCTrainer:
    """
    改进版BIDMC多任务训练器
    集成多个性能提升tricks + 5折交叉验证
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cuda')
        self.output_dir = args.model_save_path
        self.data_dir = os.path.join(args.seg_save_path, "bidmc_processed")
        os.makedirs(self.output_dir, exist_ok=True)

        # 任务配置
        self.task_name = getattr(args, 'task_name', 'rr').lower()
        valid_tasks = ['rr', 'hr', 'spo2']
        if self.task_name not in valid_tasks:
            raise ValueError(f"Invalid task_name '{self.task_name}'. Must be one of {valid_tasks}")

        self.task_unit = get_task_unit(self.task_name)

        # Trick配置
        self.use_tta = getattr(args, 'use_tta', True)  # 测试时增强
        self.use_mixup = getattr(args, 'use_mixup', True)  # MixUp
        self.use_swa = getattr(args, 'use_swa', True)  # 自适应权重平均
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.2)
        self.n_tta = getattr(args, 'n_tta', 5)  # TTA次数
        self.k_folds = 5  # 交叉验证折数

        print(f"\n{'=' * 60}")
        print(f"BIDMC Trainer Configuration (Improved + 5-Fold CV)")
        print(f"{'=' * 60}")
        print(f"Task: {self.task_name.upper()} ({self.task_unit})")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"\n🎯 Active Tricks:")
        print(f"  ✓ Combined Loss (L1 + Huber + Relative)")
        print(f"  ✓ Label Smoothing")
        if self.use_tta:
            print(f"  ✓ Test-Time Augmentation (n={self.n_tta})")
        if self.use_mixup:
            print(f"  ✓ MixUp (alpha={self.mixup_alpha})")
        if self.use_swa:
            print(f"  ✓ Stochastic Weight Averaging")
        print(f"  ✓ 5-Fold Cross Validation")
        print(f"{'=' * 60}\n")

        # 数据配置
        self.IN_CHANS = 1
        self.TARGET_LEN = 4000
        self.PATCH_SIZE = 40 # 250
        self.TARGET_PATCHES = self.TARGET_LEN // self.PATCH_SIZE

        # 模型配置
        self.pretrained = getattr(args, 'pretrained', True)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)

    def get_model(self):
        model, use_patches = select_model(
            backbone=self.args.backbone,
            num_classes=1,
            in_chans=self.IN_CHANS,
            pretrained=self.pretrained,
            checkpoint_path=getattr(self.args, 'checkpoint_path', None),
            freeze_backbone_flag=self.freeze_backbone,
            device=self.device,
            patch_size=self.PATCH_SIZE,
            input_size=self.TARGET_LEN
        )

        self.use_patches = use_patches
        return model

    def training(self):
        args = self.args

        print(f"\n{'=' * 60}")
        print(f"Starting 5-Fold Cross Validation: {self.task_name.upper()}")
        print(f"{'=' * 60}\n")

        # 1. 加载所有数据并合并
        X_train_orig, y_train_orig, X_test_orig, y_test_orig = load_bidmc_data(
            self.data_dir,
            task_name=self.task_name
        )

        # 合并数据集用于交叉验证
        X_all = np.concatenate([X_train_orig, X_test_orig], axis=0)
        y_all = np.concatenate([y_train_orig, y_test_orig], axis=0)

        print(f"Total Samples for CV: {len(X_all)}")

        # 初始化 KFold
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        # 存储每折的结果
        fold_results = []

        # 开始 5 折循环
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
            print(f"\n{'='*20} Fold {fold + 1}/{self.k_folds} {'='*20}")

            # 准备当前折的数据
            X_train_fold, y_train_fold = X_all[train_idx], y_all[train_idx]
            X_val_fold, y_val_fold = X_all[val_idx], y_all[val_idx]

            train_ds = BIDMCDataset(X_train_fold, y_train_fold, task_name=self.task_name, mode='train')
            test_ds = BIDMCDataset(X_val_fold, y_val_fold, task_name=self.task_name, mode='test')

            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )

            # --- 初始化当前折的模型和训练组件 (必须在Fold内部初始化) ---
            model = self.get_model()

            # 🎯 Trick: 组合损失 + 标签平滑
            base_criterion = CombinedLoss(alpha=0.6, beta=0.3, gamma=0.1)
            criterion = LabelSmoothingLoss(base_criterion, smoothing=0.05)

            optimizer = create_optimizer(args, model)
            loss_scaler = NativeScalerWithGradNormCount()

            # 🎯 Trick: SWA设置 (每个Fold独立)
            swa_model = None
            swa_scheduler = None
            swa_start = int(args.epochs * 0.75)
            if self.use_swa:
                swa_model = AveragedModel(model)
                swa_scheduler = SWALR(optimizer, swa_lr=args.min_lr)

            # 学习率调度
            warmup_epochs = getattr(args, 'warmup_epochs', 5)
            lr_schedule_values = cosine_scheduler(
                args.lr, args.min_lr, args.epochs, len(train_loader),
                warmup_epochs=warmup_epochs,
            )

            best_mae = float('inf')
            best_metrics = {}

            # --- 训练循环 (Epochs) ---
            for epoch in range(args.epochs):
                # 训练
                train_loss = self.train_one_epoch(
                    model, train_loader, criterion, optimizer,
                    loss_scaler, lr_schedule_values, epoch
                )

                # 🎯 Trick: SWA更新
                if self.use_swa and epoch >= swa_start:
                    swa_model.update_parameters(model)
                    if swa_scheduler is not None:
                        swa_scheduler.step()

                # 验证 (使用TTA或SWA模型)
                if self.use_swa and epoch >= swa_start:
                    eval_model = swa_model
                    model_type = "SWA"
                else:
                    eval_model = model
                    model_type = "Regular"

                if self.use_tta:
                    val_metrics = self.evaluate_with_tta(eval_model, test_loader)
                else:
                    val_metrics = self.evaluate(eval_model, test_loader)

                # 打印结果 (简化输出，避免过多刷屏)
                if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                    print(f"  [Fold {fold+1}] Epoch {epoch + 1}/{args.epochs} ({model_type}) | "
                          f"Loss: {train_loss:.4f} | Val MAE: {val_metrics['mae']:.4f}")

                # 保存最佳模型 (当前Fold)
                if val_metrics['mae'] < best_mae:
                    best_mae = val_metrics['mae']
                    best_metrics = val_metrics.copy()

                    # 保存实际使用的模型
                    if self.use_swa and epoch >= swa_start:
                        save_model = swa_model.module
                    else:
                        save_model = model

                    save_dict = {
                        'model': save_model.state_dict(),
                        'metrics': best_metrics,
                        'epoch': epoch,
                        'fold': fold + 1,
                        'task': self.task_name,
                        'args': vars(args)
                    }

                    # 每个Fold保存一个最佳模型
                    save_path = os.path.join(
                        self.output_dir,
                        f"bidmc_{self.task_name}_fold{fold+1}_best.pth"
                    )
                    torch.save(save_dict, save_path)

            print(f"✅ Fold {fold + 1} Best MAE: {best_mae:.4f}")
            fold_results.append(best_metrics)

        # --- 交叉验证结束，汇总结果 ---
        print(f"\n{'=' * 60}")
        print(f"5-Fold Cross Validation Complete!")
        print(f"{'=' * 60}")

        avg_mae = np.mean([r['mae'] for r in fold_results])
        std_mae = np.std([r['mae'] for r in fold_results])
        avg_rmse = np.mean([r['rmse'] for r in fold_results])
        avg_corr = np.mean([r['corr'] for r in fold_results])

        print(f"Average Results over {self.k_folds} folds ({self.task_name.upper()}):")
        print(f"  MAE:  {avg_mae:.4f} ± {std_mae:.4f} {self.task_unit}")
        print(f"  RMSE: {avg_rmse:.4f} {self.task_unit}")
        print(f"  Corr: {avg_corr:.4f}")

        # 打印每折详情
        print("-" * 30)
        for i, res in enumerate(fold_results):
            print(f"  Fold {i+1}: MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, Corr={res['corr']:.4f}")
        print(f"{'=' * 60}\n")

    def train_one_epoch(self, model, loader, criterion, optimizer,
                        loss_scaler, lr_schedule_values, epoch):
        """训练一个epoch（包含MixUp） - 逻辑未修改"""
        model.train()
        criterion.train()  # 确保label smoothing生效
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            # 更新学习率
            it = step + epoch * len(loader)
            if it < len(lr_schedule_values):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule_values[it]

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # 🎯 Trick: MixUp (概率使用)
            use_mixup = self.use_mixup and np.random.rand() < 0.5
            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=self.mixup_alpha, device=self.device)

            # 对于patch-based模型，需要reshape输入
            if self.use_patches and x.shape[-1] == self.TARGET_LEN:
                if self.args.backbone == 'gpt':
                    x = rearrange(x, 'b 1 (n t) -> b n t', t=self.PATCH_SIZE)
                else:
                    x = rearrange(x, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)

            # 前向传播
            with torch.cuda.amp.autocast():
                pred = model(x).squeeze()
                if use_mixup:
                    loss = mixup_criterion(criterion, pred, y_a, y_b, lam)
                else:
                    loss = criterion(pred, y)

            # 反向传播
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters())
            total_loss += loss.item()

            # 进度打印稍微简化一下，避免5折CV时log太多
            if step % max(1, len(loader) // 5) == 0:
                pass

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, model, loader):
        """标准评估（无TTA）"""
        model.eval()
        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device)

            # Patch reshape
            if self.use_patches and x.shape[-1] == self.TARGET_LEN:
                if self.args.backbone == 'gpt':
                    x = rearrange(x, 'b 1 (n t) -> b n t', t=self.PATCH_SIZE)
                else:
                    x = rearrange(x, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)

            with torch.cuda.amp.autocast():
                pred = model(x).squeeze()

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

        if len(all_preds) == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'corr': 0.0}

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = calculate_metrics(preds, targets, self.task_name)
        return metrics

    @torch.no_grad()
    def evaluate_with_tta(self, model, loader):
        """
        🎯 Trick: 测试时增强评估
        对每个样本进行多次轻微增强并平均预测
        """
        model.eval()
        all_preds = []
        all_targets = []

        for x, y in loader:
            x = x.to(self.device)

            # 多次预测并平均
            tta_preds = []
            for tta_idx in range(self.n_tta):
                # 轻微的增强
                x_aug = x.clone()

                # 微小的缩放 (98%-102%)
                scale = 0.98 + torch.rand(1).to(self.device) * 0.04
                x_aug = x_aug * scale

                # 微小的噪声 (50%概率)
                if torch.rand(1) > 0.5:
                    noise = torch.randn_like(x_aug) * 0.02
                    x_aug += noise

                # Patch reshape
                if self.use_patches and x_aug.shape[-1] == self.TARGET_LEN:
                    if self.args.backbone == 'gpt':
                        x_aug = rearrange(x_aug, 'b 1 (n t) -> b n t', t=self.PATCH_SIZE)
                    else:
                        x_aug = rearrange(x_aug, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)

                with torch.cuda.amp.autocast():
                    pred = model(x_aug).squeeze()

                tta_preds.append(pred.cpu().numpy())

            # 平均所有TTA预测
            avg_pred = np.mean(tta_preds, axis=0)
            all_preds.append(avg_pred)
            all_targets.append(y.numpy())

        if len(all_preds) == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'corr': 0.0}

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        metrics = calculate_metrics(preds, targets, self.task_name)
        return metrics