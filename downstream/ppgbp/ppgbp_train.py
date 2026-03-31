import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from codebook.optim_factory import create_optimizer
from codebook.utils import NativeScalerWithGradNormCount, cosine_scheduler
from downstream.ppgbp.ppgbp_tools import (
    load_ppgbp_kfold_data,
    PPGBPDataset,
    calculate_regression_metrics,
    calculate_classification_metrics,
    get_task_info
)
from downstream.model_select import select_model
from einops import rearrange


class PPGBPTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cuda')
        self.output_dir = args.model_save_path
        self.data_dir = args.seg_save_path
        os.makedirs(self.output_dir, exist_ok=True)

        self.task_name = getattr(args, 'task_name', 'sbp').lower()
        valid_tasks = ['sbp', 'dbp', 'hr', 'hyper']
        if self.task_name not in valid_tasks:
            raise ValueError(f"Invalid task_name '{self.task_name}'. Must be one of {valid_tasks}")

        self.task_info = get_task_info(self.task_name)
        self.is_classification = (self.task_info['type'] == 'classification')

        if hasattr(args, 'rsfreq') and args.rsfreq is not None:
            self.sampling_rate = args.rsfreq
        else:
            self.sampling_rate = 50

        self.n_folds = getattr(args, 'n_folds', 5)
        self.patience = getattr(args, 'patience', 15)
        self.noise_std = getattr(args, 'noise_std', 0.02)

        print(f"\n{'=' * 70}")
        print(f"PPGBP K-Fold Cross-Validation Trainer")
        print(f"{'=' * 70}")
        print(f"Task: {self.task_info['full_name']} ({self.task_name.upper()})")
        print(f"Task Type: {self.task_info['type'].upper()}")
        print(f"Number of Folds: {self.n_folds}")
        print(f"Sampling Rate: {self.sampling_rate} Hz")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"\nTraining Configuration:")
        print(f"  Loss: {'CrossEntropy' if self.is_classification else 'MSE'}")
        print(f"  Data Augmentation: Gaussian Noise (std={self.noise_std})")
        print(f"  Early Stopping: patience={self.patience}")
        print(f"{'=' * 70}\n")

        self.IN_CHANS = 1
        self.TARGET_LEN = 10 * self.sampling_rate


        self.PATCH_SIZE = 50

        self.TARGET_PATCHES = self.TARGET_LEN // self.PATCH_SIZE

        print(f"Signal Configuration:")
        print(f"  Length: {self.TARGET_LEN} samples ({self.TARGET_LEN / self.sampling_rate:.1f} seconds)")
        print(f"  Patch size: {self.PATCH_SIZE}")
        print(f"  Number of patches: {self.TARGET_PATCHES}")
        print(f"{'=' * 70}\n")

        self.pretrained = getattr(args, 'pretrained', True)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)

    def get_model(self):
        num_classes = 2 if self.task_name == 'hyper' else 1

        model, use_patches = select_model(
            backbone=self.args.backbone,
            num_classes=num_classes,
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

    def _get_criterion(self):
        if self.is_classification:
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()

    def training(self):
        """K折交叉验证训练主流程"""
        args = self.args

        # 存储每折的结果
        fold_results = []

        # 对每一折进行训练
        for fold_idx in range(self.n_folds):
            print(f"\n{'#' * 70}")
            print(f"#  Training Fold {fold_idx + 1}/{self.n_folds}")
            print(f"{'#' * 70}\n")

            # 训练当前fold
            fold_metrics = self.train_single_fold(fold_idx)
            fold_results.append(fold_metrics)

            # 打印当前fold的结果
            print(f"\n{'=' * 70}")
            print(f"Fold {fold_idx + 1} Results:")
            print(f"{'=' * 70}")
            if self.is_classification:
                print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
                print(f"  F1 Score: {fold_metrics['f1']:.4f}")
                print(f"  AUROC:    {fold_metrics['auroc']:.4f}")
            else:
                print(f"  MAE:  {fold_metrics['mae']:.4f} {self.task_info['unit']}")
                print(f"  RMSE: {fold_metrics['rmse']:.4f} {self.task_info['unit']}")
                print(f"  Corr: {fold_metrics['corr']:.4f}")
            print(f"{'=' * 70}\n")

        # 计算平均结果
        self.print_final_results(fold_results)

    def train_single_fold(self, fold_idx):
        """训练单个fold"""
        args = self.args

        # 加载当前fold的数据
        print(f"Loading data for fold {fold_idx}...")
        train_x, train_y, test_x, test_y = load_ppgbp_kfold_data(
            self.data_dir, fold_idx, self.task_name, self.sampling_rate
        )

        # 创建数据集
        train_dataset = PPGBPDataset(
            train_x, train_y,
            task_name=self.task_name,
            mode='train',
            noise_std=self.noise_std
        )
        test_dataset = PPGBPDataset(
            test_x, test_y,
            task_name=self.task_name,
            mode='test',
            noise_std=0
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print(f"Data loaded: Train={len(train_dataset)}, Test={len(test_dataset)}")

        # 创建模型
        print("Creating model...")
        model = self.get_model()
        model = model.to(self.device)

        # 创建优化器和调度器
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScalerWithGradNormCount()

        lr_schedule_values = cosine_scheduler(
            args.lr,
            args.min_lr,
            args.epochs,
            len(train_loader),
            warmup_epochs=args.warmup_epochs
        )

        criterion = self._get_criterion()

        # 早停相关变量
        best_metric = float('inf') if not self.is_classification else 0.0
        best_metrics = {}
        patience_counter = 0

        print(f"\nStarting training for {args.epochs} epochs...")

        # 训练循环
        for epoch in range(args.epochs):
            train_loss = self.train_one_epoch(
                model, train_loader, criterion, optimizer,
                loss_scaler, lr_schedule_values, epoch
            )

            # 在测试集上评估
            test_metrics = self.evaluate(model, test_loader)

            if self.is_classification:
                current_metric = test_metrics['accuracy']
                metric_name = 'Accuracy'
                is_better = lambda curr, best: curr > best
            else:
                current_metric = test_metrics['mae']
                metric_name = 'MAE'
                is_better = lambda curr, best: curr < best

            if self.is_classification:
                print(f"Epoch {epoch + 1}/{args.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Test Acc: {test_metrics['accuracy']:.4f} | "
                      f"F1: {test_metrics['f1']:.4f} | "
                      f"AUROC: {test_metrics['auroc']:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{args.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Test MAE: {test_metrics['mae']:.4f} | "
                      f"RMSE: {test_metrics['rmse']:.4f} | "
                      f"Corr: {test_metrics['corr']:.4f}")

            if is_better(current_metric, best_metric):
                best_metric = current_metric
                best_metrics = test_metrics.copy()
                patience_counter = 0

                save_dict = {
                    'model': model.state_dict(),
                    'metrics': best_metrics,
                    'epoch': epoch,
                    'fold': fold_idx,
                    'task': self.task_name,
                    'sampling_rate': self.sampling_rate,
                    'args': vars(args)
                }

                save_path = os.path.join(
                    self.output_dir,
                    f"ppgbp_{self.task_name}_fold{fold_idx}_best.pth"
                )
                torch.save(save_dict, save_path)
                print(f"  ✓ Saved best model (Epoch {epoch + 1}, {metric_name}={current_metric:.4f})")
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        return best_metrics

    def train_one_epoch(self, model, loader, criterion, optimizer,
                        loss_scaler, lr_schedule_values, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0

        for step, (x, y) in enumerate(loader):
            it = step + epoch * len(loader)
            if it < len(lr_schedule_values):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule_values[it]

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if self.use_patches and x.shape[-1] == self.TARGET_LEN:
                if self.args.backbone == 'gpt':
                    if x.shape[-1] == 500:
                        x = x[..., :480]

                    x = rearrange(x, 'b 1 (n t) -> b n t', t=self.PATCH_SIZE)
                else:
                    x = rearrange(x, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)

            with torch.amp.autocast(device_type='cuda'):
                pred = model(x)
                if not self.is_classification:
                    # 只squeeze最后一维，保持batch维度
                    pred = pred.squeeze(-1)

                loss = criterion(pred, y)

            optimizer.zero_grad()
            clip_grad = getattr(self.args, 'clip_grad', 1.0)
            loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, model, loader):
        """评估模型，返回所有指标"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []  # 用于计算AUROC

        for x, y in loader:
            x = x.to(self.device)

            if self.use_patches and x.shape[-1] == self.TARGET_LEN:
                if self.args.backbone == 'gpt':
                    if x.shape[-1] == 500:
                        x = x[..., :480]
                    x = rearrange(x, 'b 1 (n t) -> b n t', t=self.PATCH_SIZE)
                else:
                    x = rearrange(x, 'b c (n t) -> b c n t', t=self.PATCH_SIZE)

            with torch.amp.autocast(device_type='cuda'):
                pred = model(x)

                if self.is_classification:
                    # 保存概率用于AUROC
                    probs = F.softmax(pred, dim=1)
                    all_probs.append(probs.cpu().numpy())

                    # 保存预测类别
                    pred_class = torch.argmax(pred, dim=1)
                    all_preds.append(pred_class.cpu().numpy())
                else:
                    # 确保回归预测始终是1维的
                    pred = pred.squeeze(-1)  # 只squeeze最后一维
                    if pred.dim() == 0:  # 如果变成了标量，转换为1维数组
                        pred = pred.unsqueeze(0)
                    all_preds.append(pred.cpu().numpy())

            all_targets.append(y.cpu().numpy())  # 添加.cpu()确保一致性

        if len(all_preds) == 0:
            if self.is_classification:
                return {'accuracy': 0.0, 'f1': 0.0, 'auroc': 0.0}
            else:
                return {'mae': 0.0, 'rmse': 0.0, 'corr': 0.0}

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        if self.is_classification:
            probs = np.concatenate(all_probs)
            metrics = calculate_classification_metrics(preds, targets, probs)
        else:
            metrics = calculate_regression_metrics(preds, targets)

        return metrics

    def print_final_results(self, fold_results):
        """打印最终的K折交叉验证结果"""
        print(f"\n{'#' * 70}")
        print(f"#  {self.n_folds}-Fold Cross-Validation Results - {self.task_info['full_name']}")
        print(f"{'#' * 70}\n")

        if self.is_classification:
            # 分类任务
            accs = [r['accuracy'] for r in fold_results]
            f1s = [r['f1'] for r in fold_results]
            aurocs = [r['auroc'] for r in fold_results]

            print(f"Individual Fold Results:")
            print(f"{'Fold':<8} {'Accuracy':<12} {'F1':<12} {'AUROC':<12}")
            print(f"{'-' * 44}")
            for i, (acc, f1, auroc) in enumerate(zip(accs, f1s, aurocs)):
                print(f"{i + 1:<8} {acc:<12.4f} {f1:<12.4f} {auroc:<12.4f}")

            print(f"\n{'=' * 70}")
            print(f"Final Average Results:")
            print(f"{'=' * 70}")
            print(f"  Accuracy: {np.nanmean(accs):.4f} ± {np.nanstd(accs):.4f}")
            print(f"  F1 Score: {np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}")
            print(f"  AUROC:    {np.nanmean(aurocs):.4f} ± {np.nanstd(aurocs):.4f}")

        else:
            # 回归任务
            maes = [r['mae'] for r in fold_results]
            rmses = [r['rmse'] for r in fold_results]
            corrs = [r['corr'] for r in fold_results]

            print(f"Individual Fold Results:")
            print(f"{'Fold':<8} {'MAE':<15} {'RMSE':<15} {'Corr':<12}")
            print(f"{'-' * 50}")
            for i, (mae, rmse, corr) in enumerate(zip(maes, rmses, corrs)):
                corr_str = f"{corr:.4f}" if not np.isnan(corr) else "nan"
                print(f"{i + 1:<8} {mae:<15.4f} {rmse:<15.4f} {corr_str:<12}")

            # 计算有效的相关系数数量
            valid_corrs = [c for c in corrs if not np.isnan(c)]
            n_valid_corrs = len(valid_corrs)

            print(f"\n{'=' * 70}")
            print(f"Final Average Results:")
            print(f"{'=' * 70}")
            print(f"  MAE:  {np.nanmean(maes):.4f} ± {np.nanstd(maes):.4f} {self.task_info['unit']}")
            print(f"  RMSE: {np.nanmean(rmses):.4f} ± {np.nanstd(rmses):.4f} {self.task_info['unit']}")

            if n_valid_corrs > 0:
                print(
                    f"  Corr: {np.nanmean(corrs):.4f} ± {np.nanstd(corrs):.4f} (valid in {n_valid_corrs}/{self.n_folds} folds)")
            else:
                print(f"  Corr: nan (no valid correlation in any fold)")

        print(f"{'=' * 70}\n")

        # 保存最终结果到文件
        results_path = os.path.join(
            self.output_dir,
            f"ppgbp_{self.task_name}_kfold_results.txt"
        )
        with open(results_path, 'w') as f:
            f.write(f"{self.n_folds}-Fold Cross-Validation Results\n")
            f.write(f"Task: {self.task_info['full_name']}\n")
            f.write(f"=" * 70 + "\n\n")

            if self.is_classification:
                f.write(f"Accuracy: {np.nanmean(accs):.4f} ± {np.nanstd(accs):.4f}\n")
                f.write(f"F1 Score: {np.nanmean(f1s):.4f} ± {np.nanstd(f1s):.4f}\n")
                f.write(f"AUROC:    {np.nanmean(aurocs):.4f} ± {np.nanstd(aurocs):.4f}\n")
            else:
                f.write(f"MAE:  {np.nanmean(maes):.4f} ± {np.nanstd(maes):.4f} {self.task_info['unit']}\n")
                f.write(f"RMSE: {np.nanmean(rmses):.4f} ± {np.nanstd(rmses):.4f} {self.task_info['unit']}\n")
                if n_valid_corrs > 0:
                    f.write(
                        f"Corr: {np.nanmean(corrs):.4f} ± {np.nanstd(corrs):.4f} (valid in {n_valid_corrs}/{self.n_folds} folds)\n")
                else:
                    f.write(f"Corr: nan (no valid correlation in any fold)\n")

        print(f"✓ Results saved to: {results_path}\n")