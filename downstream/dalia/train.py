import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
import sys

# Import LabRAM components
try:
    from codebook import modeling_finetune
except ImportError:
    print("Warning: codebook.modeling_finetune not found")
    modeling_finetune = None

try:
    from codebook.optim_factory import create_optimizer
except ImportError:
    print("Warning: codebook.optim_factory not found, will use standard optimizer")
    create_optimizer = None

try:
    from codebook.utils import NativeScalerWithGradNormCount, cosine_scheduler
except ImportError:
    print("Warning: codebook.utils not found")
    NativeScalerWithGradNormCount = None
    cosine_scheduler = None

# Import model selection (if available)
try:
    from downstream.model_select import select_model
except ImportError:
    print("Warning: downstream.model_select not found, will use basic model loading")
    select_model = None

# Adapt for newer PyTorch AMP API
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from downstream.dalia.tools import load_dalia_loso, load_dalia_all_data, DaliaActivityDataset, calculate_metrics

torch.set_float32_matmul_precision('high')


class LabRAMAdapter(nn.Module):
    """
    Adapter to handle different input formats for LabRAM models
    Supports both patch-based (Transformer) and non-patch models (CNN)
    """

    def __init__(self, model, target_len=400, patch_size=50, use_patches=True):
        super(LabRAMAdapter, self).__init__()
        self.model = model
        self.target_len = target_len
        self.patch_size = patch_size
        self.use_patches = use_patches

        print(f"LabRAMAdapter: use_patches={use_patches}, target_len={target_len}, patch_size={patch_size}")

    def forward(self, x):
        # x shape: [Batch, Channel, Length]
        # Expected: [B, C, L] where C=1 (PPG only) or C=4 (PPG+ACC)

        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B, 1, L]
        elif x.ndim == 3 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)  # [B, 1, L]
        elif x.ndim == 4:
            print(f"Warning: Input is already 4D: {x.shape}. Flattening...")
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W)

        # Ensure correct length
        current_len = x.shape[-1]
        if current_len != self.target_len:
            print(f"Warning: Input length mismatch! Expected {self.target_len}, got {current_len}. Fixing...")
            if current_len < self.target_len:
                x = F.pad(x, (0, self.target_len - current_len))
            else:
                x = x[..., :self.target_len]

        # Reshape for patch-based models (Transformer)
        if self.use_patches:
            # [B, C, L] -> [B, C, N, T]
            # 400 / 50 = 8 patches
            x = rearrange(x, 'b c (n t) -> b c n t', t=self.patch_size)

            try:
                return self.model(x, input_chans=[0])
            except TypeError:
                return self.model(x)
        else:
            # For Conv1D models, keep 3D input [B, C, L]
            return self.model(x)


class DaliaTrainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cuda')
        self.output_dir = args.model_save_path
        self.data_dir = os.path.join(args.seg_save_path, "dalia_activity_processed")

        os.makedirs(self.output_dir, exist_ok=True)

        # Task parameters - PPG only
        self.IN_CHANS = 1
        self.NUM_CLASSES = 9  # Activities 0-8

        # Signal parameters (8s at 50Hz)
        self.target_len = 400
        self.patch_size = 40 # 50  # 400 / 50 = 8 patches

        # Training parameters
        self.pretrained = getattr(args, 'pretrained', True)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)

        # AMP settings
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler('cuda')

        self.amp_dtype = torch.bfloat16 if (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ) else torch.float16

        print(f"\n{'=' * 60}")
        print(f"DALIA Activity Classification Trainer")
        print(f"{'=' * 60}")
        print(f"  Task: 9-class Activity Recognition")
        print(f"  Backbone: {args.backbone}")
        print(f"  Input Channels: {self.IN_CHANS} (PPG only)")
        print(f"  Sampling Rate: 50Hz")
        print(f"  Window Size: {self.target_len} samples (8s)")
        print(f"  Patch Size: {self.patch_size}")
        print(f"  Num Patches: {self.target_len // self.patch_size}")
        print(f"  Device: {self.device}")
        print(f"  AMP dtype: {self.amp_dtype}")
        print(f"{'=' * 60}\n")

    def set_seed(self, seed=42):
        """Set random seed for reproducibility"""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_model(self):
        num_classes = self.NUM_CLASSES

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

    def train_one_epoch(self, model, train_loader, optimizer, criterion,
                        lr_schedule_values, epoch, num_steps_per_epoch, return_probs=False):
        """Train for one epoch

        Args:
            return_probs: if True, also return probability predictions for AUROC
        """
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = [] if return_probs else None

        for b_idx, (signal, label) in enumerate(train_loader):
            # Update learning rate
            global_step = b_idx + (epoch - 1) * num_steps_per_epoch
            if lr_schedule_values is not None and global_step < len(lr_schedule_values):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_schedule_values[global_step]

            signal = signal.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            # 🔧 修复：手动reshape for patch-based models (like BIDMC)
            if self.use_patches and signal.shape[-1] == self.target_len:
                signal = rearrange(signal, 'b c (n t) -> b c n t', t=self.patch_size)

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

            # Collect predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(label.cpu())

            # Get probabilities if requested (for AUROC calculation)
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.detach().cpu())  # 🔧 添加detach()

        avg_loss = total_loss / len(train_loader)
        # Convert to appropriate dtype before numpy conversion (BFloat16 not supported by numpy)
        all_preds = torch.cat(all_preds).long().numpy()
        all_targets = torch.cat(all_targets).long().numpy()

        if return_probs:
            all_probs = torch.cat(all_probs).float().numpy()
            return avg_loss, all_preds, all_targets, all_probs
        else:
            return avg_loss, all_preds, all_targets

    @torch.no_grad()
    def evaluate(self, model, data_loader, criterion, return_probs=False):
        """
        Evaluate model

        Args:
            model: the model to evaluate
            data_loader: data loader
            criterion: loss function
            return_probs: if True, also return probability predictions for AUROC

        Returns:
            avg_loss, all_preds, all_targets, [all_probs if return_probs=True]
        """
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = [] if return_probs else None

        for signal, label in data_loader:
            signal = signal.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            # 🔧 修复：手动reshape for patch-based models (like BIDMC)
            if self.use_patches and signal.shape[-1] == self.target_len:
                signal = rearrange(signal, 'b c (n t) -> b c n t', t=self.patch_size)

            if self.use_amp:
                with autocast('cuda', dtype=self.amp_dtype):
                    logits = model(signal)
                    loss = criterion(logits, label)
            else:
                logits = model(signal)
                loss = criterion(logits, label)

            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(label.cpu())

            # Get probabilities if requested (for AUROC calculation)
            if return_probs:
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.detach().cpu())  # 添加detach()保持一致性

        avg_loss = total_loss / len(data_loader)
        # Convert to appropriate dtype before numpy conversion (BFloat16 not supported by numpy)
        all_preds = torch.cat(all_preds).long().numpy()
        all_targets = torch.cat(all_targets).long().numpy()

        if return_probs:
            all_probs = torch.cat(all_probs).float().numpy()
            return avg_loss, all_preds, all_targets, all_probs
        else:
            return avg_loss, all_preds, all_targets

    def training(self):
        """
        Main training function using Leave-One-Subject-Out validation
        """
        self.set_seed(42)

        # Get test subject from args
        target_subject = getattr(self.args, 'test_subject', 'S15')

        print(f"\n{'=' * 60}")
        print(f"Starting DALIA Activity Classification Training")
        print(f"Strategy: Leave-One-Subject-Out (LOSO)")
        print(f"Test Subject: {target_subject}")
        print(f"{'=' * 60}\n")

        # Load data
        X_train, y_train, X_test, y_test = load_dalia_loso(
            self.data_dir,
            target_subject
        )

        # Create datasets
        train_ds = DaliaActivityDataset(
            X_train, y_train,
            mode='train'
        )
        test_ds = DaliaActivityDataset(
            X_test, y_test,
            mode='test'
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"\nDataLoader created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

        # Get model
        model = self.get_model()

        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount(y_train, minlength=self.NUM_CLASSES)
        total_samples = len(y_train)

        # Avoid division by zero: add small epsilon for classes with 0 samples
        class_counts = np.maximum(class_counts, 1)  # Replace 0 with 1
        weights = total_samples / (self.NUM_CLASSES * class_counts.astype(np.float32))
        weights = weights / weights.mean()  # Normalize

        # Clip weights to avoid extreme values
        weights = np.clip(weights, 0.1, 10.0)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        print(f"\nClass weights (for imbalanced data):")
        activity_names = ['Transient', 'Sitting', 'Stairs', 'Soccer',
                          'Cycling', 'Driving', 'Lunch', 'Walking', 'Working']
        for i, (name, w, count) in enumerate(zip(activity_names, weights, class_counts)):
            print(f"  {i}: {name:12s} - weight: {w:.3f}, samples: {count}")

        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Optimizer
        if create_optimizer is not None:
            optimizer = create_optimizer(self.args, model)
        else:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.args.lr,
                weight_decay=getattr(self.args, 'weight_decay', 0.05)
            )

        # Learning rate scheduler
        epochs = self.args.epochs
        num_steps_per_epoch = len(train_loader)
        min_lr = getattr(self.args, 'min_lr', 1e-6)
        warmup_epochs = getattr(self.args, 'warmup_epochs', 5)

        if cosine_scheduler is not None:
            lr_schedule_values = cosine_scheduler(
                self.args.lr, min_lr, epochs, num_steps_per_epoch,
                warmup_epochs=warmup_epochs
            )
        else:
            lr_schedule_values = None
            print("Warning: cosine_scheduler not available, using constant learning rate")

        # Training loop
        # 🔧 修改：使用AUROC作为保存模型的标准
        best_auroc = 0.0
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = getattr(self.args, 'patience', 20)

        print(f"\n{'=' * 60}")
        print(f"Starting Training...")
        print(f"{'=' * 60}\n")

        for epoch in range(1, epochs + 1):
            # Train (获取概率用于AUROC计算)
            train_loss, train_preds, train_targets, train_probs = self.train_one_epoch(
                model, train_loader, optimizer, criterion,
                lr_schedule_values, epoch, num_steps_per_epoch, return_probs=True
            )

            # Evaluate (获取概率用于AUROC计算)
            test_loss, test_preds, test_targets, test_probs = self.evaluate(
                model, test_loader, criterion, return_probs=True
            )

            # Calculate metrics (包含AUROC)
            train_metrics = calculate_metrics(train_preds, train_targets, probs=train_probs)
            test_metrics = calculate_metrics(test_preds, test_targets, probs=test_probs)

            # 🔧 修改：根据AUROC判断是否保存模型
            current_auroc = test_metrics.get('auroc_macro', 0.0)
            if np.isnan(current_auroc):
                current_auroc = 0.0

            if current_auroc > best_auroc:
                best_auroc = current_auroc
                best_acc = test_metrics['accuracy']
                best_f1 = test_metrics['f1']
                best_epoch = epoch
                patience_counter = 0

                # Save best model for this subject
                save_dict = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'accuracy': best_acc,
                    'f1': best_f1,
                    'auroc': best_auroc,  # 保存AUROC
                    'test_subject': target_subject
                }
                save_path = os.path.join(
                    self.output_dir,
                    f"dalia_activity_{target_subject}_best.pth"
                )
                torch.save(save_dict, save_path)
                print(f"✅ New best model saved! AUROC: {best_auroc:.4f}")
            else:
                patience_counter += 1

            # Print progress (包含AUROC)
            if epoch % 5 == 0 or epoch == 1:
                train_auroc_str = f"{train_metrics.get('auroc_macro', 0):.4f}" if not np.isnan(
                    train_metrics.get('auroc_macro', 0)) else "N/A"
                test_auroc_str = f"{test_metrics.get('auroc_macro', 0):.4f}" if not np.isnan(
                    test_metrics.get('auroc_macro', 0)) else "N/A"

                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_loss:.4f}")
                print(
                    f"  Train: Acc={train_metrics['accuracy'] * 100:.2f}%, F1={train_metrics['f1']:.4f}, AUROC={train_auroc_str}")
                print(
                    f"  Test:  Acc={test_metrics['accuracy'] * 100:.2f}%, F1={test_metrics['f1']:.4f}, AUROC={test_auroc_str}")
                print(f"  Best:  AUROC={best_auroc:.4f} (Epoch {best_epoch})")

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # Final results
        print(f"\n{'=' * 60}")
        print(f"Training Completed!")
        print(f"{'=' * 60}")
        print(f"Test Subject: {target_subject}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best AUROC: {best_auroc:.4f}")  # 🔧 修改：优先显示AUROC
        print(f"Best Accuracy: {best_acc * 100:.2f}%")
        print(f"Best F1-Score: {best_f1:.4f}")
        print(f"{'=' * 60}\n")

        # Compute confusion matrix for final evaluation
        model.eval()
        with torch.no_grad():
            test_loss, final_preds, final_targets, final_probs = self.evaluate(
                model, test_loader, criterion, return_probs=True
            )

        # Calculate final metrics with AUROC
        final_metrics = calculate_metrics(final_preds, final_targets, probs=final_probs)

        # Generate confusion matrix with all possible labels (0-8 for DALIA)
        cm = confusion_matrix(final_targets, final_preds, labels=list(range(self.NUM_CLASSES)))
        print("Confusion Matrix:")
        print(cm)
        print("\nPer-class metrics:")
        for i, name in enumerate(activity_names):
            if i < len(cm) and cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum()
                class_auroc = final_metrics.get('auroc_per_class', [float('nan')] * self.NUM_CLASSES)[i]
                auroc_str = f"{class_auroc:.4f}" if not np.isnan(class_auroc) else "N/A"
                print(f"  {i}: {name:12s} - Acc: {class_acc * 100:.2f}% ({cm[i, i]}/{cm[i].sum()}), AUROC: {auroc_str}")
            else:
                print(f"  {i}: {name:12s} - N/A (no samples in test set)")

        return {
            'test_subject': target_subject,
            'best_epoch': best_epoch,
            'best_auroc': best_auroc,  # 🔧 修改：返回AUROC
            'best_accuracy': best_acc,
            'best_f1': best_f1,
            'confusion_matrix': cm
        }