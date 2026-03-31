import math
import sys
import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

_orig_makedirs = os.makedirs


def _patched_makedirs(name, mode=0o777, exist_ok=False):
    if "pyhealth" in str(name):
        exist_ok = True
    return _orig_makedirs(name, mode, exist_ok)


os.makedirs = _patched_makedirs
print("[Fix] Patched os.makedirs to prevent pyhealth race condition.")

from codebook.utils import *
from einops import rearrange
from contextlib import nullcontext
from codebook.utils import SmoothedValue

if not hasattr(SmoothedValue, '_patched_safe_methods'):
    def safe_global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


    def safe_max(self):
        if len(self.deque) == 0:
            return 0.0
        return max(self.deque)


    def safe_median(self):
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()


    def safe_avg(self):
        if len(self.deque) == 0:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()


    def safe_value(self):
        if len(self.deque) == 0:
            return 0.0
        return self.deque[-1]


    # Replace original attributes
    SmoothedValue.global_avg = property(safe_global_avg)
    SmoothedValue.max = property(safe_max)
    SmoothedValue.median = property(safe_median)
    SmoothedValue.avg = property(safe_avg)
    SmoothedValue.value = property(safe_value)

    SmoothedValue._patched_safe_methods = True
    print(
        "[Fix] SmoothedValue methods (global_avg, max, median, avg, value) have been patched to avoid errors on empty data.")


def limit_mask_continuous_length_vectorized(mask, max_len=5):
    if mask.dtype != torch.bool:
        mask = mask.bool()

    B, A = mask.shape
    device = mask.device

    padded_mask = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), mask], dim=1)

    indices = torch.arange(A + 1, device=device).unsqueeze(0).expand(B, -1)

    zero_locations = indices * (~padded_mask).long()
    last_zero_idx = zero_locations.cummax(dim=1).values

    consecutive_counts = (indices - last_zero_idx)[:, 1:]

    limit = max_len + 1
    to_unmask = (consecutive_counts % limit == 0) & mask

    new_mask = mask.masked_fill(to_unmask, False)

    return new_mask


def apply_random_punching(mask, punch_ratio):
    if punch_ratio <= 0:
        return mask

    rand_probs = torch.rand_like(mask, dtype=torch.float32)

    to_unmask = mask & (rand_probs < punch_ratio)

    mask = mask.masked_fill(to_unmask, False)

    return mask


def generate_adios_mask_random(samples, ratio, device, punch_ratio=0.0, max_len=5):
    B, N, A, T = samples.shape
    len_mask = int(A * ratio)

    noise = torch.rand(B, A, device=device)

    ids = torch.argsort(noise, dim=1)[:, :len_mask]

    mask = torch.zeros([B, A], device=device, dtype=torch.bool)
    mask.scatter_(1, ids, True)

    if punch_ratio > 0:
        mask = apply_random_punching(mask, punch_ratio)

    mask = limit_mask_continuous_length_vectorized(mask, max_len=max_len)

    return mask


def generate_knowledge_guided_mask(knowledge_scores, ratio, device, punch_ratio=0.0, max_len=5):
    B, A = knowledge_scores.shape
    len_mask = int(A * ratio)

    mean = knowledge_scores.mean(dim=-1, keepdim=True)
    std = knowledge_scores.std(dim=-1, keepdim=True) + 1e-6
    norm_scores = (knowledge_scores - mean) / std

    temperature = 0.8
    logits = norm_scores / temperature

    probs = F.softmax(logits, dim=-1)

    ids_mask = torch.multinomial(probs, num_samples=len_mask, replacement=False)
    mask = torch.zeros([B, A], device=device, dtype=torch.bool)
    mask.scatter_(1, ids_mask, True)

    if punch_ratio > 0:
        mask = apply_random_punching(mask, punch_ratio)

    mask = limit_mask_continuous_length_vectorized(mask, max_len=max_len)

    return mask


def sample_mask_from_teacher(teacher_model, samples, knowledge_scores, current_ratio, device,
                             use_knowledge_bias=True, punch_ratio=0.0, max_len=5):
    B, N, A, T = samples.shape
    len_mask = int(A * current_ratio)

    teacher_logits = teacher_model(samples)

    if use_knowledge_bias:
        prior_bias = (knowledge_scores - knowledge_scores.mean(dim=1, keepdim=True)) / (
                knowledge_scores.std(dim=1, keepdim=True) + 1e-6)
        final_logits = teacher_logits + 2.0 * prior_bias  # L_Final (Eq. 10)
    else:
        final_logits = teacher_logits

    # ── Gumbel-Top-k sampling (Section 2.2.2, Eq. 12-14) ──────────────────
    # Eq. 12: G_i ~ Gumbel(0, 1)  via the inverse-CDF transform:
    #         if U ~ Uniform(0,1), then -log(-log(U)) ~ Gumbel(0,1)
    U = torch.rand_like(final_logits).clamp(min=1e-10, max=1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(U))          # G_i

    # Eq. 13: y_tilde_i = L_Final_i + G_i
    perturbed_logits = final_logits + gumbel_noise

    # Eq. 14: M_i = 1  iff  y_tilde_i  is in top-k({ y_tilde_1,...,y_tilde_N })
    _, ids_mask = torch.topk(perturbed_logits, k=len_mask, dim=1)
    # ───────────────────────────────────────────────────────────────────────

    # Entropy and log-probs are computed from the *unperturbed* distribution,
    # consistent with the Plackett-Luce policy  π_θ(M|X)  used in REINFORCE.
    probs = F.softmax(final_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

    mask = torch.zeros([B, A], device=device, dtype=torch.bool)
    mask.scatter_(1, ids_mask, True)

    selected_probs = torch.gather(probs, 1, ids_mask)
    log_probs = torch.log(selected_probs + 1e-10).sum(dim=1)

    avg_probs = selected_probs.mean()

    if punch_ratio > 0:
        mask = apply_random_punching(mask, punch_ratio)

    mask = limit_mask_continuous_length_vectorized(mask, max_len=max_len)

    return mask, log_probs, entropy, avg_probs


def train_one_epoch(model: torch.nn.Module, vqnsp: torch.nn.Module, teacher_model: torch.nn.Module,
                    data_loader_list: Iterable,
                    optimizer: torch.optim.Optimizer, optimizer_teacher: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    teacher_model.train()

    progress = epoch / args.epochs

    # Use fixed mask ratio
    current_mask_ratio = args.mask_ratio

    if args.use_teacher:
        teacher_active = (progress >= args.teacher_warmup_ratio)
    else:
        teacher_active = False

    use_knowledge = args.use_knowledge_masking
    punch_ratio = args.punch_ratio

    max_len = args.mask_max_len if hasattr(args, 'mask_max_len') else 5

    # Determine if we should apply symmetric masking
    if teacher_active:
        apply_symmetric = False
    else:
        apply_symmetric = args.use_symmetric_masking

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = get_input_chans(ch_names)

        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[start_steps + step + step_loader] * param_group.get(
                            "lr_scale", 1.0)
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[start_steps + step + step_loader]

            # Unpack batch
            samples, feat_stack = batch
            samples = samples.to(device, non_blocking=True)
            feat_stack = feat_stack.to(device, non_blocking=True)

            # ==========================================
            # [CRITICAL FIX] Handle NaNs and Infs on the fly
            # ==========================================
            # 1. Cleaning: Replace NaN/Inf with 0.0 (assuming normalized data mean is roughly 0)
            # This prevents skipping batches and keeps training running even with dirty data.
            if torch.isnan(samples).any() or torch.isinf(samples).any():
                samples = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

            # Clean feature stack as well (used for knowledge masking)
            if torch.isnan(feat_stack).any() or torch.isinf(feat_stack).any():
                feat_stack = torch.nan_to_num(feat_stack, nan=0.0, posinf=0.0, neginf=0.0)

            knowledge_scores = feat_stack[:, 2, :]

            # ==========================================
            # Step 1: Generate Mask
            # ==========================================
            mask_log_probs = None
            t_entropy = None
            t_avg_prob = None

            patch_size = args.patch_size if hasattr(args, 'patch_size') else 100
            B, C, Length = samples.shape
            num_patches = Length // patch_size

            if teacher_active:
                samples_4d = samples.reshape(B, C, num_patches, patch_size)

                mask, mask_log_probs, t_entropy, t_avg_prob = sample_mask_from_teacher(
                    teacher_model, samples_4d, knowledge_scores, current_mask_ratio, device,
                    use_knowledge_bias=args.use_knowledge_bias, punch_ratio=punch_ratio, max_len=max_len
                )

            elif use_knowledge:
                # Knowledge-guided masking
                mask = generate_knowledge_guided_mask(
                    knowledge_scores, current_mask_ratio, device, punch_ratio=punch_ratio, max_len=max_len
                )
            else:
                samples_4d = samples.reshape(B, C, num_patches, patch_size)
                # Random masking
                mask = generate_adios_mask_random(
                    samples_4d, current_mask_ratio, device, punch_ratio=punch_ratio, max_len=max_len
                )

            bool_masked_pos = mask

            # Calculate Teacher monitoring metrics (k_bias)
            with torch.no_grad():
                mask_float = mask.float()
                # Prevent division by zero
                masked_mean_score = (knowledge_scores * mask_float).sum() / (mask_float.sum() + 1e-6)
                global_mean_score = knowledge_scores.mean()
                k_bias = masked_mean_score / (global_mean_score + 1e-6)

            metric_logger.update(k_bias=k_bias.item())
            if t_entropy is not None:
                metric_logger.update(t_ent=t_entropy.item())
                metric_logger.update(t_prob=t_avg_prob.item())

            # ==========================================
            # Step 2: Prepare labels (Codebook Indices)
            # ==========================================
            with torch.no_grad():
                # [CRITICAL FIX] Force FP32 (disable autocast) for Tokenizer Inference
                # This prevents FP16 overflow errors (CUBLAS_STATUS_EXECUTION_FAILED)
                # caused by large values or spikes in the input data.
                with torch.cuda.amp.autocast(enabled=False):
                    input_ids = vqnsp.get_codebook_indices(samples.float(), input_chans)

                labels = input_ids[bool_masked_pos]

                if apply_symmetric:
                    labels_sym = input_ids[~bool_masked_pos]
                else:
                    labels_sym = None

            # ==========================================
            # Step 3: Student learning (Student Forward & Backward)
            # ==========================================
            my_context = model.no_sync if args.distributed and (
                    step + 1) % args.gradient_accumulation_steps != 0 else nullcontext

            with my_context():
                with torch.cuda.amp.autocast():
                    # Student Forward
                    outputs = model(samples, input_chans, bool_masked_pos=bool_masked_pos,
                                    apply_symmetric_masking=apply_symmetric)

                    x_rec, x_rec_sym = outputs

                    raw_loss = loss_fn(x_rec, labels)
                    loss_rec = raw_loss.mean()

                    if apply_symmetric and x_rec_sym is not None:
                        loss_rec_sym = loss_fn(x_rec_sym, labels_sym).mean()
                        loss = loss_rec + loss_rec_sym
                    else:
                        loss = loss_rec
                        loss_rec_sym = torch.tensor(0.0, device=device)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                # Since inputs are now clean, NaN loss means divergence.
                print(f"Loss is {loss_value}, stopping training at rank {get_rank()}", force=True)
                sys.exit(1)

            loss /= args.gradient_accumulation_steps

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            # ==========================================
            # Step 4: Teacher learning (Teacher Update via RL)
            # ==========================================
            if teacher_active and mask_log_probs is not None:
                # Calculate per-sample loss
                num_masked_per_sample = bool_masked_pos.sum(dim=1)  # [B]

                # Build a [B, A] loss map
                loss_map = torch.zeros_like(bool_masked_pos, dtype=torch.float32)
                loss_map[bool_masked_pos] = raw_loss.detach()

                # Calculate mean for each sample (only mask part)
                per_sample_loss = loss_map.sum(dim=1) / (num_masked_per_sample + 1e-6)  # [B]

                # Subtract Baseline (Batch Mean)
                reward = (per_sample_loss - per_sample_loss.mean()).detach()

                # RL Update
                teacher_loss = -(reward * mask_log_probs).mean()
                optimizer_teacher.zero_grad()
                teacher_loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
                optimizer_teacher.step()

                metric_logger.update(t_loss=teacher_loss.item())

            torch.cuda.synchronize()

            # ==========================================
            # Logging
            # ==========================================
            # Calculate MLM Accuracy
            pred_ids = x_rec.max(-1)[1]
            correct = (pred_ids == labels)
            mlm_acc = correct.float().mean().item()

            # Calculate student monitoring metrics
            with torch.no_grad():
                # 1. Calculate Student Entropy
                probs_s = F.softmax(x_rec, dim=-1)
                log_probs_s = torch.log(probs_s + 1e-10)
                s_entropy = -(probs_s * log_probs_s).sum(dim=-1).mean()

                # 2. Calculate Peak/Flat Accuracy
                masked_k_scores = knowledge_scores[bool_masked_pos]

                k_threshold = masked_k_scores.mean()

                peak_mask = masked_k_scores > k_threshold
                flat_mask = masked_k_scores <= k_threshold

                if peak_mask.sum() > 0:
                    acc_peak = correct[peak_mask].float().mean().item()
                else:
                    acc_peak = 0.0

                if flat_mask.sum() > 0:
                    acc_flat = correct[flat_mask].float().mean().item()
                else:
                    acc_flat = 0.0

            metric_logger.update(mlm_acc=mlm_acc)
            metric_logger.update(loss_rec=loss_rec.item())

            # Update new student metrics
            metric_logger.update(s_ent=s_entropy.item())
            metric_logger.update(acc_peak=acc_peak)
            metric_logger.update(acc_flat=acc_flat)

            if log_writer is not None:
                log_writer.update(mlm_acc=mlm_acc, head="loss")
                log_writer.update(loss_rec=loss_rec.item(), head="loss")
                log_writer.update(mask_ratio=current_mask_ratio, head="curriculum")
                log_writer.update(k_bias=k_bias.item(), head="stats")
                if t_entropy is not None:
                    log_writer.update(t_ent=t_entropy.item(), head="stats")
                    log_writer.update(t_prob=t_avg_prob.item(), head="stats")

                # Record student metrics to Tensorboard
                log_writer.update(s_ent=s_entropy.item(), head="student")
                log_writer.update(acc_peak=acc_peak, head="student")
                log_writer.update(acc_flat=acc_flat, head="student")

            if apply_symmetric:
                mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
                metric_logger.update(mlm_acc_sym=mlm_acc_sym)
                metric_logger.update(loss_rec_sym=loss_rec_sym.item())
                if log_writer is not None:
                    log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)

            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)

        step_loader += step

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}