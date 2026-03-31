# --------------------------------------------------------
# Codebook Training Engine
# Modified to support proper phase reconstruction visualization
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import io
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from einops import rearrange

import codebook.utils as utils
from codebook.utils import NativeScalerWithGradNormCount as NativeScaler


def plot_spectrum_comparison(original_amp, recon_amp, original_phase=None, recon_phase=None, sample_idx=0,
                             save_path=None):
    """
    Plot spectrum comparison.
    If phase data is provided, plot 2x2 grid:
        - Top row: Amplitude comparison (overlay + error)
        - Bottom row: Phase comparison (overlay + error)
    Otherwise, plot 1x2 grid (amplitude only).
    """

    # Determine if we need to plot phase
    has_phase = (original_phase is not None) and (recon_phase is not None)

    if has_phase:
        # 2x2 layout:
        # Row 1: Amplitude (comparison + error)
        # Row 2: Phase (comparison + error)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        orig_a = original_amp[sample_idx, 0, :].detach().cpu().numpy()
        rec_a = recon_amp[sample_idx, 0, :].detach().cpu().numpy()
        orig_p = original_phase[sample_idx, 0, :].detach().cpu().numpy()
        rec_p = recon_phase[sample_idx, 0, :].detach().cpu().numpy()

        # ============ Row 1: Amplitude ============
        # Left: Original vs Reconstructed overlay
        axs[0, 0].plot(orig_a, label='Original', alpha=0.7, color='blue', linewidth=2)
        axs[0, 0].plot(rec_a, label='Reconstructed', alpha=0.7, linestyle='--', color='orange', linewidth=2)
        axs[0, 0].set_title('Amplitude Spectrum Comparison (0-25 Hz)', fontsize=12, fontweight='bold')
        axs[0, 0].set_xlabel('Frequency (Hz)')
        axs[0, 0].set_ylabel('Normalized Amplitude')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # Right: Amplitude difference
        amp_diff = orig_a - rec_a
        axs[0, 1].plot(amp_diff, alpha=0.7, color='red', linewidth=2)
        axs[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axs[0, 1].set_title('Amplitude Reconstruction Error', fontsize=12, fontweight='bold')
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Amplitude Difference')
        axs[0, 1].grid(True, alpha=0.3)

        # ============ Row 2: Phase ============
        # Left: Original vs Reconstructed overlay
        axs[1, 0].plot(orig_p, label='Original', alpha=0.7, color='green', linewidth=2)
        axs[1, 0].plot(rec_p, label='Reconstructed', alpha=0.7, linestyle='--', color='purple', linewidth=2)
        axs[1, 0].set_title('Phase Spectrum Comparison (0-25 Hz)', fontsize=12, fontweight='bold')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Phase (normalized)')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # Right: Phase difference
        phase_diff = orig_p - rec_p
        axs[1, 1].plot(phase_diff, alpha=0.7, color='red', linewidth=2)
        axs[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axs[1, 1].set_title('Phase Reconstruction Error', fontsize=12, fontweight='bold')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Phase Difference')
        axs[1, 1].grid(True, alpha=0.3)

    else:
        # 1x2 layout: amplitude comparison and difference
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        orig_a = original_amp[sample_idx, 0, :].detach().cpu().numpy()
        rec_a = recon_amp[sample_idx, 0, :].detach().cpu().numpy()

        # Left: Original vs Reconstructed overlay
        axs[0].plot(orig_a, label='Original', alpha=0.7, color='blue', linewidth=2)
        axs[0].plot(rec_a, label='Reconstructed', alpha=0.7, linestyle='--', color='orange', linewidth=2)
        axs[0].set_title('Amplitude Spectrum Comparison (0-25 Hz)', fontsize=12, fontweight='bold')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Normalized Amplitude')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # Right: Difference
        diff = orig_a - rec_a
        axs[1].plot(diff, alpha=0.7, color='red', linewidth=2)
        axs[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axs[1].set_title('Amplitude Reconstruction Error', fontsize=12, fontweight='bold')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude Difference')
        axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        try:
            plt.savefig(save_path, dpi=150)
            print(f"Saved visualization image to: {save_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    image = torch.from_numpy(image).permute(2, 0, 1)[:3, :, :]
    plt.close(fig)
    return image


def train_one_epoch(model: torch.nn.Module,
                    data_loader_list: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    log_writer=None,
                    lr_scheduler=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    ch_names_list=None,
                    args=None
                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = [0, 1]
        for step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
            else:
                images = batch_data

            images = images.float().to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, log_loss, _ = model(images, input_chans=input_chans)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= 1
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=False,
                                    update_grad=(step + 1) % 1 == 0)
            if (step + 1) % 1 == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            new_log_loss = {k.split('/')[-1]: v for k, v in log_loss.items() if k not in ['total_loss']}
            metric_logger.update(**new_log_loss)

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
                log_writer.update(train_loss=loss_value, head="loss", step=it)
                log_writer.update(**new_log_loss, head="loss", step=it)
                log_writer.update(lr=max_lr, head="opt", step=it)
                log_writer.update(min_lr=min_lr, head="opt", step=it)
                log_writer.update(weight_decay=weight_decay_value, head="opt", step=it)
                log_writer.update(grad_norm=grad_norm, head="opt", step=it)

        start_steps += len(data_loader)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if utils.is_main_process():
        try:
            quantizer = utils.get_model(model).quantize
            embedding = quantizer.embedding.weight
            cluster_size = quantizer.cluster_size

            zero_cnt = (cluster_size == 0).sum().item()

            usage_sum = cluster_size.sum() + 1e-6
            probs = cluster_size / usage_sum
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            perplexity = torch.exp(entropy).item()

            max_cluster = quantizer.n_e
            norm_perplexity = perplexity / max_cluster

            print(f"Train Codebook Stats: Unused={zero_cnt}, Perplexity={perplexity:.2f}, Norm_Perp={norm_perplexity:.4f}")

            if log_writer is not None:
                log_writer.update(train_unused_code=zero_cnt, head="codebook", step=epoch)
                log_writer.update(train_perplexity=perplexity, head="codebook", step=epoch)
                log_writer.update(train_norm_perplexity=norm_perplexity, head="codebook", step=epoch)

        except Exception as e:
            print(f"Warning: Failed to get codebook stats: {e}")

    return train_stats


@torch.no_grad()
def evaluate(data_loader_list, model, device, log_writer=None, epoch=None, ch_names_list=None, args=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'

    model.eval()

    try:
        n_code = utils.get_model(model).quantize.num_tokens
    except:
        n_code = args.codebook_n_emd if args else 4096

    val_codebook_usage = torch.zeros(n_code, device=device)

    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
        except:
            pass

    vis_original_amp = None
    vis_recon_amp = None
    vis_original_phase = None
    vis_recon_phase = None

    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        input_chans = [0, 1]
        for step, batch_data in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if isinstance(batch_data, (list, tuple)):
                images = batch_data[0]
            else:
                images = batch_data

            images = images.float().to(device, non_blocking=True)

            loss, log_loss, embed_ind = model(images, input_chans=input_chans)

            val_codebook_usage += torch.bincount(embed_ind.flatten(), minlength=n_code)

            if step == 0 and utils.is_main_process():
                x = rearrange(images, 'B N (A T) -> B N A T', T=args.patch_size)
                x_fft = torch.fft.fft(x, dim=-1)

                # Ground Truth
                target_amp_full = torch.abs(x_fft)
                target_phase_full = torch.angle(x_fft)

                half_len = x.shape[-1] // 2
                target_amp = target_amp_full[..., :half_len]
                target_phase = target_phase_full[..., :half_len]

                x_norm = utils.get_model(model).std_norm(x) if hasattr(utils.get_model(model), 'std_norm') else x
                quantize, _, _, _ = utils.get_model(model).encode(x_norm, input_chans)

                # Check if phase reconstruction is enabled
                reconstruct_phase = getattr(args, 'reconstruct_phase', False) if args else False

                if reconstruct_phase:
                    # Decode returns both amplitude and phase
                    decode_output = utils.get_model(model).decode(quantize, input_chans)
                    if isinstance(decode_output, tuple) and len(decode_output) == 2:
                        rec_amp_full, rec_phase_full = decode_output
                        rec_amp = rec_amp_full[..., :half_len]
                        rec_phase = rec_phase_full[..., :half_len]
                    else:
                        # Fallback: only amplitude
                        rec_amp_full = decode_output
                        rec_amp = rec_amp_full[..., :half_len]
                        rec_phase = None
                else:
                    # Only amplitude reconstruction
                    rec_amp_full = utils.get_model(model).decode(quantize, input_chans)
                    rec_amp = rec_amp_full[..., :half_len]
                    rec_phase = None

                B, N, A, T = x.shape
                vis_original_amp = target_amp
                vis_recon_amp = rearrange(rec_amp, 'b (n a) c -> b n a c', n=N, a=A)

                if reconstruct_phase and rec_phase is not None:
                    # 标准化相位到[-1, 1]以匹配训练时的处理
                    vis_original_phase = target_phase / torch.pi
                    vis_recon_phase = rearrange(rec_phase, 'b (n a) c -> b n a c', n=N, a=A)

            metric_logger.update(loss=loss.item())
            new_log_loss = {k.split('/')[-1]: v for k, v in log_loss.items() if k not in ['total_loss']}
            metric_logger.update(**new_log_loss)

    # 3. Plot and log visualization
    if log_writer is not None and vis_original_amp is not None:
        # Randomly select a sample index for visualization
        sample_idx = np.random.randint(0, vis_original_amp.shape[0])

        # Build save path
        save_path = None
        if args is not None and args.output_dir:
            import os
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}_spectrum_recon.png")

        # Plot with or without phase
        plot_img = plot_spectrum_comparison(
            vis_original_amp[:, 0, :, :],
            vis_recon_amp[:, 0, :, :],
            original_phase=vis_original_phase[:, 0, :, :] if vis_original_phase is not None else None,
            recon_phase=vis_recon_phase[:, 0, :, :] if vis_recon_phase is not None else None,
            sample_idx=sample_idx,
            save_path=save_path
        )

        log_writer.update_image(reconstruction=plot_img, head="spectrum_recon", step=epoch)
        phase_status = "with phase" if vis_original_phase is not None else "amplitude only"
        print(f"Saved spectrum reconstruction visualization ({phase_status}) for Epoch {epoch}")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # stat the codebook usage information
    utils.all_reduce(val_codebook_usage)

    # 2. 计算 Perplexity (困惑度)
    val_usage_sum = val_codebook_usage.sum() + 1e-6
    probs = val_codebook_usage / val_usage_sum
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy).item()

    zero_cnt = (val_codebook_usage == 0).sum().item()

    print(f"Val Codebook Stats (Manual): Unused={zero_cnt}, Perplexity={perplexity:.2f}")

    # 4. 记录到日志
    test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_stat['unused_code'] = zero_cnt
    test_stat['perplexity'] = perplexity

    if log_writer is not None:
        log_writer.update(val_unused_code=zero_cnt, head="codebook_val")
        log_writer.update(val_perplexity=perplexity, head="codebook_val")

    return test_stat


@torch.no_grad()
def calculate_codebook_usage(data_loader, model, device, log_writer=None, epoch=None, args=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Calculating codebook usage:'

    # switch to evaluation mode
    model.eval()

    codebook_num = args.codebook_n_emd
    codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(device)

    for step, batch_data in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if isinstance(batch_data, (list, tuple)):
            images = batch_data[0]
        else:
            images = batch_data

        images = images.float().to(device, non_blocking=True) / 100

        outputs = utils.get_model(model).get_tokens(images)['token'].view(-1)

        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(utils.get_world_size())]
        torch.distributed.all_gather(outputs_gather_list, outputs)
        all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1)  # [B * N * Ngpu, ]

        codebook_cnt += torch.bincount(all_tokens, minlength=codebook_num)

    # statistic
    zero_cnt = (codebook_cnt == 0).sum()  # 0
    probs = codebook_cnt / (codebook_cnt.sum() + 1e-6)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    perplexity = torch.exp(entropy).item()

    print(f"STAT: {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) never used.")
    print(f"STAT: Perplexity = {perplexity:.2f} / {codebook_num} (Higher is better, closer to uniform)")

    return {
        'unused_code': zero_cnt,
        'perplexity': perplexity
    }