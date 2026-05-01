import torch
import torch.nn as nn
import torch.nn.functional as F
from codebook import modeling_finetune
from model.anyppg import Net1D
from model.papagei import ResNet1DMoE, ResNet1D
from model.papagei_utils import load_model_without_module_prefix
from model.pulse_ppg import PulsePPG_ResNet1D
from functools import partial
from codebook.modeling_finetune import NeuralTransformer
from collections import defaultdict


class GAP1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool(x)
        return x.flatten(1)


class TupleSelector(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            return x[self.index]
        return x


def resize_pos_embed(model, target_patches):
    embed_keys = ['time_embed', 'pos_embed']
    target_found = False

    for key in embed_keys:
        if hasattr(model, key):
            param = getattr(model, key)
            current_patches = param.shape[1]

            if current_patches == target_patches:
                print(f"✅ {key} already correct size: {param.shape}")
                target_found = True
            else:
                print(f"🔧 Resizing {key}: {param.shape} -> ", end="")
                dim = param.shape[-1]

                v = param.permute(0, 2, 1)
                v_new = F.interpolate(v, size=target_patches, mode='linear', align_corners=False)
                v_new = v_new.permute(0, 2, 1)

                with torch.no_grad():
                    delattr(model, key)
                    new_param = nn.Parameter(v_new)
                    setattr(model, key, new_param)

                print(f"{new_param.shape}")
                target_found = True

    if not target_found:
        print("⚠️ No position/time embedding found in model")

    return model


def load_pretrained_weights(model, checkpoint_path, target_patches=None, verbose=True,
                          strict_shape_match=False):
    print(f"\n{'=' * 80}")
    print(f"📦 Loading Pretrained Weights")
    print(f"   Path: {checkpoint_path}")
    print(f"   Strict shape match: {strict_shape_match}")
    print(f"{'=' * 80}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                ckpt_state = checkpoint['model']
                print("✅ Found checkpoint['model']")
            elif 'state_dict' in checkpoint:
                ckpt_state = checkpoint['state_dict']
                print("✅ Found checkpoint['state_dict']")
            else:
                ckpt_state = checkpoint
                print("✅ Using checkpoint as state_dict")
        else:
            ckpt_state = checkpoint
            print("✅ Checkpoint is state_dict")

        ckpt_lookup = {}
        for k, v in ckpt_state.items():
            clean_key = k.replace('module.', '').replace('student.', '')
            ckpt_lookup[clean_key] = v

        print(f"\n📊 Checkpoint: {len(ckpt_lookup)} parameters")

        model_state_dict = model.state_dict()
        print(f"📊 Model: {len(model_state_dict)} parameters")

        pretrained_dict = {}
        mapping_stats = defaultdict(int)

        for model_key in model_state_dict.keys():
            source_key = None
            match_type = "Direct"

            if model_key in ckpt_lookup:
                source_key = model_key
                match_type = "Direct"

            elif model_key.startswith("patch_embed."):
                suffix = model_key[len("patch_embed."):]
                candidates = [
                    f"patch_embed_conv.{suffix}",
                    f"patch_embed_seq.0.{suffix}",
                ]
                for cand in candidates:
                    if cand in ckpt_lookup:
                        source_key = cand
                        match_type = "Alias (patch_embed)"
                        break

            elif model_key.startswith("patch_proj."):
                suffix = model_key[len("patch_proj."):]
                candidates = [
                    f"patch_embed_projection.{suffix}",
                    f"patch_proj.{suffix}",
                ]
                for cand in candidates:
                    if cand in ckpt_lookup:
                        source_key = cand
                        match_type = "Alias (patch_proj)"
                        break

            elif model_key.startswith("fc_norm."):
                suffix = model_key[len("fc_norm."):]
                if f"norm.{suffix}" in ckpt_lookup:
                    source_key = f"norm.{suffix}"
                    match_type = "Alias (fc_norm)"

            if source_key:
                target_shape = model_state_dict[model_key].shape
                source_tensor = ckpt_lookup[source_key]

                if source_tensor.shape == target_shape:
                    pretrained_dict[model_key] = source_tensor
                    mapping_stats[match_type] += 1

                elif ('pos_embed' in model_key or 'time_embed' in model_key) and len(source_tensor.shape) == 3:
                    if source_tensor.shape[2] == target_shape[2]:
                        if verbose:
                            print(f"  🔧 Interpolating {model_key}: {source_tensor.shape} -> {target_shape}")

                        v_perm = source_tensor.permute(0, 2, 1)
                        v_interp = F.interpolate(
                            v_perm, size=target_shape[1], mode='linear', align_corners=False
                        )
                        v_final = v_interp.permute(0, 2, 1)
                        pretrained_dict[model_key] = v_final
                        mapping_stats["Interpolated"] += 1
                    else:
                        if verbose:
                            print(f"  ⚠️ Dimension mismatch for {model_key}")

                else:
                    if verbose and not strict_shape_match:
                        if any(key_part in model_key for key_part in
                               ["patch_embed", "blocks.0", "blocks.1", "norm", "head"]):
                            print(f"  ⚠️ Shape mismatch: {model_key} {target_shape} != {source_tensor.shape}")

        msg = model.load_state_dict(pretrained_dict, strict=False)

        print(f"\n{'=' * 80}")
        print(f"📈 Loading Results")
        print(f"{'=' * 80}")

        loading_rate = 100 * len(pretrained_dict) / len(model_state_dict) if len(model_state_dict) > 0 else 0
        print(f"\n✅ Successfully loaded: {len(pretrained_dict)}/{len(model_state_dict)} parameters")
        print(f"   Loading rate: {loading_rate:.1f}%")

        if mapping_stats:
            print(f"\n📊 Loading breakdown:")
            for match_type, count in sorted(mapping_stats.items()):
                print(f"   {match_type}: {count} params")

        if msg.missing_keys:
            print(f"\n⚠️ Missing keys: {len(msg.missing_keys)}")

            missing_by_prefix = defaultdict(list)
            for key in msg.missing_keys:
                prefix = key.split('.')[0]
                missing_by_prefix[prefix].append(key)

            for prefix, keys in sorted(missing_by_prefix.items()):
                print(f"\n  📌 {prefix}: {len(keys)} params")

                if prefix in ['head', 'fc', 'classifier']:
                    print(f"     ✅ Expected - task-specific head (random init)")
                elif prefix == 'fc_norm':
                    print(f"     ✅ Expected - final norm layer (random init)")
                elif prefix == 'blocks':
                    sample = keys[:3]
                    if any('gamma' in k for k in sample):
                        print(f"     ⚠️ LayerScale params (gamma_1/2) missing")
                    else:
                        print(f"     🔴 CRITICAL - Backbone blocks missing!")
                    for k in sample:
                        print(f"     • {k}")
                    if len(keys) > 3:
                        print(f"     ... and {len(keys) - 3} more")
                elif prefix in ['patch_embed', 'patch_proj']:
                    print(f"     🔴 CRITICAL - Input embedding missing!")
                    for k in keys[:3]:
                        print(f"     • {k}")
                elif prefix in ['pos_embed', 'time_embed']:
                    print(f"     • {keys[0]}")
                elif prefix == 'cls_token':
                    print(f"     • {keys[0]}")
                elif prefix == 'norm':
                    for k in keys:
                        print(f"     • {k}")
                else:
                    for k in keys[:3]:
                        print(f"     • {k}")
                    if len(keys) > 3:
                        print(f"     ... and {len(keys) - 3} more")

        print(f"\n{'=' * 80}")
        print(f"🎯 Critical Check - Backbone Loading")
        print(f"{'=' * 80}")

        backbone_keys = [k for k in model_state_dict.keys()
                        if not any(x in k for x in ['head', 'fc', 'classifier'])]
        backbone_loaded = sum(1 for k in backbone_keys if k in pretrained_dict)
        backbone_rate = 100 * backbone_loaded / len(backbone_keys) if len(backbone_keys) > 0 else 0

        print(f"\n📊 Backbone: {backbone_loaded}/{len(backbone_keys)} ({backbone_rate:.1f}%)")

        if backbone_rate < 50:
            print("🔴 CRITICAL - Most backbone missing!")
            print("💡 Possible reasons:")
            print("   1. Model architecture mismatch (e.g., base vs pro vs large)")
            print("   2. Different embed_dim or out_chans")
            print("   3. Checkpoint from different model version")
            print("\n💡 Suggestion:")
            print("   - Check if checkpoint matches your model architecture")
            print("   - Consider training from scratch if architectures differ significantly")
        elif backbone_rate < 90:
            print("⚠️ WARNING - Some backbone params missing")
        else:
            print("✅ Excellent - Backbone well loaded")

        print(f"\n📋 Key components check:")
        component_groups = {
            'patch_embed': [k for k in backbone_keys if k.startswith('patch_embed')],
            'blocks': [k for k in backbone_keys if k.startswith('blocks')],
            'pos_embed': [k for k in backbone_keys if 'pos_embed' in k],
            'time_embed': [k for k in backbone_keys if 'time_embed' in k],
        }

        for comp_name, comp_keys in component_groups.items():
            if comp_keys:
                loaded = sum(1 for k in comp_keys if k in pretrained_dict)
                rate = 100 * loaded / len(comp_keys) if len(comp_keys) > 0 else 0
                status = "✅" if rate > 90 else "⚠️" if rate > 50 else "🔴"
                print(f"{status} {comp_name:<15} {loaded}/{len(comp_keys):>3} ({rate:>5.1f}%)")

        print(f"\n{'=' * 80}")
        print(f"✅ Weight loading completed!")
        print(f"{'=' * 80}")

        return model, loading_rate

    except Exception as e:
        print(f"❌ Error loading pretrained weights: {e}")
        import traceback
        traceback.print_exc()
        return model, 0.0


def freeze_backbone(model, freeze_flag):
    if not freeze_flag:
        return model

    print("🔒 Freezing backbone...")
    frozen_count = 0
    trainable_count = 0
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        is_task_head = (
            name.startswith('head.') or          # head.weight, head.bias
            name.startswith('fc.') or            # fc.weight, fc.bias (如果head叫fc)
            name.startswith('classifier.') or    # classifier.*
            name == 'head' or                    # 单参数的head (极少见)
            name == 'fc' or
            name == 'classifier'
        )

        if is_task_head:
            param.requires_grad = True
            trainable_count += 1
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_count += 1
            frozen_params += param.numel()

    print(f"  ✓ Frozen {frozen_count} parameter tensors ({frozen_params:,} elements)")
    print(f"  ✓ Trainable {trainable_count} parameter tensors ({trainable_params:,} elements)")

    total_params = frozen_params + trainable_params
    if total_params > 0:
        trainable_pct = 100 * trainable_params / total_params
        print(f"  ✓ Trainable ratio: {trainable_pct:.2f}%")

        if trainable_pct > 5.0:
            print(f"  ⚠️  Warning: {trainable_pct:.1f}% trainable - this seems high for linear probing!")
            print(f"     Expected: < 1% for typical classification heads")
            print(f"     Please check if head layer is correctly identified")

    return model


SIGMA_CONFIGS = {
    'sigma_ppg_pro': {
        'out_chans': 12,
        'embed_dim': 360,
        'depth': 18,
        'num_heads': 12,
        'drop_rate': 0.08,
        'attn_drop_rate': 0.08,
        'drop_path_rate': 0.15,
    },
}


def create_sigma_model(model_size, num_classes, input_size, patch_size, in_chans=1, init_values=0.1):
    if model_size not in SIGMA_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(SIGMA_CONFIGS.keys())}")

    config = SIGMA_CONFIGS[model_size]

    print(f"\n{'=' * 80}")
    print(f"🏗️  Creating Model - Dynamic Configuration")
    print(f"{'=' * 80}")
    print(f"Model size: {model_size}")
    print(f"Input size: {input_size}")
    print(f"Patch size: {patch_size}")
    print(f"Number of patches: {input_size // patch_size}")
    print(f"Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 80}\n")

    model = NeuralTransformer(
        patch_size=patch_size,
        PPG_size=input_size,
        in_chans=in_chans,
        num_classes=num_classes,
        out_chans=config['out_chans'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4.0,
        drop_rate=config['drop_rate'],
        attn_drop_rate=config['attn_drop_rate'],
        drop_path_rate=config['drop_path_rate'],
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=init_values  # 启用LayerScale
    )

    return model


def select_model(
    backbone='sigma_ppg_pro',
    num_classes=2,
    device='cuda',
    pretrained=False,
    checkpoint_path=None,
    freeze_backbone_flag=False,
    in_chans=1,
    input_size=12000,
    patch_size=100,
):

    print(f"\n{'=' * 80}")
    print(f"🎯 Model Selection - Dynamic Configuration")
    print(f"{'=' * 80}")
    print(f"Backbone: {backbone}")
    print(f"Input size: {input_size}")
    print(f"Patch size: {patch_size}")
    print(f"Num classes: {num_classes}")
    print(f"In channels: {in_chans}")
    print(f"Pretrained: {pretrained}")
    print(f"{'=' * 80}\n")

    target_patches = input_size // patch_size
    use_patches = False

    if backbone in SIGMA_CONFIGS:
        print(f"📦 Creating {backbone} with dynamic parameters...")

        model = create_sigma_model(
            model_size=backbone,
            num_classes=num_classes,
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            init_values=0.1
        )

        print(f"✅ Model created with LayerScale (init_values=0.1)")

        model = resize_pos_embed(model, target_patches)

        if pretrained and checkpoint_path:
            model, loading_rate = load_pretrained_weights(
                model,
                checkpoint_path,
                target_patches,
                verbose=True,
                strict_shape_match=False
            )

            if loading_rate < 50:
                print("\n⚠️ WARNING: Low loading rate detected!")
                print("💡 This might indicate architecture mismatch between checkpoint and current model.")
                print("💡 Consider:")
                print("   1. Using matching model size (base/pro/large/huge)")
                print("   2. Training from scratch for this architecture")
                print("   3. Fine-tuning with a smaller learning rate")

        elif not pretrained:
            print("🚫 Pretrained disabled. Training from scratch.")
        elif not checkpoint_path:
            print("⚠️ No checkpoint path provided. Training from scratch.")

        model = freeze_backbone(model, freeze_backbone_flag)

        use_patches = True

    # ========================
    # AnyPPG
    # ========================
    elif backbone == 'anyppg':
        print("Loading AnyPPG model...")

        anyppg_cfg = {
            "in_channels": in_chans,
            "base_filters": 64,
            "ratio": 1.0,
            "filter_list": [64, 160, 160, 400, 400, 1024],
            "m_blocks_list": [2, 2, 2, 3, 3, 1],
            "kernel_size": 3,
            "stride": 2,
            "groups_width": 16,
            "verbose": False,
        }

        encoder = Net1D(**anyppg_cfg)

        if pretrained and checkpoint_path:
            print(f"Loading pretrained weights from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            encoder.load_state_dict(state_dict)
            print("✅ Pretrained weights loaded")

        linear_head = nn.Linear(1024, num_classes)
        model = nn.Sequential(encoder, GAP1d(), linear_head)

        if freeze_backbone_flag:
            print("🔒 Freezing encoder...")
            for param in encoder.parameters():
                param.requires_grad = False
            print("  ✓ Only head trainable")

        use_patches = False

    # ========================
    # PAPAGEI-S (MoE)
    # ========================
    elif backbone == 'papagei_s':
        print("Loading PAPAGEI-S (MoE) model...")

        model_config = {
            'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
            'n_experts': 3
        }

        encoder = ResNet1DMoE(
            in_channels=in_chans,
            base_filters=model_config['base_filters'],
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes'],
            n_experts=model_config['n_experts']
        )

        if pretrained and checkpoint_path:
            print(f"Loading pretrained weights from: {checkpoint_path}")
            encoder = load_model_without_module_prefix(encoder, checkpoint_path)
            print("✅ Pretrained weights loaded")

        linear_head = nn.Linear(512, num_classes)
        model = nn.Sequential(encoder, TupleSelector(3), linear_head)

        if freeze_backbone_flag:
            print("🔒 Freezing encoder...")
            for param in encoder.parameters():
                param.requires_grad = False
            print("  ✓ Only head trainable")

        use_patches = False

    # ========================
    # PAPAGEI-P (standard)
    # ========================
    elif backbone == 'papagei_p':
        print("Loading PAPAGEI-P (standard ResNet) model...")

        model_config = {
            'base_filters': 32,
            'kernel_size': 3,
            'stride': 2,
            'groups': 1,
            'n_block': 18,
            'n_classes': 512,
        }

        encoder = ResNet1D(
            in_channels=in_chans,
            base_filters=model_config['base_filters'],
            kernel_size=model_config['kernel_size'],
            stride=model_config['stride'],
            groups=model_config['groups'],
            n_block=model_config['n_block'],
            n_classes=model_config['n_classes']
        )

        if pretrained and checkpoint_path:
            print(f"Loading pretrained weights from: {checkpoint_path}")
            encoder = load_model_without_module_prefix(encoder, checkpoint_path)
            print("✅ Pretrained weights loaded")

        linear_head = nn.Linear(512, num_classes)
        model = nn.Sequential(encoder, TupleSelector(1), linear_head)

        if freeze_backbone_flag:
            print("🔒 Freezing encoder...")
            for param in encoder.parameters():
                param.requires_grad = False
            print("  ✓ Only head trainable")

        use_patches = False

    # ========================
    # PULSE-PPG
    # ========================
    elif backbone == 'pulse-ppg':
        print("Using Pulse-PPG configuration...")

        encoder = PulsePPG_ResNet1D(
            in_channels=in_chans,
            base_filters=128,
            kernel_size=11,
            stride=2,
            groups=1,
            n_block=12,
            finalpool="max",
            use_bn=True,
            use_do=True
        )

        if pretrained and checkpoint_path:
            print(f"Loading Pulse-PPG pretrained weights from: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                if isinstance(checkpoint, dict) and 'net' in checkpoint:
                    state_dict = checkpoint['net']
                else:
                    state_dict = checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v

                msg = encoder.load_state_dict(new_state_dict, strict=True)
                print(f"✅ Pulse-PPG weights loaded successfully: {msg}")
            except Exception as e:
                print(f"❌ Error loading Pulse-PPG weights: {e}")
                msg = encoder.load_state_dict(new_state_dict, strict=False)
                print(f"⚠️ Loaded with strict=False: {msg}")

        linear_head = nn.Linear(512, num_classes)
        model = nn.Sequential(encoder, linear_head)

        if freeze_backbone_flag:
            print("🔒 Freezing encoder...")
            for param in encoder.parameters():
                param.requires_grad = False
            print("  ✓ Only head trainable")

        use_patches = False

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Use patches (4D input): {use_patches}")
    print(f"{'=' * 80}\n")

    return model, use_patches