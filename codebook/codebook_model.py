# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from codebook.modeling_finetune import NeuralTransformer
from codebook.norm_ema_quantizer import NormEMAVectorQuantizer


class VQNSP(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=4096,
                 embed_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=256,
                 smooth_l1_loss=False,
                 use_consistency_loss=True,
                 consistency_weight=1.0,
                 reconstruct_phase=False,  # 新增：是否重建相位
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = NeuralTransformer(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.patch_size = encoder_config['patch_size']

        self.token_shape = (62, encoder_config['PPG_size'] // self.patch_size)

        self.decoder_out_dim = decoder_out_dim

        # Consistency loss parameters
        self.use_consistency_loss = use_consistency_loss
        self.consistency_weight = consistency_weight

        # Phase reconstruction flag
        self.reconstruct_phase = reconstruct_phase

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)  # for quantize
        )

        # 振幅重建层
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        # 相位重建层（如果启用）
        if self.reconstruct_phase:
            self.decode_task_layer_angle = nn.Sequential(
                nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
                nn.Tanh(),
                nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
            )

        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        if self.reconstruct_phase:
            self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, input_chans=None):
        if x.dim() == 3:
            x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size)
        batch_size, n, a, t = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w)
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        # Return un-quantized features for consistency loss
        return quantize, embed_ind, loss, to_quantizer_features

    def decode(self, quantize, input_chans=None, **kwargs):
        """
        解码量化后的特征

        Returns:
            如果 reconstruct_phase=True: 返回 (rec_amplitude, rec_phase) 元组
            如果 reconstruct_phase=False: 返回 rec_amplitude
        """
        # Reshape from (B, C, H, W) to (B, C, H*W) for decoder
        B, C, H, W = quantize.shape
        quantize = rearrange(quantize, 'b c h w -> b c (h w)')

        decoder_features = self.decoder(quantize, input_chans, return_patch_tokens=True)

        # 重建振幅
        rec_amplitude = self.decode_task_layer(decoder_features)

        if self.reconstruct_phase:
            # 重建相位
            rec_phase = self.decode_task_layer_angle(decoder_features)
            return rec_amplitude, rec_phase
        else:
            return rec_amplitude

    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        if x.dim() == 3:
            x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size)
        return self.encode(x, input_chans)[1].view(x.shape[0], -1)

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def augment(self, x):
        """
        Light augmentation for PPG signals:
        1. Random Scaling: simulate signal strength variations
        2. Gaussian Noise: simulate sensor noise
        """
        B, N, A, T = x.shape
        device = x.device

        # 1. Random Scaling (0.9 ~ 1.1)
        scale = torch.rand(B, 1, 1, 1, device=device) * 0.2 + 0.9
        x_aug = x * scale

        # 2. Gaussian Noise (std=0.05)
        noise = torch.randn_like(x_aug) * 0.05
        x_aug = x_aug + noise

        return x_aug

    def forward(self, x, input_chans=None, **kwargs):

        x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size)
        x_fft = torch.fft.fft(x, dim=-1)

        # 计算振幅和相位
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)

        if self.reconstruct_phase:
            angle = torch.angle(x_fft)
            # 标准化相位到[-1, 1]范围
            angle = angle / torch.pi

        # 1. Original path
        quantize, embed_ind, emb_loss, z_e = self.encode(x, input_chans)

        # 解码
        decode_output = self.decode(quantize, input_chans)

        if self.reconstruct_phase:
            xrec_amplitude, xrec_angle = decode_output
            # 计算振幅重建损失
            rec_loss = self.calculate_rec_loss(xrec_amplitude, amplitude)
            # 计算相位重建损失
            rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)
        else:
            xrec_amplitude = decode_output
            rec_loss = self.calculate_rec_loss(xrec_amplitude, amplitude)
            rec_angle_loss = torch.tensor(0.0, device=x.device)

        # 2. Consistency loss path
        con_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.use_consistency_loss:
            # Generate augmented view
            x_aug = self.augment(x)
            # Encode augmented view
            _, _, _, z_e_aug = self.encode(x_aug, input_chans)

            # Consistency loss (MSE between un-quantized features)
            con_loss = F.mse_loss(z_e.detach(), z_e_aug) * self.consistency_weight

        # 总损失
        loss = emb_loss + rec_loss + rec_angle_loss + con_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        if self.reconstruct_phase:
            log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()
        if self.use_consistency_loss:
            log[f'{split}/con_loss'] = con_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log, embed_ind


def get_model_default_params():
    return dict(PPG_size=12000, patch_size=50, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True,
                use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


@register_model
def vqnsp_encoder_base_decoder_3x250x12(pretrained=False, pretrained_weight=None, as_tokenzer=False,
                                        PPG_size=12000,
                                        n_code=4096,
                                        code_dim=64,
                                        patch_size=100,
                                        reconstruct_phase=False,  # 新增参数
                                        **kwargs):
    encoder_config = get_model_default_params()
    decoder_config = get_model_default_params()

    # Encoder settings
    encoder_config['PPG_size'] = PPG_size
    encoder_config['patch_size'] = patch_size
    encoder_config['num_classes'] = 0
    encoder_config['embed_dim'] = 200
    encoder_config['depth'] = 12
    encoder_config['in_chans'] = 1
    encoder_config['out_chans'] = 8
    encoder_config['use_temporal_conv'] = True

    # Decoder settings
    num_patches = PPG_size // patch_size
    decoder_config['PPG_size'] = num_patches
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['embed_dim'] = 200
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_config['use_abs_pos_emb'] = False
    decoder_config['use_mean_pooling'] = False
    decoder_config['use_temporal_conv'] = False

    decoder_out_dim = patch_size

    # Pass the correctly configured dicts to the VQNSP constructor
    model = VQNSP(encoder_config, decoder_config, n_code, code_dim,
                  decoder_out_dim=decoder_out_dim,
                  reconstruct_phase=reconstruct_phase,  # 传递相位重建参数
                  **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu', weights_only=False)

        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model


if __name__ == '__main__':
    pass