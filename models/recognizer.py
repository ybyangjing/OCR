# from imaplib import Debug
from typing import Dict, Any
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork, DeformConv2d
import matplotlib.pyplot as plt



def create_robust_gn(num_channels, group_size=16, eps=1e-5):
    if not isinstance(num_channels, int) or num_channels <= 0:
        raise ValueError(f"num_channels must be a positive integer, but got {num_channels}")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, but got {group_size}")


    actual_group_size = min(group_size, num_channels)

    if num_channels % actual_group_size == 0:
        num_groups = num_channels // actual_group_size
    else:

        found_divisor = False
        for gs in range(actual_group_size, 0, -1):
            if num_channels % gs == 0:
                num_groups = num_channels // gs
                found_divisor = True
                break
        if not found_divisor:
            num_groups = 1


    return nn.GroupNorm(num_groups, num_channels, eps=eps)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        seq_len_dim = 1 if x.dim() == 3 and x.size(2) == self.pe.size(1) else 0
        seq_len = x.size(seq_len_dim)
        if seq_len > self.pe.size(0):
            raise ValueError(f"Input sequence length {seq_len} exceeds the maximum length of positional encoding {self.pe.size(0)}")

        if x.dim() == 3 and x.size(2) == self.pe.size(1):
            x = x + self.pe[:seq_len, :].unsqueeze(0)
        else:
            x = x + self.pe[:seq_len, :].unsqueeze(1)
        return x


#
class LearnedPositionalEncoding2D(nn.Module):
    def __init__(self, embedding_dim, height=16, width=32):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, embedding_dim, height, width))
        nn.init.trunc_normal_(self.pos_embedding, std=.02)


    def forward(self, x):
        if x.shape[2:] != self.pos_embedding.shape[2:]:

            pos_embedding_resized = F.interpolate(self.pos_embedding, size=x.shape[2:], mode='bicubic',
                                                  align_corners=False)
            return x + pos_embedding_resized,pos_embedding_resized
        return x + self.pos_embedding,self.pos_embedding


class ConditionalPositionalEncoding2D(nn.Module):
    def __init__(self, input_feature_channels, output_embedding_dim, hidden_dim=256, num_conv_layers=2):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim

        layers = []
        current_channels = input_feature_channels
        for _ in range(num_conv_layers - 1):
            layers.append(
                ConvNormAct(current_channels, hidden_dim, kernel_size=3, stride=1, use_gn=True))
            current_channels = hidden_dim


        layers.append(nn.Conv2d(current_channels, output_embedding_dim, kernel_size=1))
        self.cpe_generator = nn.Sequential(*layers)
        print(
            f"Initialize ConditionalPositionalEncoding2D，Input channels: {input_feature_channels}, Output embedding dimension: {output_embedding_dim}")

    def forward(self, x_feat):

        conditional_pe = self.cpe_generator(x_feat)


        if x_feat.shape[1] != conditional_pe.shape[1]:
            raise ValueError(
                f"Number of channels for Conditional Positional Encoding ({conditional_pe.shape[1]}) and input feature channels ({x_feat.shape[1]}) do not match。")

        return x_feat + conditional_pe,conditional_pe



class DeformableConditionalPositionalEncoding2D(nn.Module):
    def __init__(self, input_feature_channels, output_embedding_dim, model_cfg: Dict[str, Any]):
        super().__init__()
        self.output_embedding_dim = output_embedding_dim
        hidden_dim = model_cfg.get('cpe_hidden_dim', 256)
        num_conv_layers = model_cfg.get('cpe_num_conv_layers', 2)
        use_gn_in_cpe = model_cfg.get('cnn_use_gn', True)  #
        group_size_gn_cpe = model_cfg.get('cnn_group_size', 16)
        gn_eps_cpe = model_cfg.get('cnn_gn_eps', 1e-5)

        self.layers = nn.ModuleList()
        self.offset_predictors = nn.ModuleList()

        current_channels = input_feature_channels
        deform_kernel_size = 3

        for i in range(num_conv_layers):
            is_last_layer = (i == num_conv_layers - 1)
            out_c = output_embedding_dim if is_last_layer else hidden_dim
            k_size = 1 if is_last_layer else deform_kernel_size


            if not is_last_layer:

                offset_channels = 1 * 2 * deform_kernel_size * deform_kernel_size
                self.offset_predictors.append(
                    nn.Conv2d(current_channels, offset_channels, kernel_size=3, padding=1)
                )

                nn.init.constant_(self.offset_predictors[-1].weight, 0.)
                nn.init.constant_(self.offset_predictors[-1].bias, 0.)

                self.layers.append(
                    ConvNormAct(current_channels, out_c, kernel_size=deform_kernel_size, stride=1,
                                use_gn=use_gn_in_cpe, group_size_gn=group_size_gn_cpe, gn_eps=gn_eps_cpe,
                                is_deformable=True, offset_groups=1)
                )
            else:
                self.offset_predictors.append(None)
                self.layers.append(
                    nn.Conv2d(current_channels, out_c, kernel_size=1)
                )
            current_channels = out_c

        print(
            f"Initialize DeformableConditionalPositionalEncoding2D，Input channels: {input_feature_channels}, Output embedding dimension: {output_embedding_dim}")

    def forward(self, x_feat):
        conditional_pe_prog = x_feat
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvNormAct) and layer.is_deformable:
                offset_predictor = self.offset_predictors[i]
                if offset_predictor is None:
                    raise ValueError("Offset predictor missing for deformable layer")
                offset = offset_predictor(conditional_pe_prog)
                conditional_pe_prog = layer(conditional_pe_prog, offset)
            else:
                conditional_pe_prog = layer(conditional_pe_prog)

        conditional_pe = conditional_pe_prog
        if x_feat.shape[1] != conditional_pe.shape[1]:
            raise ValueError(f"DCN_CPE Number of channels ({conditional_pe.shape[1]}) Input features ({x_feat.shape[1]}) do not match。")
        return x_feat + conditional_pe, conditional_pe


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)

    attn_mask = torch.zeros(sz, sz, dtype=torch.float32, device=device)
    attn_mask = attn_mask.masked_fill(mask, float('-inf'))
    return attn_mask


class ConvNormAct(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1,
                 use_gn=False, group_size_gn=16, gn_eps=1e-5, act_fn='silu', is_deformable=False, offset_groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.is_deformable = is_deformable
        if self.is_deformable:

            self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                     groups=groups, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)

        if use_gn:
            self.norm = create_robust_gn(out_channels, group_size_gn, gn_eps)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        if act_fn == 'silu':
            self.act = nn.SiLU()
        elif act_fn == 'relu':
            self.act = nn.ReLU()
        elif act_fn is None:
            self.act = nn.Identity()
        else:
            raise NotImplementedError(f"Activation function {act_fn} Unrealized")

    def forward(self, x, offset=None):
        if self.is_deformable:
            if offset is None:
                raise ValueError("Deformable convolution requires an offset input.")
            conv_out = self.conv(x, offset)
        else:


            conv_out = self.conv(x)

        return self.act(self.norm(conv_out))


class YoloMSBottleneck(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, shortcut=True,
                 kernel_sizes=(3, 5), use_gn=False, group_size_gn=16, gn_eps=1e-5):
        super().__init__()
        mid_channels = out_channels // 2
        num_branches = len(kernel_sizes)

        branch_out_channels = mid_channels // num_branches

        actual_total_branch_out = 0

        self.conv1 = ConvNormAct(in_channels, mid_channels, kernel_size=1, use_gn=use_gn, group_size_gn=group_size_gn)

        self.branches = nn.ModuleList()
        for i in range(num_branches):

            current_branch_out = branch_out_channels
            if i == num_branches - 1:
                current_branch_out = mid_channels - (branch_out_channels * (num_branches - 1))

            self.branches.append(
                ConvNormAct(mid_channels, current_branch_out, kernel_size=kernel_sizes[i],
                            use_gn=use_gn, group_size_gn=group_size_gn, gn_eps=gn_eps)
            )
            actual_total_branch_out += current_branch_out


        self.conv2 = ConvNormAct(actual_total_branch_out, out_channels, kernel_size=1, use_gn=use_gn,
                                 group_size_gn=group_size_gn)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        identity = x
        out_conv1 = self.conv1(x)
        branch_outputs = []
        for branch_module in self.branches:
            branch_outputs.append(branch_module(out_conv1))

        out = torch.cat(branch_outputs, dim=1)
        out = self.conv2(out)

        if self.use_shortcut:
            return identity + out
        return out


class YoloMSCSPStage(nn.Module):

    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut_in_bottleneck=True,
                 kernel_sizes_bottleneck=(3, 5), use_gn=False, group_size_gn=16, gn_eps=1e-5,
                 downsample_first=True):
        super().__init__()
        csp_branch_channels = out_channels // 2

        if downsample_first:
            self.downsample_conv = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                               use_gn=use_gn, group_size_gn=group_size_gn)

            csp_input_channels = out_channels
        else:
            self.downsample_conv = nn.Identity()

            if in_channels != out_channels:
                self.projection_conv = ConvNormAct(in_channels, out_channels, kernel_size=1, use_gn=use_gn,
                                                   group_size_gn=group_size_gn)
                csp_input_channels = out_channels
            else:
                self.projection_conv = nn.Identity()
                csp_input_channels = in_channels


        self.conv_main_path = ConvNormAct(csp_input_channels, csp_branch_channels, kernel_size=1, use_gn=use_gn,
                                          group_size_gn=group_size_gn)

        self.conv_short_path = ConvNormAct(csp_input_channels, csp_branch_channels, kernel_size=1, use_gn=use_gn,
                                           group_size_gn=group_size_gn)

        self.bottlenecks = nn.Sequential(
            *[YoloMSBottleneck(csp_branch_channels, csp_branch_channels, shortcut=shortcut_in_bottleneck,
                               kernel_sizes=kernel_sizes_bottleneck, use_gn=use_gn,
                               group_size_gn=group_size_gn, gn_eps=gn_eps)
              for _ in range(num_bottlenecks)]
        )

        self.conv_final_fusion = ConvNormAct(csp_branch_channels * 2, out_channels, kernel_size=1, use_gn=use_gn,
                                             group_size_gn=group_size_gn)

    def forward(self, x):
        x_processed = self.downsample_conv(x)
        if hasattr(self, 'projection_conv'):
            x_processed = self.projection_conv(x_processed)

        main_path = self.conv_main_path(x_processed)
        main_path = self.bottlenecks(main_path)

        short_path = self.conv_short_path(x_processed)

        concatenated = torch.cat([main_path, short_path], dim=1)
        return self.conv_final_fusion(concatenated)


class YoloMSBackbone(nn.Module):


    def __init__(self, input_channels=3, stem_out_channels=64,
                 stage_configs=None, use_gn=False, group_size_gn=16, gn_eps=1e-5):
        super().__init__()
        self.use_gn = use_gn
        self.group_size_gn = group_size_gn
        self.gn_eps = gn_eps


        self.stem = ConvNormAct(input_channels, stem_out_channels, kernel_size=3, stride=2, padding=1,
                                use_gn=self.use_gn, group_size_gn=self.group_size_gn, gn_eps=self.gn_eps)

        if stage_configs is None:

            stage_configs = [
                (2, 1, (3, 5), True),
                (4, 2, (3, 5), True),
                (8, 2, (3, 5), True),
            ]

        self.stages = nn.ModuleList()
        self.feature_info = []

        current_channels = stem_out_channels
        current_stride = 2

        for i, (ch_factor, num_b, k_sizes, downsample_this_stage) in enumerate(stage_configs):
            out_c = stem_out_channels * ch_factor
            stage = YoloMSCSPStage(
                current_channels, out_c,
                num_bottlenecks=num_b,
                kernel_sizes_bottleneck=k_sizes,
                use_gn=self.use_gn, group_size_gn=self.group_size_gn, gn_eps=self.gn_eps,
                downsample_first=downsample_this_stage
            )
            self.stages.append(stage)
            current_channels = out_c
            if downsample_this_stage:
                current_stride *= 2
            self.feature_info.append({'channels': current_channels, 'stride': current_stride, 'name': f'stage{i + 1}'})

    def forward(self, x):
        outputs = {}
        S1 = x
        x = self.stem(x)
        S2 = x

        for i, stage_module in enumerate(self.stages):
            x = stage_module(x)

            outputs[self.feature_info[i]['name']] = x
        return outputs,S1,S2



class AdaptiveGradualDownsampler(nn.Module):
    def __init__(self, input_channels, output_channels, target_h, target_w,
                 max_downsample_stages=3,
                 use_gn=True, group_size_gn=16, gn_eps=1e-5):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        self.output_channels = output_channels

        self.downsample_convs = nn.ModuleList()
        current_channels = input_channels
        for i in range(max_downsample_stages):

            self.downsample_convs.append(
                ConvNormAct(current_channels, self.output_channels,
                            kernel_size=3, stride=2, padding=1,
                            use_gn=use_gn, group_size_gn=group_size_gn, gn_eps=gn_eps)
            )
            current_channels = self.output_channels


        self.final_adaptive_pool = nn.AdaptiveAvgPool2d((target_h, target_w))


    def forward(self, x, mask_in):

        current_x = x

        current_mask = mask_in.float()
        for i, conv_stage in enumerate(self.downsample_convs):
            current_h, current_w = current_x.shape[2:]

            if current_h > self.target_h * 1.8 and current_w > self.target_w * 1.8:

                current_x = conv_stage(current_x)
                #
                current_mask = F.max_pool2d(current_mask, kernel_size=2, stride=2, ceil_mode=False)

            else:

                if current_x.shape[1] != conv_stage.conv.in_channels and i == 0:
                    current_x = conv_stage(current_x)
                break

        output_feature = self.final_adaptive_pool(current_x)

        output_mask_float = F.adaptive_max_pool2d(current_mask, (self.target_h, self.target_w))

        return output_feature, (output_mask_float > 0.0).bool()  # 返回bool掩码 (True for valid)


class YoloMSFeatureExtractor(nn.Module):

    def __init__(self, model_cfg ,output_channels_final_feat=256, output_h=16, output_w=32,
                 input_channels=3,

                 bb_stem_out_channels=32,
                 bb_stage_configs=None,


                 fpn_out_channels=128,
                 fpn_fuse_levels=('0', '1', '2'),  #

                 attention_mlp_hidden_dim=128,

                 use_gn=True, group_size_gn=16, gn_eps=1e-5,
                 cnn_pretrained=False
                 ):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.d_model = output_channels_final_feat
        self.fpn_fuse_levels = fpn_fuse_levels
        self.use_segmentation_module = model_cfg.get('use_segmentation_module', False)
        num_fuse_levels = len(fpn_fuse_levels)
        adaptive_downsampler_max_stages = model_cfg.get('adaptive_downsampler_max_stages',5)  #

        self.backbone = YoloMSBackbone(
            input_channels=input_channels,
            stem_out_channels=bb_stem_out_channels,
            stage_configs=bb_stage_configs,
            use_gn=use_gn, group_size_gn=group_size_gn, gn_eps=gn_eps
        )


        fpn_input_channels_list = [info['channels'] for info in self.backbone.feature_info]

        self.fpn_input_layer_names = [info['name'] for info in self.backbone.feature_info]


        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_input_channels_list,
            out_channels=fpn_out_channels

        )


        self.projection_layers = nn.ModuleDict()
        for level_key in self.fpn_fuse_levels:

            self.projection_layers[level_key] = ConvNormAct(
                fpn_out_channels,
                self.d_model,
                kernel_size=1,
                use_gn=use_gn, group_size_gn=group_size_gn, gn_eps=gn_eps
            )

        self.scale_attention_mlp = nn.Sequential(
            nn.Linear(num_fuse_levels * self.d_model, attention_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attention_mlp_hidden_dim, num_fuse_levels)
        )


        self.adaptive_final_downsampler = AdaptiveGradualDownsampler(
            input_channels=self.d_model,
            output_channels=self.d_model,
            target_h=output_h,
            target_w=output_w,
            max_downsample_stages=adaptive_downsampler_max_stages,
            use_gn=use_gn,
            group_size_gn=group_size_gn,
            gn_eps=gn_eps
        )
        if cnn_pretrained:
            print("Warning: YoloMSFeatureExtractor的cnn_pretrained=True The parameters are not used to load publicly available ResNet weights. The model will be randomly initialized.")

    def forward(self, x, input_mask):

        x = x * input_mask.float()
        #
        backbone_outputs_dict,S1,S2 = self.backbone(x)


        fpn_input_dict_for_tv_fpn = {str(i): backbone_outputs_dict[name] for i ,name in enumerate(self.fpn_input_layer_names)}
        #
        fpn_outputs_dict = self.fpn(fpn_input_dict_for_tv_fpn)#[B,128,hw(4,8,16)]

        S4 = fpn_outputs_dict['0']


        projected_features_list = []
        resized_features_list = []

        highest_res_key_in_fuse = self.fpn_fuse_levels[0]
        if highest_res_key_in_fuse not in fpn_outputs_dict:
            raise KeyError(
                f"Desired FPN fusion level '{highest_res_key_in_fuse}' Not in FPN output. Available keys: {list(fpn_outputs_dict.keys())}")

        #
        #
        target_spatial_size = self.projection_layers[highest_res_key_in_fuse](
            fpn_outputs_dict[highest_res_key_in_fuse]).shape[2:]

        for level_key in self.fpn_fuse_levels:
            if level_key not in fpn_outputs_dict:
                print(
                    f"Warning:Desired FPN fusion level  '{level_key}' Not in FPN output {list(fpn_outputs_dict.keys())},this level will be skipped.")
                continue


            proj_p = self.projection_layers[level_key](fpn_outputs_dict[level_key])  # [B, d_model=256, Hp, Wp]
            projected_features_list.append(proj_p)


            if proj_p.shape[2:] == target_spatial_size:
                resized_p = proj_p
            else:
                resized_p = F.interpolate(proj_p, size=target_spatial_size, mode='bilinear', align_corners=False)
            resized_features_list.append(resized_p)  #

        if not resized_features_list:
            raise ValueError("No FPN feature layers available for fusion. Please check the fpn_fuse_levels configuration and FPN output.")


        gap_features = [F.adaptive_avg_pool2d(p_resized, 1).flatten(start_dim=1) for p_resized in
                        resized_features_list]
        concatenated_gap = torch.cat(gap_features, dim=1)

        attention_scores = self.scale_attention_mlp(concatenated_gap)
        attention_weights = F.softmax(attention_scores, dim=1)


        fused_features = torch.zeros_like(resized_features_list[0])
        for i, p_resized in enumerate(resized_features_list):
            weight = attention_weights[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            fused_features += p_resized * weight

        original_input_key_for_highest_res_fpn = self.fpn_input_layer_names[
            int(highest_res_key_in_fuse)]

        stride_to_target_spatial_size = -1
        for info in self.backbone.feature_info:
            if info['name'] == original_input_key_for_highest_res_fpn:
                stride_to_target_spatial_size = info['stride']
                break
        if stride_to_target_spatial_size == -1:
            raise ValueError(f"{highest_res_key_in_fuse}")


        mask_at_fused_resolution = F.max_pool2d(input_mask.float(),
                                                kernel_size=stride_to_target_spatial_size,
                                                stride=stride_to_target_spatial_size,
                                                padding=0)
        if mask_at_fused_resolution.shape[2:] != target_spatial_size:
            mask_at_fused_resolution = F.adaptive_max_pool2d(mask_at_fused_resolution, target_spatial_size)


        output_x, final_mask_bool = self.adaptive_final_downsampler(fused_features,mask_at_fused_resolution
                                                                                  )  # [B, d_model, output_h, output_w]
        return output_x, final_mask_bool
class RefinedMemorySegmentationHead(nn.Module):
    def __init__(self, memory_d_model, encoder_feat_h, encoder_feat_w,
                 internal_channels=128, num_output_classes=1,
                 use_gn=True, group_size_gn=16, gn_eps=1e-5, act_fn='silu'):
        super().__init__()
        self.memory_d_model = memory_d_model
        self.encoder_feat_h = encoder_feat_h
        self.encoder_feat_w = encoder_feat_w


        self.conv_block = nn.Sequential(
            ConvNormAct(memory_d_model, internal_channels, kernel_size=3, stride=1,
                        use_gn=use_gn, group_size_gn=group_size_gn, gn_eps=gn_eps, act_fn=act_fn),
            ConvNormAct(internal_channels, internal_channels // 2, kernel_size=3, stride=1,
                        use_gn=use_gn, group_size_gn=group_size_gn, gn_eps=gn_eps, act_fn=act_fn)
        )
        self.final_conv = nn.Conv2d(internal_channels // 2, num_output_classes, kernel_size=1)



    def forward(self, memory_reshaped):
        x = self.conv_block(memory_reshaped)
        refined_seg_logits = self.final_conv(x)
        return refined_seg_logits

class ResNetFPNFeatureExtractor(nn.Module):
    def __init__(self, output_channels=256, output_h=16, output_w=32,
                 input_channels=3, pretrained=True, group_size=16, gn_eps=1e-5):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.use_gn = True


        if pretrained:

            backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18',
                                      weights='IMAGENET1K_V1' if pretrained else None)
        else:
            print("从头Initialize ResNet18...")
            backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=None)

        if self.use_gn:
            self._replace_bn_with_gn(backbone, group_size, gn_eps)

        if input_channels != 3:
            original_conv1_weight = backbone.conv1.weight.clone()
            backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:  # if使用了预训练权重
                if input_channels == 1 and original_conv1_weight.shape[1] == 3:

                    backbone.conv1.weight.data = original_conv1_weight.mean(dim=1, keepdim=True)
                else:
                    print(
                        f"Warning: Input channel count is {input_channels} However, the pre-trained weights are designed for 3 channels, so they may not be fully applicable. The first convolutional layer will be re-initialized.")
                    if input_channels != 3:
                        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        elif not pretrained:
            nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.body_conv1 = backbone.conv1

        self.body_norm1 = backbone.bn1
        self.body_relu = backbone.relu
        self.body_maxpool = backbone.maxpool
        self.body_layer1 = backbone.layer1  #
        self.body_layer2 = backbone.layer2
        self.body_layer3 = backbone.layer3

        fpn_in_channels_list_resnet = [
            self.body_layer1[-1].conv2.out_channels if hasattr(self.body_layer1[-1], 'conv2') else self.body_layer1[
                -1].out_channels,
            self.body_layer2[-1].conv2.out_channels if hasattr(self.body_layer2[-1], 'conv2') else self.body_layer2[
                -1].out_channels,
            self.body_layer3[-1].conv2.out_channels if hasattr(self.body_layer3[-1], 'conv2') else self.body_layer3[
                -1].out_channels,
        ]
        fpn_out_channels_resnet = 256
        self.fpn = FeaturePyramidNetwork(fpn_in_channels_list_resnet, fpn_out_channels_resnet)

        self.output_projection = nn.Sequential(
            ConvNormAct(fpn_out_channels_resnet, output_channels, kernel_size=1,
                        use_gn=self.use_gn, group_size_gn=group_size, gn_eps=gn_eps),

        )
        self.final_pool = nn.AdaptiveAvgPool2d((output_h, output_w))
        self.mask_final_pool = nn.AdaptiveMaxPool2d((output_h, output_w))

    def _replace_bn_with_gn(self, module, group_size, gn_eps):
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.BatchNorm2d):
                num_channels = child_module.num_features

                gn_layer = create_robust_gn(num_channels, group_size, eps=gn_eps)
                module.add_module(name, gn_layer)
            else:
                self._replace_bn_with_gn(child_module, group_size, gn_eps)

    def forward(self, x, input_mask):
        x = x * input_mask.float()


        res_x_s2 = self.body_maxpool(self.body_relu(self.body_norm1(self.body_conv1(x))))  # S/4


        fpn_input_dict_resnet = {}

        f_s4 = self.body_layer1(res_x_s2)
        fpn_input_dict_resnet['feat_s4'] = f_s4

        f_s8 = self.body_layer2(f_s4)
        fpn_input_dict_resnet['feat_s8'] = f_s8

        f_s16 = self.body_layer3(f_s8)
        fpn_input_dict_resnet['feat_s16'] = f_s16

        fpn_features = self.fpn(fpn_input_dict_resnet)

        projected_features = self.output_projection(fpn_features['0'])
        output_x = self.final_pool(projected_features)


        final_feat_stride = 4

        if input_mask.shape[2] < final_feat_stride or input_mask.shape[3] < final_feat_stride:
            mask_before_final_pool = F.adaptive_max_pool2d(input_mask.float(), projected_features.shape[2:])
        else:
            mask_before_final_pool = F.max_pool2d(input_mask.float(),
                                                  kernel_size=final_feat_stride,
                                                  stride=final_feat_stride,
                                                  padding=0)
        if mask_before_final_pool.shape[2:] != projected_features.shape[2:]:
            mask_before_final_pool = F.adaptive_max_pool2d(mask_before_final_pool, projected_features.shape[2:])

        final_mask = self.mask_final_pool(mask_before_final_pool)
        final_mask_bool = (final_mask > 0.0).bool()
        return output_x, final_mask_bool


class EnhancedIndustrialOCR(nn.Module):
    def __init__(self, vocab_size, model_cfg: Dict[str, Any],vocab_size_ctc = 20,blank_idx_ctc: int = 19):
        super().__init__()
        d_model = model_cfg.get('d_model', 256)
        dropout = model_cfg.get('dropout', 0.1)
        input_channels = model_cfg.get('input_channels', 3)

        self.encoder_input_h = model_cfg.get('feature_map_h', 16)
        self.encoder_input_w = model_cfg.get('feature_map_w', 32)

        cnn_group_size = model_cfg.get('cnn_group_size', 16)
        cnn_gn_eps = model_cfg.get('cnn_gn_eps', 1e-5)
        cnn_pretrained = model_cfg.get('cnn_pretrained', False)
        use_gn_in_cnn = model_cfg.get('cnn_use_gn', True)
        feature_extractor_type = model_cfg.get('feature_extractor_type', 'yoloms').lower()

        self.use_conditional_pe = model_cfg.get('use_conditional_pe', True)
        self.use_deformable_cpe = model_cfg.get('use_deformable_cpe', False)

        self.use_segmentation_module = model_cfg.get('use_segmentation_module', True)

        self.d_model = d_model
        self.pad_idx = model_cfg.get("pad_idx", 18)
        self.model_type = model_cfg.get("model_type", "attention_only")
        self.blank_idx_ctc = blank_idx_ctc

        if feature_extractor_type == 'yoloms':

            bb_stem_out_channels = model_cfg.get('yoloms_bb_stem_out_channels', 32)
            default_yoloms_stages = [
                {'ch_factor': 2, 'num_b': 1, 'k_sizes': (3, 5), 'downsample': True},
                {'ch_factor': 4, 'num_b': 2, 'k_sizes': (3, 5), 'downsample': True},
                {'ch_factor': 8, 'num_b': 2, 'k_sizes': (3, 5), 'downsample': True},
            ]
            raw_stage_configs = model_cfg.get('yoloms_bb_stage_configs_detailed', default_yoloms_stages)
            bb_stage_configs_tuples = [(s['ch_factor'], s['num_b'], tuple(s['k_sizes']), s['downsample']) for s in
                                       raw_stage_configs]
            fpn_out_channels_yoloms = model_cfg.get('yoloms_fpn_out_channels', 128)

            self.feature_extractor = YoloMSFeatureExtractor(
                model_cfg=model_cfg,
                output_channels_final_feat=d_model,
                output_h=self.encoder_input_h,
                output_w=self.encoder_input_w,
                input_channels=input_channels,
                bb_stem_out_channels=bb_stem_out_channels,
                bb_stage_configs=bb_stage_configs_tuples,
                fpn_out_channels=fpn_out_channels_yoloms,
                use_gn=use_gn_in_cnn, group_size_gn=cnn_group_size, gn_eps=cnn_gn_eps,
                cnn_pretrained=cnn_pretrained
            )
        elif feature_extractor_type == 'resnetfpn':

            self.feature_extractor = ResNetFPNFeatureExtractor(
                output_channels=d_model,
                output_h=self.encoder_input_h,
                output_w=self.encoder_input_w,
                input_channels=input_channels, pretrained=cnn_pretrained,
                group_size=cnn_group_size, gn_eps=cnn_gn_eps
            )
        else:
            raise ValueError(f": {feature_extractor_type}")

        if self.use_conditional_pe:
            if self.use_deformable_cpe:
                self.positional_encoder_2d = DeformableConditionalPositionalEncoding2D(
                    input_feature_channels=d_model, output_embedding_dim=d_model, model_cfg=model_cfg
                )
                print("Conditional positional encoding。")
            else:
                self.positional_encoder_2d = ConditionalPositionalEncoding2D(
                    input_feature_channels=d_model, output_embedding_dim=d_model,

                    hidden_dim=model_cfg.get('cpe_hidden_dim', 256),
                    num_conv_layers=model_cfg.get('cpe_num_conv_layers', 2)
                )
                print("Conditional positional encoding。")
        else:
            self.positional_encoder_2d = LearnedPositionalEncoding2D(
                embedding_dim=d_model,
                height=self.encoder_input_h,
                width=self.encoder_input_w
            )
            print("Learnable 2D absolute positional encoding。")

        encoder_nhead = model_cfg.get('encoder_nhead', 8)
        encoder_ffn_dim = model_cfg.get('encoder_dim_feedforward', d_model * 4)
        encoder_layers_num = model_cfg.get('encoder_num_layers', 3)
        transformer_activation = model_cfg.get('transformer_activation', 'relu')

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=encoder_nhead, dim_feedforward=encoder_ffn_dim,
            dropout=dropout, activation=transformer_activation
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers_num,
            norm=nn.LayerNorm(d_model)
        )

        if self.model_type == "hybrid":
            self.ctc_fc = nn.Linear(d_model, vocab_size_ctc)


        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_1d_pe = PositionalEncoding(
            d_model=d_model, max_len=model_cfg.get('max_len', 19)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        decoder_nhead = model_cfg.get('decoder_nhead', 8)
        decoder_ffn_dim = model_cfg.get('decoder_dim_feedforward', d_model * 4)
        decoder_layers_num = model_cfg.get('decoder_num_layers', 4)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=decoder_nhead, dim_feedforward=decoder_ffn_dim,
            dropout=dropout, activation=transformer_activation
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layers_num,
            norm=nn.LayerNorm(d_model)
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        if self.use_segmentation_module:
            self.refined_memory_segmentation_head = RefinedMemorySegmentationHead(
                memory_d_model=d_model,
                encoder_feat_h=self.encoder_input_h,
                encoder_feat_w=self.encoder_input_w,
                internal_channels=model_cfg.get('refined_seg_internal_channels', d_model // 2),
                num_output_classes=1,
                use_gn=use_gn_in_cnn, group_size_gn=cnn_group_size, gn_eps=cnn_gn_eps,
                act_fn=model_cfg.get('cnn_act_fn', 'silu')
            )
            print("RefinedMemorySegmentationHead")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def u_shaped_segmentation_penalty(
            self,
            seg_logits: torch.Tensor,
            target_min_proportion: float = 0.05,
            target_max_proportion: float = 0.95,
            penalty_weight: float = 1.0
    ) -> torch.Tensor:

        if not ((0.0 < target_min_proportion < target_max_proportion) and (target_max_proportion <= 1.0)):
            raise ValueError("target_min_proportion and target_max_proportion into [0, 1] ，and min < max.")


        seg_probs = torch.sigmoid(seg_logits)

        dims_to_average = tuple(range(1, seg_probs.dim()))

        avg_fg_prob_per_image = torch.mean(seg_probs, dim=dims_to_average)  # shape: (B,)

        penalty_too_low = torch.pow(F.relu(target_min_proportion - avg_fg_prob_per_image), 2)


        penalty_too_high = torch.pow(F.relu(avg_fg_prob_per_image - target_max_proportion), 2)


        total_penalty_per_image = penalty_too_low + penalty_too_high
        batch_penalty = torch.mean(total_penalty_per_image)

        return penalty_weight * batch_penalty

    def forward(self, src, tgt_input, src_img_mask=None,batch_image_id = None,return_visualizations = False):

        if src_img_mask is None:
            raise ValueError("src_img_mask must not be None")

        x_feat_for_encoder, encoder_input_mask_bool = self.feature_extractor(src, src_img_mask)
        B, D, H_enc, W_enc = x_feat_for_encoder.shape


        x_pe_enc, cpe_visualization_data = self.positional_encoder_2d(x_feat_for_encoder)


        memory_src = x_pe_enc.flatten(2).permute(2, 0, 1)

        encoder_padding_mask_for_encoder_and_decoder = ~(encoder_input_mask_bool.squeeze(1).flatten(start_dim=1))


        memory = self.encoder(memory_src,
                              src_key_padding_mask=encoder_padding_mask_for_encoder_and_decoder)

        ctc_input_lengths = (~encoder_padding_mask_for_encoder_and_decoder).sum(dim=1)


        refined_seg_logits_for_loss = None
        if self.use_segmentation_module and hasattr(self, 'refined_memory_segmentation_head'):
            memory_reshaped = memory.permute(1, 2, 0).reshape(B, D, H_enc, W_enc)
            refined_seg_logits_for_loss = self.refined_memory_segmentation_head(memory_reshaped)


        ctc_log_probs = None
        if self.model_type == "hybrid":  #
            ctc_logits_from_memory = self.ctc_fc(memory)
            ctc_log_probs = F.log_softmax(ctc_logits_from_memory, dim=2)


        tgt_embedded = self.tgt_embedding(tgt_input) * math.sqrt(self.d_model)  #
        tgt_permuted = tgt_embedded.permute(1, 0, 2)  #
        tgt_pe = self.tgt_1d_pe(tgt_permuted)  #
        tgt_pe = self.embedding_dropout(tgt_pe)  #


        tgt_seq_len = tgt_pe.shape[0]  #
        tgt_mask_causal = generate_square_subsequent_mask(tgt_seq_len, device=tgt_pe.device)  #
        tgt_key_padding_mask = (tgt_input == self.pad_idx)  #
        tgt_mask_causal_bool = tgt_mask_causal == float('-inf')


        output_decoder = self.decoder(  #
            tgt=tgt_pe, memory=memory,
            tgt_mask=tgt_mask_causal_bool,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_padding_mask_for_encoder_and_decoder
        )


        final_output_logits = self.fc_out(output_decoder)  # [L_tgt_in, B, VocabSize]


        if return_visualizations:
            return (ctc_log_probs, final_output_logits,
                    src, x_feat_for_encoder, cpe_visualization_data, x_pe_enc,
                    refined_seg_logits_for_loss,
                    ctc_input_lengths)
        else:
            return ctc_log_probs, final_output_logits, refined_seg_logits_for_loss, ctc_input_lengths


    @torch.no_grad()
    def predict(self, src, src_img_mask, max_len, sos_idx, eos_idx, pad_idx):
        self.eval()
        B = src.shape[0]
        device = src.device

        if src_img_mask is None:
            src_img_mask = torch.ones_like(src[:, :1, :, :], dtype=torch.bool, device=src.device)

        x_feat_for_encoder, encoder_input_mask_bool = self.feature_extractor(src, src_img_mask)
        x_pe_enc, _ = self.positional_encoder_2d(x_feat_for_encoder)  #

        memory_src = x_pe_enc.flatten(2).permute(2, 0, 1)  #
        encoder_padding_mask_for_encoder_and_decoder = ~(encoder_input_mask_bool.squeeze(1).flatten(start_dim=1))  #
        memory = self.encoder(memory_src, src_key_padding_mask=encoder_padding_mask_for_encoder_and_decoder)  #

        tgt_tokens = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)  #

        for _ in range(max_len - 1):  #
            tgt_embedded = self.tgt_embedding(tgt_tokens) * math.sqrt(self.d_model)  #
            tgt_permuted = tgt_embedded.permute(1, 0, 2)  #
            tgt_pe = self.tgt_1d_pe(tgt_permuted)  #

            current_len = tgt_pe.size(0)  #
            tgt_mask_causal = generate_square_subsequent_mask(current_len, device=device)  #

            current_tgt_key_padding_mask = (tgt_tokens == pad_idx)  #

            output_decoder = self.decoder(  #
                tgt=tgt_pe, memory=memory,
                tgt_mask=tgt_mask_causal,
                tgt_key_padding_mask=current_tgt_key_padding_mask if current_tgt_key_padding_mask.any() else None,
                memory_key_padding_mask=encoder_padding_mask_for_encoder_and_decoder
            )

            logits = self.fc_out(output_decoder[-1, :, :])  #
            next_token = logits.argmax(dim=-1)  #
            tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(1)], dim=1)  #

            if ((next_token == eos_idx) | (tgt_tokens.size(1) >= max_len)).all():  #
                break

        return tgt_tokens  #



def show_feats(src,
               tensor_to_plot1, tensor_to_plot2, tensor_to_plot3,
               title1="Plot 1", title2="Plot 2", title3="Plot 3",
               sample_idx=0,
               H_feat=16, W_feat=32,
               suptitle_prefix=""):


    original_image_tensor = src[sample_idx].cpu().detach()
    original_image_to_viz = original_image_tensor.permute(1, 2, 0).numpy()
    cmap_original = None
    if original_image_to_viz.shape[2] == 1:
        original_image_to_viz = original_image_to_viz.squeeze(axis=2)
        cmap_original = 'gray'
    original_image_to_viz = np.clip(original_image_to_viz, 0, 1)

    plots_data = [
        (title1, tensor_to_plot1),
        (title2, tensor_to_plot2),
        (title3, tensor_to_plot3)
    ]

    num_total_plots = 1 + len(plots_data)
    fig, axes = plt.subplots(1, num_total_plots, figsize=(6 * num_total_plots, 5.5))
    if num_total_plots == 1:
        axes = [axes]
    axes = axes.flatten()

    axes[0].imshow(original_image_to_viz, cmap=cmap_original)
    axes[0].set_title(f"Original Image (Sample {sample_idx})")
    axes[0].axis('off')


    for i, (title, tensor_batch) in enumerate(plots_data):
        ax = axes[i + 1]
        if tensor_batch is None:
            ax.set_title(f"{title}\n(Data is empty.)")
            ax.axis('off')
            continue

        tensor_s = tensor_batch[sample_idx].cpu().detach()

        data_to_display = None
        cmap_selected = 'viridis'
        show_colorbar = True

        if tensor_s.dtype == torch.bool:
            cmap_selected = 'gray'
            show_colorbar = False
            if tensor_s.ndim == 1 and H_feat is not None and W_feat is not None:
                if tensor_s.numel() == H_feat * W_feat:
                    try:
                        data_to_display = tensor_s.reshape(H_feat, W_feat).numpy()
                    except Exception as e:
                        print(f"Warning:  '{title}' ({H_feat},{W_feat}) failed: {e}")
                        data_to_display = None
                else:
                    print(f"Warning: bool-mask '{title}' Number of elements {tensor_s.numel()} Compared to the target size {H_feat}x{W_feat} do not match。")
                    data_to_display = None
            elif tensor_s.ndim == 2:
                data_to_display = tensor_s.numpy()
            elif tensor_s.ndim == 3 and tensor_s.shape[0] == 1:
                data_to_display = tensor_s.squeeze(0).numpy()
            else:
                if tensor_s.numel() > 0:
                    data_to_display = tensor_s.float().mean(dim=0).numpy() if tensor_s.ndim > 1 and tensor_s.shape[
                        0] > 1 else tensor_s.float().numpy()
                else:
                    data_to_display = np.array([[0]])

        elif tensor_s.ndim == 3:
            if tensor_s.shape[0] == 1:
                data_to_display = tensor_s.squeeze(0).numpy()

                unique_vals = torch.unique(tensor_s)
                if len(unique_vals) < 5 and torch.all((unique_vals == 0) | (unique_vals == 1)):
                    cmap_selected = 'gray'
                    show_colorbar = False
            else:
                data_to_display = tensor_s.mean(dim=0).numpy()
        elif tensor_s.ndim == 2:
            data_to_display = tensor_s.numpy()

            unique_vals = torch.unique(tensor_s)
            if len(unique_vals) < 5 and torch.all((unique_vals == 0) | (unique_vals == 1)):
                cmap_selected = 'gray'
                show_colorbar = False

        if data_to_display is not None and data_to_display.ndim == 2:
            try:
                im = ax.imshow(data_to_display, cmap=cmap_selected, aspect='auto')
                ax.set_title(title)
                if show_colorbar:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            except TypeError as e_imshow:
                ax.set_title(f"{title}\n(Drawing error: {e_imshow})")

            except Exception as e_gen:
                ax.set_title(f"{title}\n(Unknown drawing error)")
        elif data_to_display is not None and data_to_display.ndim != 2:
            ax.set_title(f"{title}\n(Data is not 2D: {data_to_display.shape})")

        else:
            ax.set_title(f"{title}\n(Unable to display)")


        ax.axis('off')

    fig.suptitle(f"{suptitle_prefix}vis - Sample_index {sample_idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    plt.show(block=False)
    plt.pause(1)
    plt.close(fig)

