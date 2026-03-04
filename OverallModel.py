from pickle import TRUE
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from basicModule import *
import numpy as np
from VolFormer import *                      #VolFormer


class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()
        kernel_size = 3

        self.net1 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale)
        self.net2 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale)
        self.net3 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale)

    def forward(self, x):
        res = self.net1(x)
        res = self.net2(res)
        res = self.net3(res)
        res += x
        return res


class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_outputs, n_blocks, act, res_scale, up_scale, use_tail=True, conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        self.head = conv(n_colors, n_feats, kernel_size)
        self.body = SSPN(n_feats, n_blocks, act, res_scale)

        self.upsample = Upsampler(conv, up_scale, n_feats)
        self.tail = None
        if use_tail:
            self.tail = conv(n_feats, n_outputs, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)
        return y


class General_VolFormer(nn.Module):
    def __init__(
        self,
        n_subs,
        n_ovls,
        n_colors,
        n_blocks,
        n_feats,
        n_scale,
        res_scale,
        use_share=True,
        conv=default_conv,
        vf_embed_dim: int = 60,
        vf_depth: int = 2,
        vf_stages: int = 4,
        vf_num_heads: int = 2,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ) -> object:
        super(General_VolFormer, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        self.sca = n_scale
        self.window_size = int(window_size)

        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        depths = [int(vf_depth)] * int(vf_stages)
        num_heads = [int(vf_num_heads)] * int(vf_stages)

        self.branch1 = VolFormer(
            in_chans=n_subs,
            out_chans=n_subs,
            window_size=self.window_size,
            img_range=1,
            depths=depths,
            embed_dim=int(vf_embed_dim),
            num_heads=num_heads,
            mlp_ratio=float(mlp_ratio),
            upscale=n_scale,
        )

        self.trunk = BranchUnit(n_colors, n_feats, n_subs, n_blocks, act, res_scale, up_scale=1, use_tail=False, conv=default_conv)
        self.skip_conv = conv(n_colors, n_feats, kernel_size)
        self.final = conv(n_feats, n_colors, kernel_size)

        self.trunk_RGB = BranchUnit(n_subs, n_feats, n_feats, n_blocks, act, res_scale, up_scale=1, use_tail=False, conv=default_conv)
        self.skip_conv_RGB = conv(n_subs, n_feats, kernel_size)
        self.final_RGB = conv(n_feats, n_subs, kernel_size)

    def _pad_to_window(self, x: torch.Tensor):
        ws = self.window_size
        _, _, h, w = x.shape
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, h, w)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w, h, w)

    def forward(self, x, lms, modality, img_size=None):
        b, c, h, w = x.shape

        # pad LR (ms) to multiples of window_size
        x_pad, meta = self._pad_to_window(x)

        # IMPORTANT: lms is HR size; to keep shapes consistent after padding,
        # we recompute lms from padded LR.
        lms_pad = F.interpolate(x_pad, scale_factor=self.sca, mode="bicubic", align_corners=False)

        if modality == "spectral":
            y_pad = x_pad.new_zeros(b, c, self.sca * x_pad.shape[2], self.sca * x_pad.shape[3])

            channel_counter = x_pad.new_zeros(c)
            for g in range(self.G):
                sta_ind = self.start_idx[g]
                end_ind = self.end_idx[g]

                xi = x_pad[:, sta_ind:end_ind, :, :]
                xi = self.branch1(xi, img_size)
                y_pad[:, sta_ind:end_ind, :, :] += xi
                channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

            y_pad = y_pad / channel_counter.unsqueeze(1).unsqueeze(2)

            y_pad = self.trunk(y_pad)
            y_pad = y_pad + self.skip_conv(lms_pad)
            y_pad = self.final(y_pad)

            return y_pad[:, :, : (h * self.sca), : (w * self.sca)]

        elif modality == "rgb":
            y_pad = self.branch1(x_pad, img_size)
            y_pad = self.trunk_RGB(y_pad)
            y_pad = y_pad + self.skip_conv_RGB(lms_pad)
            y_pad = self.final_RGB(y_pad)
            return y_pad[:, :, : (h * self.sca), : (w * self.sca)]

        else:
            raise ("Not implemented!!!")
