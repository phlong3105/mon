#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GCE-Net models.

./run.sh gcenet none none train 100 sice-zerodce all vision/enhance/llie no last
"""

from __future__ import annotations

__all__ = [
    "IPT",
]

from functools import partial
from typing import Any

import kornia
import torch
from torch import Tensor

import mon
from mon.globals import MODELS, LAYERS
from mon import core, nn
from mon.core.typing import _callable, _size_2_t
from mon.nn import functional as F
from mon.vision.enhance.derain import base
from mon.vision.feature import OPEmbedder
from mon.vision.enhance.derain.modules.ipt_layers import MeanShift, default_conv, ResBlock, Upsampler
import copy

math = core.math
console = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Loss
class Loss(nn.Loss):

    def __init__(
            self,
            weight_L1: float = 1,
            weight_MSE: float = 0.1,
            reduction: str = "mean",
            verbose: bool = False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.weight_L1 = weight_L1
        self.weight_MSE = weight_MSE
        self.loss_l1 = nn.L1Loss(reduction=reduction)
        self.loss_mse = nn.MSELoss(reduction=reduction)

    def __str__(self) -> str:
        return f"loss"

    def forward(
            self,
            pred: torch.Tensor | list[torch.Tensor],
            target: torch.Tensor | list[torch.Tensor],
            **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_l1 = self.loss_l1(input=pred, target=target) if self.weight_L1 > 0 else 0
        loss_mse = self.loss_mse(input=pred, target=target) if self.weight_MSE > 0 else 0

        loss = (
                self.weight_L1 * loss_l1
                + self.weight_MSE * loss_mse
        )

        if self.verbose:
            console.log(f"{self.loss_l1.__str__():<30} : {loss_l1}")
            console.log(f"{self.loss_mse.__str__():<30} : {loss_mse}")
        return loss, pred
# endregion


# region Model

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_queries,
            positional_encoding_type="learned",
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos

        if not self.mlp:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, query_idx, con=False):

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0,
                                                                                                           1).contiguous()

        if not self.mlp:
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos)
            x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


@MODELS.register(name="ipt")
class IPT(base.DerainingModel):
    """IPT (Image Processing Transformer) model.
    
    See Also: :class:`mon.vision.enhance.derain.base.DerainingModel`
    """

    zoo = {}

    def __init__(
            self,
            n_feats: int = 64,
            patch_size: int = 48,
            shift_mean: bool = True,
            loss: Any = Loss(),
            transformer_config: dict[str, Any] = None,
            n_colors: int = 3,
            rgb_range: int = 255,
            scale: list[int] = 1,
            conv=default_conv,
            name: str = "ipt",
            *args, **kwargs
    ):
        transformer_config = transformer_config if transformer_config is not None else {}

        self.name = name
        super().__init__(
            *args, **kwargs
        )
        self.loss = loss
        self.scale_idx = 0
        self.n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(inplace=True)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(n_colors, n_feats, kernel_size),
                ResBlock(conv, n_feats, 5, act=act),
                ResBlock(conv, n_feats, 5, act=act),
            ) for _ in scale
        ])

        self.body = VisionTransformer(
            img_dim=patch_size,
            patch_dim=transformer_config.get("patch_dim", 3),
            num_channels=n_feats,
            embedding_dim=n_feats * transformer_config.get("patch_dim", 3) * transformer_config.get("patch_dim", 3),
            num_heads=transformer_config.get("num_heads", 12),
            num_layers=transformer_config.get("num_layers", 12),
            hidden_dim=n_feats * transformer_config.get("patch_dim", 3) * transformer_config.get("patch_dim", 3) * 4,
            num_queries=transformer_config.get("num_queries", 1),
            dropout_rate=transformer_config.get("dropout_rate", 0),
            mlp=transformer_config.get("no_mlp", False),
            pos_every=transformer_config.get("pos_every", False),
            no_pos=transformer_config.get("no_pos", False),
            no_norm=transformer_config.get("no_norm", False)
        )

        self.tail = nn.ModuleList([
            nn.Sequential(
                Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, n_colors, kernel_size)
            ) for s in scale
        ])

    @property
    def config_dir(self) -> core.Path:
        return core.Path(__file__).absolute().parent / "config"

    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02

    def forward_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor | None,
            *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.
            
        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss, self.previous = self.loss(input, pred, self.previous) if self.loss else (None, None)
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss

    def forward(
            self,
            input: torch.Tensor,
            augment: bool = False,
            profile: bool = False,
            out_index: int = -1,
            *args, **kwargs
    ) -> Tensor:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape :math`[B, C, H, W]`.
            augment: If ``True``, perform test-time augmentation. Default:
                ``False``.
            profile: If ``True``, Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: -1 means the last layer.

        Return:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            return self.forward_once(input=input, profile=profile, *args, **kwargs)
        else:
            return self.forward_once(input=input, profile=profile, *args, **kwargs)

    def forward_once(
            self,
            input: torch.Tensor,
            profile: bool = False,
            out_index: int = -1,
            *args, **kwargs
    ):
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.
                
        Return:
            Predictions.
        """
        x = input
        x = self.sub_mean(x)
        x = self.head[self.scale_idx](x)

        res = self.body(x, self.scale_idx)
        res += x

        x = self.tail[self.scale_idx](res)
        x = self.add_mean(x)
        return x

    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss

# endregion
