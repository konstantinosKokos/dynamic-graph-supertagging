import pdb

import torch
from torch.nn import Module, Embedding, Linear, Dropout, ModuleList, Parameter
from torch import Tensor
from math import sqrt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as sparse_softmax, remove_isolated_nodes
from torch_geometric.typing import OptTensor
from torch import Tensor, sigmoid


class DecoderLayer(Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, cross_heads: int, self_heads: int, dropout_rate: float):
        super().__init__()
        self.nodes_to_root = CrossMHA(encoder_dim, decoder_dim, cross_heads, dropout_rate)
        self.ntr_norm = RMSNorm(encoder_dim)
        self.roots_to_root = SelfMHA(encoder_dim, self_heads, dropout_rate)
        self.rtr_norm = RMSNorm(encoder_dim)
        self.ffn = SwiGLU(encoder_dim, int(8 / 3 * encoder_dim))
        self.ffn_norm = RMSNorm(encoder_dim)
        self.root_to_fringe = CrossMHA(decoder_dim, encoder_dim, cross_heads, dropout_rate)

    def forward(self,
                root_features: Tensor,
                node_features: Tensor,
                fringe_features: Tensor,
                node_to_root_index: Tensor,
                root_to_root_index: Tensor,
                root_to_fringe_index: Tensor,
                root_edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        # todo: avoid updating inactive roots
        ntr = self.nodes_to_root.forward(xs=root_features, ctx=node_features, edge_index=node_to_root_index)
        ntr = self.ntr_norm(root_features + ntr)
        rtr = self.roots_to_root.forward(xs=ntr, edge_index=root_to_root_index, edge_attr=root_edge_attr)
        rtr = self.rtr_norm(ntr + rtr)
        ffn = self.ffn(root_features)
        ffn = self.ffn_norm(ffn + rtr)
        return ffn, self.root_to_fringe.forward(xs=fringe_features, ctx=root_features, edge_index=root_to_fringe_index)


class SelfMHA(MessagePassing):
    def __init__(self, dim: int, num_heads: int, dropout_rate: float):
        super(SelfMHA, self).__init__(aggr="add", node_dim=0)
        assert dim % num_heads == 0
        self.w_qkv = Linear(dim, 3 * dim, bias=False)
        self.w_out = Linear(dim, dim, bias=False)
        self.dropout = Dropout(dropout_rate)
        self.num_heads = num_heads
        self.dim = dim
        self.hdim = dim // num_heads

    def forward(self, xs: Tensor, edge_index: Tensor, edge_attr: Tensor):
        qs, ks, vs = self.w_qkv(xs).view(-1, self.num_heads, self.hdim, 3).chunk(3, dim=-1)
        qs, ks, vs = qs.squeeze(-1), ks.squeeze(-1), vs.squeeze(-1)
        out = self.propagate(edge_index, qs=qs, ks=ks, vs=vs, edge_attr=edge_attr)
        return self.w_out(self.dropout(out).view(-1, self.dim))

    def message(self, qs_i: Tensor, ks_j: Tensor, vs_j: Tensor, edge_attr: Tensor, index: Tensor):
        atn = (qs_i * ks_j * edge_attr).sum(dim=-1)
        atn = sparse_softmax(atn, index, dim=0)
        return atn.unsqueeze(-1) * vs_j


class CrossMHA(MessagePassing):
    def __init__(self, self_dim: int, ctx_dim: int, num_heads: int, dropout_rate: float):
        super(CrossMHA, self).__init__(aggr='add', node_dim=0)
        assert self_dim % num_heads == 0
        self.w_q = Linear(self_dim, self_dim, bias=False)
        self.w_kv = Linear(ctx_dim, 2 * self_dim, bias=False)
        self.w_out = Linear(self_dim, self_dim, bias=False)
        self.dropout = Dropout(dropout_rate)
        self.num_heads = num_heads
        self.dim = self_dim
        self.hdim = self_dim // num_heads

    def forward(self, xs: Tensor, ctx: Tensor, edge_index: Tensor):
        qs = self.w_q(xs).view(-1, self.num_heads, self.hdim)
        ks, vs = self.w_kv(ctx).view(-1, self.num_heads, self.hdim, 2).chunk(2, dim=-1)
        ks, vs = ks.squeeze(-1), vs.squeeze(-1)
        out = self.propagate(edge_index, qs=qs, ks=ks, vs=vs)
        return out

    def message(self, qs_i: Tensor, ks_j: Tensor, vs_j: Tensor, index: Tensor) -> Tensor:
        atn = (qs_i * ks_j).sum(dim=-1) / sqrt(self.hdim)
        atn = sparse_softmax(atn, index, dim=0)
        out = self.dropout(atn.unsqueeze(-1) * vs_j)
        return self.w_out(out.view(-1, self.dim))


def swish(x: Tensor, b: int = 1) -> Tensor:
    return x * sigmoid(b * x)


class SwiGLU(Module):
    def __init__(self, input_dim: int, interm_dim: int):
        super(SwiGLU, self).__init__()
        self.w_in = Linear(input_dim, interm_dim, bias=False)
        self.v = Linear(input_dim, interm_dim, bias=False)
        self.w_out = Linear(interm_dim, input_dim, bias=False)

    def forward(self, x: Tensor):
        interm = self.w_in(x)
        interm = swish(interm) * self.v(x)
        return self.w_out(interm)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# class SMHA2D(MessagePassing):
#     def __init__(self, num_heads: int, dim: int):
#         super(SMHA2D, self).__init__(aggr='add', node_dim=0)
#         self.num_heads = num_heads
#         self.dim = dim
#         self.hdim = dim // num_heads
#         self.w_q = Linear(dim, dim, bias=False)
#         self.w_kv = Linear(dim, 2 * dim, bias=False)
#         self.w_out = Linear(dim, dim, bias=False)
#
#     def forward(self,
#                 target_features: Tensor,
#                 source_features: Tensor,
#                 edge_index: Tensor,
#                 edge_attr: Tensor,
#                 prev_atn: OptTensor = None,
#                 layer_depth: int = 0) -> tuple[Tensor, Tensor, int]:
#         qs = self.w_q(target_features).view(-1, self.num_heads, self.hdim)
#         ks, vs = self.w_kv(source_features).view(-1, self.num_heads, self.hdim, 2).chunk(2, dim=-1)
#         out = self.propagate(
#             edge_index, qs=qs, ks=ks.squeeze(-1), vs=vs.squeeze(-1),
#             edge_attr=edge_attr,
#             size=None,
#             prev_atn=prev_atn,
#             layer_depth=layer_depth)
#         prev_atn = self.get_buffer('prev_atn')
#         self.register_buffer('prev_atn', None)
#         pdb.set_trace()
#         return self.w_out(out.flatten(-2)), prev_atn, layer_depth + 1
#
#     def message(self,
#                 qs_i: Tensor,
#                 ks_j: Tensor,
#                 vs_j: Tensor,
#                 edge_attr: Tensor,
#                 prev_atn: Tensor,
#                 index: Tensor,
#                 layer_depth: int) -> Tensor:
#         atn = (qs_i * ks_j * edge_attr.unsqueeze(1)).sum(dim=-1) / sqrt(self.hdim)  # num_edges x num_heads
#         if prev_atn is not None:
#             atn = (atn + layer_depth * prev_atn) / (layer_depth + 1)
#         softmax_atn = sparse_softmax(atn, index, dim=0)
#         self.register_buffer('prev_atn', atn)
#         return softmax_atn.unsqueeze(-1) * vs_j




# class SelfMHA2D(Module):
#     def __init__(self, num_heads: int, dim: int, dropout_rate: float):
#         super(SelfMHA2D, self).__init__()
#         assert dim % num_heads == 0
#         self.dim = dim
#         self.num_heads = num_heads
#         self.hdim = dim // num_heads
#         self.w_in = Linear(dim, 3 * dim, bias=False)
#         self.w_out = Linear(dim, dim, bias=False)
#         self.dropout = Dropout(dropout_rate)
#
#     def forward(self,
#                 xs: Tensor,
#                 atn_mask: Tensor,
#                 alphas: Tensor,
#                 prev: Tensor | None = None,
#                 num_layers: int = 0) -> tuple[Tensor, Tensor, int]:
#         b, n, d = xs.shape
#         qs, ks, vs = self.w_in(xs).view(b, n, self.hdim, self.num_heads, 3).chunk(3, -1)
#         qs = qs.view(b, n, self.hdim, self.num_heads)
#         ks = ks.view(b, n, self.hdim, self.num_heads)
#         atn = contract('bndh,nNd,bNdh->bnNh', qs, alphas, ks)
#         if prev is not None:
#             atn = (atn + num_layers * prev) / (num_layers + 1)
#         atn = atn.masked_fill(atn_mask.view(b, n, n, 1) == 0, -1e10) / sqrt(self.hdim)
#         softmax_atn = atn.softmax(-2)
#         mha = contract('bnNh,bNdh->bndh', softmax_atn, vs.squeeze()).flatten(-2)
#         return self.w_out(self.dropout(mha)), atn, num_layers + 1



# class TreeDecoder(Module):
#     def __init__(self, num_nodes: int, max_tree_depth: int, enc_dim: int, dec_dim: int, enc_heads: int,
#                  dec_heads: int, num_layers: int, max_distance: int, dropout_rate: float):
#         super(TreeDecoder, self).__init__()
#         self.decoder = Decoder2D(enc_dim, dec_dim, enc_heads, dec_heads, num_layers, max_distance, dropout_rate)
#         self.static_embedder = Embedding(num_nodes, dec_dim)
#         self.tree_embedder = TreePE(max_tree_depth, dec_dim)
#
#     def decode_train(self,
#                      node_ids: Tensor,
#                      node_encodings: Tensor,
#                      ctx: Tensor,
#                      self_mask: Tensor,
#                      cross_mask: Tensor) -> Tensor:
#         pdb.set_trace()
#         xs = self.static_embedder(node_ids) + self.tree_embedder(node_encodings)
#         return self.decoder(xs, ctx, self_mask, cross_mask)
#
#
#
#

#
#
# class Decoder2D(Module):
#     def __init__(self, enc_dim: int, dec_dim: int, enc_heads: int, dec_heads: int,
#                  num_layers: int, max_distance: int, dropout_rate: float):
#         super(Decoder2D, self).__init__()
#         self.layers = ModuleList([
#             DecoderLayer2D(enc_dim, dec_dim, enc_heads, dec_heads, dropout_rate) for _ in range(num_layers)])
#         self.pos_emb = RelativePE(max_distance, dec_dim//dec_heads)
#
#     def forward(self,
#                 xs: Tensor,
#                 ctx: Tensor,
#                 self_mask: Tensor,
#                 cross_mask: Tensor) -> Tensor:
#         alphas, self_atn, cross_atn, n = self.pos_emb(xs.shape[1]), None, None, 0
#         for layer in self.layers:
#             xs, self_atn, cross_atn, n = layer(xs, ctx, self_mask, cross_mask, alphas, self_atn, cross_atn, n)
#         return xs
#
#
# class DecoderLayer2D(Module):
#     def __init__(self, enc_dim: int, dec_dim: int, enc_heads: int, dec_heads: int, dropout_rate: float):
#         super(DecoderLayer2D, self).__init__()
#         self.self_atn = SelfMHA2D(num_heads=dec_heads, dim=dec_dim, dropout_rate=dropout_rate)
#         self.cross_atn = CrossMHA2D(num_heads=enc_heads, enc_dim=enc_dim, dec_dim=dec_dim, dropout_rate=dropout_rate)
#         self.ffn = SwiGLU(dec_dim, round(dec_dim * 8/3))
#         self.ln_self_atn = RMSNorm(dec_dim)
#         self.ln_cross_atn = RMSNorm(dec_dim)
#         self.ln_out = RMSNorm(dec_dim)
#
#     def forward(self,
#                 xs: Tensor,
#                 ctx: Tensor,
#                 self_mask: Tensor,
#                 cross_mask: Tensor,
#                 alphas: Tensor,
#                 prev_self: Tensor,
#                 prev_cross: Tensor,
#                 num_layers: int) -> tuple[Tensor, Tensor, Tensor, int]:
#
#         self_atn, prev_self, _ = self.self_atn.forward(xs, self_mask, alphas, prev_self, num_layers)
#         self_atn = self.ln_self_atn(self_atn)
#         xs = xs + self_atn
#         cross_atn, prev_cross, n = self.cross_atn.forward(xs, ctx, cross_mask, prev_cross, num_layers)
#         cross_atn = self.ln_cross_atn(cross_atn)
#         xs = xs + cross_atn
#         ffn = self.ffn(xs)
#         ffn = self.ln_out(ffn)
#         xs = xs + ffn
#         return xs, prev_self, prev_cross, n
#
#
# def contract(expr: str, *args: Tensor) -> Tensor:
#     return _contract(expr, *args)  # type: ignore
#

#
#
# class CrossMHA2D(Module):
#     def __init__(self, num_heads: int, enc_dim: int, dec_dim: int, dropout_rate: float):
#         super(CrossMHA2D, self).__init__()
#         self.num_heads = num_heads
#         self.enc_dim = enc_dim
#         self.dec_dim = dec_dim
#         self.w_enc = Linear(enc_dim, 2 * dec_dim, bias=False)
#         self.w_dec = Linear(dec_dim, dec_dim, bias=False)
#         self.w_out = Linear(dec_dim, dec_dim, bias=False)
#         self.dropout = Dropout(dropout_rate)
#
#     def forward(self,
#                 xs: Tensor,
#                 ctx: Tensor,
#                 atn_mask: Tensor,
#                 prev: Tensor | None = None,
#                 num_layers: int = 0) -> tuple[Tensor, Tensor, int]:
#         (b, n_x, d), n_ctx = xs.shape, ctx.shape[1]
#         ks, vs = self.w_enc(ctx).view(b, n_ctx, self.dec_dim // self.num_heads, self.num_heads, 2).chunk(2, -1)
#         qs = self.w_dec(xs).view(b, n_x, self.dec_dim // self.num_heads, self.num_heads)
#         atn = contract('bndh,bNdh->bnNh', qs, ks.squeeze())/sqrt(self.dec_dim//self.num_heads)
#         if prev is not None:
#             atn = (atn + num_layers * prev)/(num_layers + 1)
#         atn = atn.masked_fill(atn_mask.view(b, n_x, n_ctx, 1) == 0, -1e10)
#         softmax_atn = atn.softmax(-2)
#         mha = contract('bnNh,bNdh->bndh', softmax_atn, vs.squeeze()).flatten(-2)
#         return self.w_out(self.dropout(mha)), atn, num_layers + 1
#
#
# def causal_mask_from_pad_mask(pad_mask: Tensor) -> Tensor:
#     # strict 1 by 1 decoding -- no intra type parallelism // assume pad value = 0
#     b, s, t = pad_mask.shape
#     causal_mask = torch.tril(torch.ones(t, t)).view(1, 1, t, 1, t).expand(b, s, t, s, t)
#     return causal_mask.masked_fill(pad_mask.view(b, s, t, 1, 1) == 0, 0)
#
#
# def causal_mask_from_depth(depth_ids: Tensor) -> Tensor:
#     # intra-type parallelism breadh first decoding // assume pad value = -1
#     b, s, t = depth_ids.shape
#     causal_mask = (depth_ids.view(b, s, t, 1, 1).expand(b, s, t, s, t) - depth_ids.view(b, 1, 1, s, t)) > 0
#     return causal_mask.masked_fill(depth_ids.view(b, s, t, 1, 1) == -1, 0)
#
#
# def cross_mask_from_pad_masks(enc_pad_mask: Tensor, dec_pad_mask: Tensor) -> Tensor:
#     return contract('bS,bst->bstS', enc_pad_mask, dec_pad_mask)
#
#
# class TreePE(Module):
#     def __init__(self, max_depth: int, dim: int,  branching_factor: int = 2):
#         if dim % (max_depth * branching_factor) != 0:
#             raise ValueError(f'dim must be divisible by max_depth * branching_factor'
#                              f' but got ({dim}, {max_depth}, {branching_factor})')
#         super(TreePE, self).__init__()
#         self.max_depth = max_depth
#         self.branching_factor = branching_factor
#         self.dim = dim
#         self.ps = Parameter(torch.nn.init.normal_(torch.empty(dim//(max_depth*branching_factor))), requires_grad=True)
#         self.powers = Parameter(torch.arange(0, max_depth).view(max_depth, 1), requires_grad=False)
#
#     def encode_nth_child(self, xs: Tensor, n: int) -> Tensor:
#         xs = xs.roll(self.branching_factor, -1)
#         xs[..., n] = 1
#         return xs
#
#     def encode_left_child(self, xs: Tensor) -> Tensor:
#         return self.encode_nth_child(xs, 0)
#
#     def encode_right_child(self, xs: Tensor) -> Tensor:
#         return self.encode_nth_child(xs, 1)
#
#     def forward(self, xs: Tensor) -> Tensor:
#         *shapes, length = xs.shape
#         depth = length // self.branching_factor
#         weights = self.ps.tanh()
#         norm = torch.sqrt(1 - weights.pow(2))
#         weights = weights.pow(self.powers[:depth]).repeat_interleave(self.branching_factor, -2) * norm
#         return pad(
#             contract('be,ed->bed', xs.flatten(0, -2), weights).view(*shapes, -1),
#             (0, self.dim - length * self.ps.shape[0])) * sqrt(self.dim/2)
#
#
# class RelativePE(Module):
#     def __init__(self, max_distance: int, dim: int):
#         super(RelativePE, self).__init__()
#         self.max_distance = max_distance
#         self.window_size = 2 * max_distance + 1
#         self.dim = dim
#         # todo: shouldnt this be a continuous function?
#         self.weights = Embedding(self.window_size, dim)
#
#     def forward(self, length: int) -> Tensor:
#         distances = self.relative_distance_matrix(length).clip(-self.max_distance, self.max_distance)
#         return self.weights(distances.to(self.weights.weight.data.device) + self.max_distance)
#
#     @staticmethod
#     @torch.no_grad()
#     def relative_distance_matrix(n: int) -> Tensor:
#         return torch.arange(n).unsqueeze(0).expand(n, n) - torch.arange(n).unsqueeze(1).expand(n, n)