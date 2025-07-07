from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn import GCN


class LayerNorm(nn.LayerNorm):
    """Copied from CLIP. Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class QuickGELU(nn.Module):
    """Copied from CLIP"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    """Copied from CLIP"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Copied from CLIP"""
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class MedTok(nn.Module):
    """Parts adapted from CLIP"""
    def __init__(self,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # graph
                 num_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_ratio: int,
                 activation: int,
                 norm_type: int,
                 codebook_size: int,
                 lamb_edge: int,
                 lamb_node: int,
                 # common
                 dim_proj: int,
                 dim_codebook: int,
                 K: int
                 ):
        super().__init__()
        self.context_length = context_length
        self.dim_proj = dim_proj
        self.K = K

        # text encoder; should be frozen
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.ln_final = LayerNorm(transformer_width)
        
        # graph encoder
        self.graph_encoder = GCN(num_layers,
                                 input_dim,
                                 hidden_dim,
                                 output_dim,
                                 dropout_ratio,
                                 activation,
                                 norm_type,
                                 codebook_size,
                                 lamb_edge,
                                 lamb_node
        )
        self.graph_ln_final = LayerNorm(output_dim)

        # linear projectors
        self.txt_lin_proj = nn.Linear(transformer_width, self.dim_proj)
        self.graph_lin_proj = nn.Linear(output_dim, self.dim_proj)

        # cross-attention matrices
        self.w_q = nn.Linear(dim_proj, self.dim_proj)
        self.w_k = nn.Linear(dim_proj, self.dim_proj)
        self.w_v = nn.Linear(dim_proj, self.dim_proj)

        # codebook initialization
        # codebook split sizes can be tuned as a hyperparameter
        self.dim_codebook = dim_codebook
        text_split = graph_split = dim_codebook // 3
        dim_shared = dim_codebook - text_split - graph_split

        self.codebook_text = nn.Parameter(torch.randn(text_split, dim_proj))
        self.codebook_graph = nn.Parameter(torch.randn(graph_split, dim_proj))
        self.codebook_shared = nn.Parameter(torch.randn(dim_shared, dim_proj))

        self.token_to_id = {f"tok_{i}": i for i in range(self.dim_codebook)}
        self.id_to_token = {i: f"tok_{i}" for i in range(self.dim_codebook)}

    def get_codebook(self):
        return torch.cat([self.codebook_text, self.codebook_graph, self.codebook_shared], dim=0)
    
    def vector_quantize(self, embeds, codebook, K=5):
        # euclidean distance
        dist = torch.cdist(embeds.unsqueeze(1), codebook.unsqueeze(0), p=2).squeeze(1)

        topk_dist, topk_indices = torch.topk(-dist, K, dim=1)
        #topk_dist = -topk_dist
        #weights = F.softmax(-topk_dist, dim=1)
        weights = F.softmax(topk_dist, dim=1)
        topk_vectors = codebook[topk_indices]

        quantized_vecs = torch.sum(weights.unsqueeze(-1) * topk_vectors, dim=1)
        return quantized_vecs, topk_indices
    
    def build_attention_mask(self):
        """Copied from CLIP"""
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def cross_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        w_attn = F.softmax(scores, dim=-1)

        output = torch.matmul(w_attn, v)
        return output, w_attn
    
    def encode_text(self, text):
        """Copied from CLIP"""
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, graph, text):
        # get raw embeds
        text_embeds = self.encode_text(text)
        text_embeds = self.ln_final(text_embeds) # x_t
        graph_embeds = self.graph_encoder(graph) 
        graph_embeds = self.graph_ln_final(graph_embeds) # x_g

        # get specific embeds
        text_spec_embeds = self.txt_lin_proj(text_embeds) # e^s_t
        graph_spec_embeds = self.graph_lin_proj(graph_embeds) # e^s_g

        # get cross-modality embeds
        # equation from paper uses raw embeds notation, may be shorthand only but mean modality-specific
        q_t = self.w_q(text_spec_embeds)
        k_t = self.w_k(graph_spec_embeds)
        v_t = self.w_v(graph_spec_embeds)
        text_cross_embeds = self.cross_attention(q=q_t, k=k_t, v=v_t) # e^c_t

        q_g = self.w_q(graph_spec_embeds)
        k_g = self.w_k(text_spec_embeds)
        v_g = self.w_v(text_spec_embeds)
        graph_cross_embeds = self.cross_attention(q=q_g, k=k_g, v=v_g) # e^c_g

        # tokenization; building codebook
        codebook_text = self.codebook_text
        codebook_graph = self.codebook_graph
        codebook_shared = self.codebook_shared

        q_text = torch.cat([codebook_text, codebook_shared], dim=0)
        q_graph = torch.cat([codebook_graph, codebook_shared], dim=0)
        q_e_s_t, idx_e_s_t = self.vector_quantize(text_spec_embeds, q_text, K=self.K)
        q_e_s_g, idx_e_s_g = self.vector_quantize(graph_spec_embeds, q_graph, K=self.K)

        q_e_c_t, idx_e_c_t = self.vector_quantize(text_cross_embeds, q_text, K=self.K)
        q_e_c_g, idx_e_c_g = self.vector_quantize(graph_cross_embeds, q_graph, K=self.K)

        return {
            "text_spec_embeds": text_spec_embeds,
            "graph_spec_embeds": graph_spec_embeds,
            "text_cross_embeds": text_cross_embeds,
            "graph_cross_embeds": graph_cross_embeds,
            "q_e_s_t": q_e_s_t,
            "q_e_s_g": q_e_s_g,
            "q_e_c_t": q_e_c_t,
            "q_e_c_g": q_e_c_g,
            "idx_e_s_t": idx_e_s_t,
            "idx_e_s_g": idx_e_s_g,
            "idx_e_c_t": idx_e_c_t,
            "idx_e_c_g": idx_e_c_g,
        }
    
    def get_vocab(self):
        return self.token_to_id
    
    @torch.no_grad()
    # def _get_fused_embedding(self, text, graph):
    #     self.eval()

    #     text_embeds = self.ln_final(self.encode_text(text))
    #     graph_embeds = self.graph_ln_final(self.graph_encoder(graph))

    #     text_spec_embeds = self.txt_lin_proj(text_embeds)
    #     graph_spec_embeds = self.graph_lin_proj(graph_embeds)

    #     q_t = self.w_q(text_spec_embeds)
    #     k_t = self.w_k(graph_spec_embeds)
    #     v_t = self.w_v(graph_spec_embeds)
    #     text_cross_embeds, _ = self.cross_attention(q=q_t, k=k_t, v=v_t)

    #     q_g = self.w_q(graph_spec_embeds)
    #     k_g = self.w_k(text_spec_embeds)
    #     v_g = self.w_v(text_spec_embeds)
    #     graph_cross_embeds, _ = self.cross_attention(q=q_g, k=k_g, v=v_g)

    #     ### fuse embeddings
    #     fused_embed = F.normalize(text_cross_embeds, dim=-1) + F.normalize(graph_cross_embeds, dim=-1)
    #     fused_embed = F.normalize(fused_embed, dim=-1)

    #     return fused_embed

    @torch.no_grad()
    def encode(self, text):
        # self.eval()
        # fused_embed = self._get_fused_embedding(text, graph)
        # codebook = self.get_codebook()
        # dist = torch.cdist(fused_embed.unsqueeze(1), codebook.unsqueeze(0), p=2).squeeze(1)
        # top_index = torch.topk(-dist, k=1, dim=-1).indices

        # return top_index
        self.eval()
        text_embeds = self.encode_text(text)
        text_spec_embeds = self.txt_lin_proj(text_embeds)
        fused_embed = F.normalize(text_spec_embeds, dim=-1)
        codebook = self.get_codebook()
        dist = torch.cdist(fused_embed.unsqueeze(1), codebook.unsqueeze(0), p=2).squeeze(1)
        top_index = torch.topk(-dist, k=1, dim=-1).indices
        return top_index
    
    @torch.no_grad()
    def decode(self, token_id):
        codebook = self.get_codebook()
        if isinstance(token_id, list):
            token_id = torch.tensor(token_id)
        return codebook[token_id]