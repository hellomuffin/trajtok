import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import List, Tuple
from rotary_embedding_torch import RotaryEmbedding
import math
import torch.nn as nn



from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many


def exists(val):
    return val is not None



def make_grid_positions(L: int, dims: int = 2, device=None):
    if dims == 1:
        t = torch.linspace(0.0, 1.0, L, device=device)
        return t[:, None]  # [L,1]
    H = int(math.ceil(math.sqrt(L)))
    W = int(math.ceil(L / H))
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)[:L]
    return coords  # [L,2]

def fourier_bank(x: torch.Tensor, max_freq: int = 4):
    L, d = x.shape
    freqs = torch.arange(1, max_freq + 1, device=x.device, dtype=x.dtype)  # [F]
    x2pi = 2 * math.pi * x[..., None] * freqs  # [L, d, F]
    s = torch.sin(x2pi)
    c = torch.cos(x2pi)
    bank = torch.cat([s, c], dim=-1).reshape(L, -1)  # [L, 2*d*max_freq]
    bank = F.normalize(bank, dim=-1)
    return bank

def make_orthonormal(rows: int, cols: int, device=None, dtype=torch.float32):
    """
    Returns M in R^{rows x cols}.
    - If rows >= cols: columns are orthonormal (M^T M = I_cols).
    - If rows <  cols: rows are    orthonormal (M M^T = I_rows).
    """
    if rows >= cols:
        A = torch.randn(rows, cols, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode='reduced')   
        return Q
    else:
        A = torch.randn(cols, rows, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode='reduced')   
        return Q.T                                  

def init_latents_fourier(L: int, D: int, grid_dims: int = 2, max_freq: int = 4, device=None, dtype=torch.float32):
    x = make_grid_positions(L, dims=grid_dims, device=device)     
    bank = fourier_bank(x, max_freq=max_freq)                      
    B = bank.shape[-1]

    if D <= B:
        latents = bank[:, :D]                                      
    else:
        W = make_orthonormal(B, D, device=device, dtype=dtype)     
        latents = bank @ W                                        

    latents = latents + 1e-3 * torch.randn_like(latents)
    latents = latents * 0.02
    return latents.to(dtype)


# --------- Small utilities ---------
class WeightedNorm(nn.Module):
    """
    LayerNorm with learnable per-dim weight (and bias).
    Matches the 'WeightedNorm' style normalization seen in many Perceiver-style stacks.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, N, D)
        mu  = x.mean(dim=-1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
        xhat = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight * xhat + self.bias



class PerceiverMLP(nn.Module):
    """
    Gated MLP: SiLU(gate) * up_proj(x) -> down_proj
    ff_multi=4 (Apollo default)
    """
    def __init__(self, dim, ff_multi=4, act=nn.SiLU):
        super().__init__()
        hidden = dim * ff_multi
        self.norm = WeightedNorm(dim)
        self.gate = nn.Linear(dim, hidden, bias=True)
        self.up   = nn.Linear(dim, hidden, bias=True)
        self.act  = act()
        self.down = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        x = self.norm(x)
        gated = self.act(self.gate(x)) * self.up(x)
        return self.down(gated)


# Define a standard transformer block for latents
class LatentTransformerBlock(nn.Module):
    def __init__(self, *, dim, heads=8, dim_head=64, ff_mult=4, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Using batch_first=True to work with shape [batch, seq, dim]
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = PerceiverMLP(dim=dim, ff_multi=ff_mult)
        
    def forward(self, x, attn_mask=None):
        # x: (batch, num_latents, dim)
        # Self-attention with residual connection
        if type(attn_mask) in [list, tuple]: 
            attn_mask = attn_mask[0][0].bool() # the second [0]: eliminate batch dimension. only perceiver attention needs that because different examples have different patches to attend

        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)
        x = x + attn_output
        # Feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        return x



class  PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        use_rotary=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.use_rotary = use_rotary
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(
                dim = int(dim_head * 0.5),
            )

    def forward(self, x, latents, attention_mask=None):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((latents, x), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        if self.use_rotary: 
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
                    
        q = q * self.scale
        
        
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        
        # Apply key padding mask if provided
        if attention_mask is not None:
            if type(attention_mask) in [list, tuple]: 
                attention_mask = torch.cat(attention_mask, dim=-1)   # (B, L, L+N)                                   # (B, L, N)
                sim = sim.masked_fill(attention_mask.unsqueeze(1), float('-inf'))
            else:  
                assert len(attention_mask.shape) == 2
                # attention_mask: (b, key_len)
                # Expand the mask for heads and queries
                query_attn_mask = torch.zeros(latents.shape[:2]).to(attention_mask.device)
                attention_mask = torch.cat([query_attn_mask, attention_mask], dim=1)
                attention_mask = repeat(attention_mask, 'b j -> b h i j', h=sim.shape[1], i=sim.shape[2])
                sim = sim.masked_fill(attention_mask == 1, float('-inf'))


        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 4,
        ff_mult = 2,
        use_rotary = False,
        use_latent_transformer=False,
        update_x=False,
        external_latent=False
    ):
        super().__init__()
        self.use_rotary = use_rotary
        self.use_latent_transformer = use_latent_transformer
        if not external_latent: 
            fourier_emb = init_latents_fourier(num_latents, dim, grid_dims=2, max_freq=int(math.sqrt(num_latents) // 2))
            self.latents = nn.Parameter(fourier_emb)
        assert use_rotary

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads, use_rotary=use_rotary),
                PerceiverMLP(dim),
                LatentTransformerBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult) if use_latent_transformer else nn.Identity()
            ]))

        self.norm_lat = WeightedNorm(dim)
        self.norm_ctx = WeightedNorm(dim)
        
        self.update_x = update_x

    def forward(self, x, attention_mask=None, input_latent=None):

        if input_latent is None: latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])
        else: latents = input_latent
        
        x = self.norm_ctx(x)
        
        for attn, ff, lformer in self.layers:
            latents = self.norm_lat(latents)
            latents = attn(x, latents, attention_mask) + latents
            latents = ff(latents) + latents
            if self.use_latent_transformer: latents = lformer(latents, attention_mask)
        
        if self.update_x: return latents, x
        else: return latents
        


if __name__ == '__main__':
    def count_all_parameters(model):
        return sum(p.numel() for p in model.parameters())

    perceiver = PerceiverResampler(dim=1024, depth=1, num_latents=1, use_rotary=True)
    
    print(count_all_parameters(perceiver))