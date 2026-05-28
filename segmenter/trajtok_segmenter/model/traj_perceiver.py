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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_freqs(dim_pairs: int, base: float = 10000.0, dtype=torch.float32, device=None):
    """
    Return inverse frequencies (size [dim_pairs]) used for sin/cos.
    dim_pairs counts pairs (so rotary_dim_axis = 2*dim_pairs).
    """
    # Standard RoPE frequencies: base^( -2i/d )
    # Here dim_pairs is the number of 2D pairs for one axis chunk.
    inv_freq = base ** (-torch.arange(0, dim_pairs, dtype=dtype, device=device) / dim_pairs)
    return inv_freq  # [dim_pairs]


def _axis_cos_sin(idx_axis: torch.Tensor, inv_freq: torch.Tensor):
    """
    idx_axis: [L] integer indices for one axis (t, h, or w)
    inv_freq: [dim_pairs] inverse frequencies
    Returns:
      cos: [L, dim_pairs]
      sin: [L, dim_pairs]
    """
    # angles: outer product [L, dim_pairs]
    angles = idx_axis[:, None].to(inv_freq.dtype) * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def _apply_rope_to_chunk(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x:   [B, H, L, D_axis]  (axis chunk of the head_dim)
    cos: [L, D_axis//2]
    sin: [L, D_axis//2]
    Returns rotated x with same shape, using pairwise rotation (even, odd).
    """
    B, H, L, D = x.shape
    assert D % 2 == 0, "Axis chunk must be even: pairs of (even, odd)."
    # Reshape to pairs: [..., D//2, 2]
    x = x.view(B, H, L, D // 2, 2)
    x_even = x[..., 0]  # [B,H,L,D//2]
    x_odd  = x[..., 1]

    # Match cos/sin to [B,H,L,D//2]
    cos = cos[None, None, ...]           # [1,1,L,D//2]
    sin = sin[None, None, ...]           # [1,1,L,D//2]

    # Rotation:
    # [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_odd*cos + x_even*sin]
    y_even = x_even * cos - x_odd * sin
    y_odd  = x_odd  * cos + x_even * sin

    y = torch.stack([y_even, y_odd], dim=-1).reshape(B, H, L, D)
    return y


def compute_thw_indices(T: int, H: int, W: int, device=None):
    """
    Produce per-token (t, h, w) indices for a flattened THW grid.
    Assumes flatten order L = T*H*W with time-major or any consistent order:
      Here we use: t-major, then h, then w:
        l = t*(H*W) + h*W + w
    Returns:
      t_idx, h_idx, w_idx each [L] on device.
    """
    # [T, H, W] grids
    t = torch.arange(T, device=device)
    h = torch.arange(H, device=device)
    w = torch.arange(W, device=device)
    tt, hh, ww = torch.meshgrid(t, h, w, indexing='ij')
    L = T * H * W
    t_idx = tt.reshape(L)
    h_idx = hh.reshape(L)
    w_idx = ww.reshape(L)
    return t_idx, h_idx, w_idx


class RotaryEmbedding3D(nn.Module):
    """
    3D Rotary Positional Embedding for video tokens on a (T,H,W) grid.

    Usage:
      rope = RotaryEmbedding3D(head_dim, rotary_dim=None, base=10000.0)
      q, k = rope(q, k, T, H, W)
    where q,k are [B, n_heads, L, head_dim] and L = T*H*W.

    Notes:
      - rotary_dim defaults to the largest multiple of 6 <= head_dim
        (pairs per axis -> need divisible by 2, and 3 axes -> *3 => divisible by 6).
      - The rotary_dim is split equally across (time, height, width).
      - Remaining channels (head_dim - rotary_dim) are passed through unchanged.
    """

    def __init__(self, head_dim: int, rotary_dim: int = None, base: float = 10000.0):
        super().__init__()
        if rotary_dim is None:
            rotary_dim = (head_dim // 6) * 6  # largest multiple of 6
        assert 0 <= rotary_dim <= head_dim
        assert rotary_dim % 6 == 0, "rotary_dim must be divisible by 6 (pairs per axis, 3 axes)."

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.per_axis_dim = rotary_dim // 3          # chunk per axis
        self.per_axis_pairs = self.per_axis_dim // 2 # pair count per axis
        self.base = float(base)

        # Registered buffers for inv_freq per axis (identical shapes; can be different if desired)
        self.register_buffer(
            'inv_freq_axis',
            _build_freqs(self.per_axis_pairs, base=self.base, dtype=torch.float32, device=None),
            persistent=False
        )

    def forward(self, q: torch.Tensor,  T: int, H: int, W: int):
        """
        q, k: [B, n_heads, L, head_dim], L=T*H*W
        Returns rotated (q, k) with RoPE applied to the first rotary_dim channels.
        """
        B, nH, L, D = q.shape
        assert D == self.head_dim, "Head dim mismatch"
        assert L == T * H * W, "Token length must equal T*H*W"

        if self.rotary_dim == 0:
            return q

        device = q.device
        dtype  = q.dtype

        # Split into rotary and pass-through
        q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        
        # Partition rotary channels into (time | height | width) chunks
        D_axis = self.per_axis_dim
        q_t, q_h, q_w = torch.split(q_rot, (D_axis, D_axis, D_axis), dim=-1)
        
        # Get per-token indices for each axis
        t_idx, h_idx, w_idx = compute_thw_indices(T, H, W, device=device)  # each [L]

        # Precompute axis cos/sin: [L, D_axis//2]
        inv_freq = self.inv_freq_axis.to(dtype=dtype, device=device)
        cos_t, sin_t = _axis_cos_sin(t_idx, inv_freq)
        cos_h, sin_h = _axis_cos_sin(h_idx, inv_freq)
        cos_w, sin_w = _axis_cos_sin(w_idx, inv_freq)

        # Apply RoPE per axis chunk
        q_t = _apply_rope_to_chunk(q_t, cos_t, sin_t)
        q_h = _apply_rope_to_chunk(q_h, cos_h, sin_h)
        q_w = _apply_rope_to_chunk(q_w, cos_w, sin_w)

        # Recombine
        q_rotated = torch.cat([q_t, q_h, q_w], dim=-1)
        
        q_out = torch.cat([q_rotated, q_pass], dim=-1)
        return q_out


    def make_pure_positional_Klin(self, B, T, H, W, n_heads,  device, dtype):
        """
        Returns K_lin of shape [B, L, n_heads*head_dim] (L = T*H*W).
        After reshape to [B, n_heads, L, head_dim], the first rotary_dim
        channels in each head are pairs set to [1,0] so RoPE turns them
        into [cos, sin] per position. Pass-through channels are 0.
        """
        assert self.rotary_dim % 2 == 0
        assert self.rotary_dim <= self.head_dim

        L = T * H * W
        K_lin = torch.zeros(B, L, n_heads * self.head_dim, device=device, dtype=dtype)

        per_axis = self.rotary_dim // 3       # time, height, width chunks
        assert per_axis % 2 == 0, "rotary_dim must be divisible by 6 for 3D RoPE."

        # indices within a head for rotary part
        rot_start, rot_end = 0, self.rotary_dim
        # we set every even index in [0, rotary_dim) to 1 (odd stays 0),
        # i.e., each (even, odd) pair is [1,0].
        even_idx = torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.long)

        for h in range(n_heads):
            head_start = h * self.head_dim
            # even positions inside head
            pos = head_start + rot_start + even_idx
            # broadcast over batch and tokens
            K_lin[:, :, pos] = 1.0

        return K_lin


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

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

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

    def reset_parameters(self):
        self.norm.reset_parameters()
        # Xavier uniform initialization for linear layers
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)

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
        
    def reset_parameters(self):
        # Reset LayerNorm parameters
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)
        
        # Reset MultiheadAttention parameters
        self.self_attn._reset_parameters()
        
        # Reset MLP parameters
        self.ff.reset_parameters()
        
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
        self.dim_head = dim_head
        inner_dim = dim_head * heads


        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.use_rotary = use_rotary
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding3D(head_dim=dim_head, rotary_dim=int(dim_head // 12) * 6, base=10000.0)

    def reset_parameters(self):
        
        # Reset linear layer parameters with proper scaling for attention
        nn.init.xavier_uniform_(self.to_q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.to_kv.weight, gain=1.0)
        nn.init.xavier_uniform_(self.to_out.weight, gain=1.0)

    def forward(self, x, latents, T,H,W, attention_mask=None):
        """
        x shape (b, thw, d), where 1d sequence thw is flattened from video patches (t, h, w)
        """
        if x is None: x = self.rotary_emb.make_pure_positional_Klin(latents.shape[0],  T, H, W, self.heads, device=latents.device, dtype=latents.dtype)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        k, v = self.to_kv(x).chunk(2, dim = -1)


        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        if self.use_rotary: 
            k = self.rotary_emb(k,  T=T, H=H, W=W)
                    
        q = q * self.scale
        
        
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        
        # Apply key padding mask if provided
        if attention_mask is not None:
            if type(attention_mask) in [list, tuple]:                                   # (B, L, N)
                sim = sim.masked_fill(attention_mask[-1].unsqueeze(1), float('-inf'))
            else:  
                assert len(attention_mask.shape) == 2
                # attention_mask: (b, key_len)
                # Expand the mask for heads and queries
                attention_mask = repeat(attention_mask, 'b j -> b h i j', h=sim.shape[1], i=sim.shape[2])
                sim = sim.masked_fill(attention_mask == 1, float('-inf'))


        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class TrajPerceiver(nn.Module):
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

    def reset_parameters(self):
        # Reset latent parameters if not external
        if hasattr(self, 'latents'):
            # Reinitialize latents using the same fourier initialization
            num_latents, dim = self.latents.shape
            fourier_emb = init_latents_fourier(num_latents, dim, grid_dims=2, max_freq=int(math.sqrt(num_latents) // 2))
            self.latents.data.copy_(fourier_emb)
        
        # Reset normalization layers
        self.norm_lat.reset_parameters()
        self.norm_ctx.reset_parameters()
        
        # Reset all layers
        for layer_group in self.layers:
            attn, mlp, lformer = layer_group
            attn.reset_parameters()
            mlp.reset_parameters()
            if hasattr(lformer, 'reset_parameters'):
                lformer.reset_parameters()

    def forward(self, x, T,H,W, attention_mask=None, input_latent=None):
        if input_latent is None: latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])
        else: latents = input_latent

        if x is not None: x = self.norm_ctx(x)
        
        for attn, ff, lformer in self.layers:
            latents = self.norm_lat(latents)
            latents = attn(x, latents, T,H,W, attention_mask) + latents
            latents = ff(latents) + latents
            if self.use_latent_transformer: latents = lformer(latents, attention_mask)
        
        if self.update_x: return latents, x
        else: return latents
        


if __name__ == '__main__':
    def count_all_parameters(model):
        return sum(p.numel() for p in model.parameters())

    perceiver = TrajPerceiver(dim=1024, depth=1, num_latents=1, use_rotary=True)
    
    print(count_all_parameters(perceiver))