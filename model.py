"""LAM-NET model implementation.

The historical class name LAA_Mamba_SpineSeg denotes LAM-NET, and LAA
implements Adaptive Attention Skip Fusion (AASF). This file contains the
network definition and its Mamba/S4D dependencies. Data preparation,
training, evaluation and trained weights are not included.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES_25 = 26
NUM_CLASSES_4  = 5
NUM_CLASSES_2  = 2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mamba block (real mamba_ssm or pure-PyTorch S4D fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_mamba():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        return None


class _S4DKernel(nn.Module):
    """
    Diagonal-SSM (S4D) as fallback when mamba_ssm is unavailable.
    Implements a 1-D causal convolution-based SSM with learnable
    complex diagonal A matrix.  Complexity: O(L log L) via FFT.
    """
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Diagonal A in log-space: shape (d_model, d_state)
        self.A_log = nn.Parameter(
            torch.randn(d_model, d_state) * 0.5
        )
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """x: (B, L, d_model) → y: (B, L, d_model)"""
        B, L, d = x.shape
        A = -torch.exp(self.A_log)                     # (d, d_state)
        # Build convolution kernel via ZOH discretization
        t = torch.arange(L, device=x.device).float()   # (L,)
        # dA shape: (L, d, d_state) via broadcasting
        dA = torch.exp(A.unsqueeze(0) * t[:, None, None])  # (L, d, d_state)
        # K(l) = C · diag(dA[l]) · B  (d_model,)
        K = (dA * self.B.unsqueeze(0) * self.C.unsqueeze(0)).sum(-1)  # (L, d)
        K = K.permute(1, 0)  # (d, L)

        # Causal conv via FFT
        x_t = x.permute(0, 2, 1)  # (B, d, L)
        L2 = 2 * L
        X = torch.fft.rfft(x_t, n=L2)
        K_padded = F.pad(K, (0, L))
        Kf = torch.fft.rfft(K_padded, n=L2)
        y = torch.fft.irfft(X * Kf.unsqueeze(0), n=L2)[..., :L]
        y = y + x_t * self.D.unsqueeze(0).unsqueeze(-1)
        return y.permute(0, 2, 1)  # (B, L, d)


class _S4DBlock(nn.Module):
    """Residual S4D block (fallback for mamba_ssm)."""
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm  = _S4DKernel(d_model, d_state)
        self.out  = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (B, L, d_model)"""
        r = x
        x = self.norm(x)
        x = self.ssm(x)
        x = F.silu(x)
        x = self.out(x)
        return x + r


class MambaBlock2D(nn.Module):
    """
    Applies Mamba/SSM to a 2D feature map.

    Strategy:
      1. B×C×H×W  → flatten spatial → B×(H*W)×C
      2. MambaSSM (or S4D fallback)
      3. Reshape → B×C×H×W
      4. Residual + LayerNorm on channel dim
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 force_s4d: bool = False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        Mamba = None if force_s4d else _try_import_mamba()
        if Mamba is not None:
            # Real mamba-ssm
            self.ssm = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._use_mamba = True
        else:
            # Fallback: pure-PyTorch S4D diagonal SSM
            tag = "force_s4d=True" if force_s4d else "mamba_ssm not available"
            print(f"[LAA-Mamba] {tag}, using S4D fallback.", flush=True)
            self.ssm = _S4DBlock(d_model=dim, d_state=max(d_state * expand, 64))
            self._use_mamba = False

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        # 2D → sequence
        x_seq = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        x_seq = self.norm(x_seq)
        y_seq = self.ssm(x_seq)                         # (B, H*W, C)
        y_seq = self.out_norm(y_seq)
        # Residual + reshape back
        y = (x_seq + y_seq).permute(0, 2, 1).reshape(B, C, H, W)
        return y + x  # residual connection to input


class MambaBlock1D(nn.Module):
    """
    1-D Mamba SSM for a K-length sequence of feature tokens (B, K, D).
    Used by the inter-slice sequential Mamba bottleneck.
    """
    def __init__(self, dim: int, d_state: int = 8, d_conv: int = 4,
                 expand: int = 2, force_s4d: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        Mamba = None if force_s4d else _try_import_mamba()
        if Mamba is not None:
            self.ssm = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self._use_mamba = True
        else:
            tag = "force_s4d=True" if force_s4d else "mamba_ssm not available"
            print(f"[MambaBlock1D] {tag}, using S4D fallback.", flush=True)
            self.ssm = _S4DBlock(d_model=dim, d_state=max(d_state * expand, 32))
            self._use_mamba = False
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, D) — K-length token sequence."""
        residual = x
        y = self.ssm(self.norm(x))   # (B, K, D)
        return residual + self.out_norm(y)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LAA – Layer-wise Adaptive Attention for skip fusion
# ─────────────────────────────────────────────────────────────────────────────

class LAA(nn.Module):
    """
    Layer-wise Adaptive Attention for encoder-decoder skip fusion.

    Given:
      skip  : encoder feature  (B, C_enc, H, W)
      query : decoder feature  (B, C_dec, H, W)  [after upsampling]

    The module:
      1. Projects query → attention gate (sigmoid)
      2. Element-wise gates the skip feature
      3. Projects concatenation [gated_skip, query] → C_out channels
      4. Residual: output + query (projected if needed)
    """
    def __init__(self, c_enc: int, c_dec: int, c_out: int, reduction: int = 4):
        super().__init__()
        c_mid = max(c_enc // reduction, 16)

        # Gate: query → spatial attention map
        self.gate_q = nn.Conv2d(c_dec, c_mid, 1)
        self.gate_s = nn.Conv2d(c_enc, c_mid, 1)
        self.gate_out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, 1, 1),
            nn.Sigmoid(),
        )

        # Fusion: (gated_skip ⊕ query) → c_out
        self.fuse = nn.Sequential(
            nn.Conv2d(c_enc + c_dec, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        # Residual projection (query → c_out if dims differ)
        self.res_proj = (
            nn.Conv2d(c_dec, c_out, 1)
            if c_dec != c_out
            else nn.Identity()
        )

    def forward(self, skip: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        skip : (B, C_enc, H, W)
        query: (B, C_dec, H, W)  — already upsampled to same spatial size
        """
        # Adaptive gate
        g = self.gate_q(query) + self.gate_s(skip)
        attn = self.gate_out(g)          # (B, 1, H, W)
        gated_skip = skip * attn

        # Fuse
        fused = self.fuse(torch.cat([gated_skip, query], dim=1))
        return fused + self.res_proj(query)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Encoder / Decoder blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LAA-Mamba SpineSeg  (main model)
# ─────────────────────────────────────────────────────────────────────────────

class LAA_Mamba_SpineSeg(nn.Module):
    """
    LAA-Mamba SpineSeg

    Encoder  : 4 ConvBlock2D stages (base → base*16) + MaxPool
    Bottleneck: ConvBlock2D + MambaBlock2D (real SSM on flattened tokens)
    Decoder  : 4 stages, each: ConvTranspose2d → LAA skip fusion → ConvBlock2D
    Heads    : head_25 (26-class), head_4 (5-class), head_2 (2-class)

    LAA positions  (decoder stage → encoder level fused):
      decoder stage 4 ← encoder e4  (c_enc=base*8,  c_dec=base*8)
      decoder stage 3 ← encoder e3  (c_enc=base*4,  c_dec=base*4)
      decoder stage 2 ← encoder e2  (c_enc=base*2,  c_dec=base*2)
      decoder stage 1 ← encoder e1  (c_enc=base,    c_dec=base)
    """

    def __init__(
        self,
        base: int = 48,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        force_s4d: bool = False,
        use_zpos: bool = False,
        use_residual_mamba: bool = True,
        use_3slice_input: bool = False,
        use_interslice_mamba: bool = False,
        interslice_K: int = 5,
    ):
        super().__init__()
        self.use_zpos = use_zpos
        self.use_residual_mamba = use_residual_mamba
        self.use_3slice_input = use_3slice_input
        self.use_interslice_mamba = use_interslice_mamba
        self.interslice_K = interslice_K
        b = base
        self.pool = nn.MaxPool2d(2)

        # ── Encoder ──────────────────────────────────────────────────────────
        if use_interslice_mamba:
            in_ch = (3 if use_3slice_input else 1) + 1  # local (3 or 1 CT slices) + z-pos per position
        else:
            in_ch = (3 if use_3slice_input else 1) + (1 if use_zpos else 0)
        self.e1 = ConvBlock2D(in_ch, b)      # → (B, b,    H, W)
        self.e2 = ConvBlock2D(b,    b*2)     # → (B, b*2,  H/2, W/2)
        self.e3 = ConvBlock2D(b*2,  b*4)     # → (B, b*4,  H/4, W/4)
        self.e4 = ConvBlock2D(b*4,  b*8)     # → (B, b*8,  H/8, W/8)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bn_conv  = ConvBlock2D(b*8, b*16)
        if use_residual_mamba:
            self.mamba_bn = MambaBlock2D(
                dim=b*16,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                force_s4d=force_s4d,
            )
            # Learnable blend: x_out = x_conv + mamba_alpha * MambaBlock2D(x_conv)
            # Initialised small (0.1) so training starts conv-dominated
            self.mamba_alpha = nn.Parameter(torch.tensor(0.1))
        if use_interslice_mamba:
            self.interslice_mamba = MambaBlock1D(
                dim=b * 16, d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand, force_s4d=force_s4d,
            )
            # FiLM: context token → (gamma, beta) for bottleneck modulation
            # Zero-initialize FiLM to an identity modulation
            self.film_linear = nn.Linear(b * 16, 2 * b * 16)
            nn.init.zeros_(self.film_linear.weight)
            nn.init.zeros_(self.film_linear.bias)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Stage 4: b*16 → b*8
        self.u4  = nn.ConvTranspose2d(b*16, b*8, 2, 2)
        self.laa4 = LAA(c_enc=b*8, c_dec=b*8, c_out=b*8)
        self.d4  = ConvBlock2D(b*8, b*8)

        # Stage 3: b*8 → b*4
        self.u3  = nn.ConvTranspose2d(b*8, b*4, 2, 2)
        self.laa3 = LAA(c_enc=b*4, c_dec=b*4, c_out=b*4)
        self.d3  = ConvBlock2D(b*4, b*4)

        # Stage 2: b*4 → b*2
        self.u2  = nn.ConvTranspose2d(b*4, b*2, 2, 2)
        self.laa2 = LAA(c_enc=b*2, c_dec=b*2, c_out=b*2)
        self.d2  = ConvBlock2D(b*2, b*2)

        # Stage 1: b*2 → b
        self.u1  = nn.ConvTranspose2d(b*2, b, 2, 2)
        self.laa1 = LAA(c_enc=b, c_dec=b, c_out=b)
        self.d1  = ConvBlock2D(b, b)

        # ── Output heads ─────────────────────────────────────────────────────
        self.head_25 = nn.Conv2d(b, NUM_CLASSES_25, 1)
        self.head_4  = nn.Conv2d(b, NUM_CLASSES_4,  1)
        self.head_2  = nn.Conv2d(b, NUM_CLASSES_2,  1)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Single-position path: x is (B, C, H, W).
        Inter-slice path: x is (B, K, C, H, W); C follows the input flags.
        Returns dict with keys: logits_25, logits_4, logits_2
        """
        if self.use_interslice_mamba:
            # ── Inter-slice sequential Mamba path ────────────────────
            B, K, C, H, W = x.shape
            xf = x.reshape(B * K, C, H, W)              # (B*K, C, H, W)
            # Shared encoder over all K slices
            e1f = self.e1(xf)                            # (B*K, b, H, W)
            e2f = self.e2(self.pool(e1f))                # (B*K, b*2, H/2, W/2)
            e3f = self.e3(self.pool(e2f))                # (B*K, b*4, H/4, W/4)
            e4f = self.e4(self.pool(e3f))                # (B*K, b*8, H/8, W/8)
            bnf = self.bn_conv(self.pool(e4f))           # (B*K, b*16, H/16, W/16)
            # GAP → K tokens → 1-D Mamba → center context
            C_bn, Hp, Wp = bnf.shape[1], bnf.shape[2], bnf.shape[3]
            tokens = bnf.mean(dim=[-2, -1]).view(B, K, C_bn)   # (B, K, C_bn)
            ctx    = self.interslice_mamba(tokens)              # (B, K, C_bn)
            ctx_c  = ctx[:, K // 2]                             # (B, C_bn)
            # Extract center-slice features for decoder
            c = K // 2
            bnf_c = bnf.view(B, K, C_bn, Hp, Wp)[:, c].contiguous()
            e4 = e4f.view(B, K, *e4f.shape[1:])[:, c].contiguous()
            e3 = e3f.view(B, K, *e3f.shape[1:])[:, c].contiguous()
            e2 = e2f.view(B, K, *e2f.shape[1:])[:, c].contiguous()
            e1 = e1f.view(B, K, *e1f.shape[1:])[:, c].contiguous()
            del e1f, e2f, e3f, e4f, bnf
            # FiLM: modulate center bottleneck with Mamba context
            gamma, beta = self.film_linear(ctx_c).chunk(2, dim=1)  # each (B, C_bn)
            bn = bnf_c * (1 + gamma.view(B, C_bn, 1, 1)) + beta.view(B, C_bn, 1, 1)
        else:
            # ── Single-position encoder path ─────────────────────────
            e1 = self.e1(x)                    # (B, b,    H,   W  )
            e2 = self.e2(self.pool(e1))        # (B, b*2,  H/2, W/2)
            e3 = self.e3(self.pool(e2))        # (B, b*4,  H/4, W/4)
            e4 = self.e4(self.pool(e3))        # (B, b*8,  H/8, W/8)
            bn_conv = self.bn_conv(self.pool(e4))
            if self.use_residual_mamba:
                bn = bn_conv + self.mamba_alpha * self.mamba_bn(bn_conv)
            else:
                bn = bn_conv

        # ── Shared decoder path ───────────────────────────────────
        # Decoder with LAA skip fusion
        up4 = self.u4(bn)                  # (B, b*8,  H/8, W/8)
        f4  = self.laa4(e4, up4)           # LAA: e4 guided by up4
        d4  = self.d4(f4)

        up3 = self.u3(d4)                  # (B, b*4,  H/4, W/4)
        f3  = self.laa3(e3, up3)
        d3  = self.d3(f3)

        up2 = self.u2(d3)                  # (B, b*2,  H/2, W/2)
        f2  = self.laa2(e2, up2)
        d2  = self.d2(f2)

        up1 = self.u1(d2)                  # (B, b,    H,   W  )
        f1  = self.laa1(e1, up1)
        d1  = self.d1(f1)

        return {
            'logits_25': self.head_25(d1),
            'logits_4':  self.head_4(d1),
            'logits_2':  self.head_2(d1),
        }

    @staticmethod
    def count_params(model=None):
        m = model or LAA_Mamba_SpineSeg()
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    @property
    def mamba_backend(self) -> str:
        """Return which SSM backend is in use."""
        return 'mamba_ssm' if self.mamba_bn._use_mamba else 'S4D-fallback'
