"""LAM-NET public core: frozen model, label hierarchy, training and physical evaluation.
Set LAMNET_DATA_ROOT to your CTSpine1K data root. See README.md.
"""
"""
LAA-Mamba SpineSeg: Layer-wise Adaptive Attention + real Mamba-SSM
for 2D vertebra segmentation with multi-head hierarchical supervision.

Architecture overview
─────────────────────
Input 2D CT slice  [B, 1, H, W]
  │
  ├─ Encoder (4 ConvBlock2D stages with MaxPool)
  │    └─ After each encoder stage: save skip feature
  │
  ├─ Bottleneck MambaBlock2D
  │    Input: B×C×H×W  →  flatten  →  B×(H*W)×C
  │    Real mamba_ssm.Mamba selective scan (or pure-PyTorch fallback)
  │    Reshape back  →  B×C×H×W
  │
  ├─ Decoder (4 stages, each with LAA skip fusion)
  │    LAA (Layer-wise Adaptive Attention):
  │      - Gating: decoder feature attends to encoder skip via cross-attention
  │      - Output: weighted skip + decoder upsample → fused feature
  │    Followed by ConvBlock2D refine
  │
  └─ Three output heads
       head_25 : 26 classes  (BG + C1-L6)
       head_4  :  5 classes  (BG + coarse anatomy)
       head_2  :  2 classes  (BG + spine)

Loss:  L = L_25 + 0.3·L_2 + 0.3·L_4

Mamba dependency
─────────────────
Requires:  mamba-ssm >= 2.0   (tested with 2.2.2)
           causal-conv1d >= 1.2
If unavailable, a pure-PyTorch SSM (S4D diagonal) fallback is used
automatically.
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
    Used by the inter-slice sequential Mamba bottleneck in V6.
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
            # Zero-init so V6 starts from same baseline as V5 (identity FiLM)
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
        V4/V5 : x is (B, C, H, W) — 1/2/4 channels
        V6    : x is (B, K, 2, H, W) — K neighboring slices × [CT, z-pos]
        Returns dict with keys: logits_25, logits_4, logits_2
        """
        if self.use_interslice_mamba:
            # ── Inter-slice Sequential Mamba path (V6) ────────────────────
            B, K, C, H, W = x.shape
            xf = x.reshape(B * K, C, H, W)              # (B*K, 2, H, W)
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
            # ── Original encoder path (V4 / V5) ─────────────────────────
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


import numpy as np

def label_25_to_4class(mask_25: np.ndarray) -> np.ndarray:
    mask_4 = np.zeros_like(mask_25, dtype=np.int64)
    mask_4[(mask_25 >= 1) & (mask_25 <= 7)] = 1
    mask_4[(mask_25 >= 8) & (mask_25 <= 13)] = 2
    mask_4[(mask_25 >= 14) & (mask_25 <= 19)] = 3
    mask_4[(mask_25 >= 20) & (mask_25 <= 25)] = 4
    return mask_4

def label_25_to_2class(mask_25: np.ndarray) -> np.ndarray:
    return (mask_25 > 0).astype(np.int64)


PAPER_SPLIT = {'split_seed': 42, 'train_volumes': ['HNSCC::HN_P032.nii.gz', 'HNSCC::HN_P017.nii.gz', 'HNSCC::HN_P027.nii.gz', 'HNSCC::HN_P019.nii.gz', 'HNSCC::HN_P010.nii.gz', 'HNSCC::HN_P011.nii.gz', 'HNSCC::HN_P034.nii.gz', 'HNSCC::HN_P028.nii.gz', 'HNSCC::HN_P014.nii.gz', 'HNSCC::HN_P001.nii.gz', 'HNSCC::HN_P005.nii.gz', 'HNSCC::HN_P018.nii.gz', 'HNSCC::HN_P007.nii.gz', 'HNSCC::HN_P015.nii.gz', 'HNSCC::HN_P013.nii.gz', 'HNSCC::HN_P026.nii.gz', 'HNSCC::HN_P002.nii.gz', 'HNSCC::HN_P003.nii.gz', 'HNSCC::HN_P029.nii.gz', 'HNSCC::HN_P004.nii.gz', 'HNSCC::HN_P024.nii.gz', 'HNSCC::HN_P030.nii.gz', 'HNSCC::HN_P020.nii.gz', 'HNSCC::HN_P035.nii.gz', 'HNSCC::HN_P023.nii.gz', 'MSD::liver_117.nii.gz', 'MSD::liver_163.nii.gz', 'MSD::liver_90.nii.gz', 'MSD::liver_128.nii.gz', 'MSD::liver_188.nii.gz', 'MSD::liver_114.nii.gz', 'MSD::liver_68.nii.gz', 'MSD::liver_56.nii.gz', 'MSD::liver_59.nii.gz', 'MSD::liver_91.nii.gz', 'MSD::liver_121.nii.gz', 'MSD::liver_179.nii.gz', 'MSD::liver_158.nii.gz', 'MSD::liver_67.nii.gz', 'MSD::liver_113.nii.gz', 'MSD::liver_60.nii.gz', 'MSD::liver_62.nii.gz', 'MSD::liver_23.nii.gz', 'MSD::liver_107.nii.gz', 'MSD::liver_63.nii.gz', 'MSD::liver_116.nii.gz', 'MSD::liver_186.nii.gz', 'MSD::liver_180.nii.gz', 'MSD::liver_195.nii.gz', 'MSD::liver_51.nii.gz', 'MSD::liver_169.nii.gz', 'MSD::liver_66.nii.gz', 'MSD::liver_185.nii.gz', 'MSD::liver_32.nii.gz', 'MSD::liver_175.nii.gz', 'MSD::liver_126.nii.gz', 'MSD::liver_8.nii.gz', 'MSD::liver_95.nii.gz', 'MSD::liver_129.nii.gz', 'MSD::liver_110.nii.gz', 'MSD::liver_159.nii.gz', 'MSD::liver_144.nii.gz', 'MSD::liver_88.nii.gz', 'MSD::liver_48.nii.gz', 'MSD::liver_29.nii.gz', 'MSD::liver_145.nii.gz', 'MSD::liver_58.nii.gz', 'MSD::liver_26.nii.gz', 'MSD::liver_189.nii.gz', 'MSD::liver_93.nii.gz', 'MSD::liver_100.nii.gz', 'MSD::liver_5.nii.gz', 'MSD::liver_70.nii.gz', 'MSD::liver_178.nii.gz', 'MSD::liver_171.nii.gz', 'MSD::liver_197.nii.gz', 'MSD::liver_176.nii.gz', 'MSD::liver_123.nii.gz', 'MSD::liver_119.nii.gz', 'MSD::liver_89.nii.gz', 'MSD::liver_53.nii.gz', 'MSD::liver_199.nii.gz', 'MSD::liver_142.nii.gz', 'MSD::liver_39.nii.gz', 'MSD::liver_12.nii.gz', 'MSD::liver_96.nii.gz', 'MSD::liver_47.nii.gz', 'MSD::liver_19.nii.gz', 'MSD::liver_65.nii.gz', 'MSD::liver_109.nii.gz', 'MSD::liver_104.nii.gz', 'MSD::liver_124.nii.gz', 'MSD::liver_94.nii.gz', 'MSD::liver_102.nii.gz', 'MSD::liver_13.nii.gz', 'MSD::liver_41.nii.gz', 'MSD::liver_7.nii.gz', 'MSD::liver_196.nii.gz', 'MSD::liver_108.nii.gz', 'MSD::liver_97.nii.gz', 'MSD::liver_71.nii.gz', 'MSD::liver_0.nii.gz', 'MSD::liver_92.nii.gz', 'MSD::liver_76.nii.gz', 'MSD::liver_184.nii.gz', 'MSD::liver_162.nii.gz', 'MSD::liver_43.nii.gz', 'MSD::liver_125.nii.gz', 'MSD::liver_155.nii.gz', 'MSD::liver_79.nii.gz', 'MSD::liver_122.nii.gz', 'MSD::liver_120.nii.gz', 'MSD::liver_83.nii.gz', 'MSD::liver_21.nii.gz', 'MSD::liver_2.nii.gz', 'MSD::liver_25.nii.gz', 'MSD::liver_15.nii.gz', 'MSD::liver_50.nii.gz', 'MSD::liver_27.nii.gz', 'MSD::liver_69.nii.gz', 'MSD::liver_192.nii.gz', 'MSD::liver_99.nii.gz', 'MSD::liver_167.nii.gz', 'MSD::liver_4.nii.gz', 'MSD::liver_86.nii.gz', 'MSD::liver_49.nii.gz', 'MSD::liver_181.nii.gz', 'MSD::liver_194.nii.gz', 'MSD::liver_130.nii.gz', 'MSD::liver_75.nii.gz', 'MSD::liver_84.nii.gz', 'MSD::liver_9.nii.gz', 'MSD::liver_182.nii.gz', 'MSD::liver_34.nii.gz', 'MSD::liver_61.nii.gz', 'MSD::liver_64.nii.gz', 'MSD::liver_54.nii.gz', 'MSD::liver_173.nii.gz', 'MSD::liver_103.nii.gz', 'MSD::liver_1.nii.gz', 'MSD::liver_101.nii.gz', 'MSD::liver_52.nii.gz', 'MSD::liver_17.nii.gz', 'MSD::liver_14.nii.gz', 'MSD::liver_20.nii.gz', 'COVID::volume-covid19-A-0173_ct.nii.gz', 'COVID::volume-covid19-A-0090_ct.nii.gz', 'COVID::volume-covid19-A-0014_ct.nii.gz', 'COVID::volume-covid19-A-0147_ct.nii.gz', 'COVID::volume-covid19-A-0316_ct.nii.gz', 'COVID::volume-covid19-A-0313_ct.nii.gz', 'COVID::volume-covid19-A-0382_ct.nii.gz', 'COVID::volume-covid19-A-0202_0_ct.nii.gz', 'COVID::volume-covid19-A-0319_ct.nii.gz', 'COVID::volume-covid19-A-0377_ct.nii.gz', 'COVID::volume-covid19-A-0263_ct.nii.gz', 'COVID::volume-covid19-A-0251_ct.nii.gz', 'COVID::volume-covid19-A-0331_ct.nii.gz', 'COVID::volume-covid19-A-0096_ct.nii.gz', 'COVID::volume-covid19-A-0003_ct.nii.gz', 'COVID::volume-covid19-A-0372_ct.nii.gz', 'COVID::volume-covid19-A-0215_ct.nii.gz', 'COVID::volume-covid19-A-0016_ct.nii.gz', 'COVID::volume-covid19-A-0282_ct.nii.gz', 'COVID::volume-covid19-A-0380_ct.nii.gz', 'COVID::volume-covid19-A-0011_ct.nii.gz', 'COVID::volume-covid19-A-0046_ct.nii.gz', 'COVID::volume-covid19-A-0106_ct.nii.gz', 'COVID::volume-covid19-A-0320_ct.nii.gz', 'COVID::volume-covid19-A-0288_ct.nii.gz', 'COVID::volume-covid19-A-0187_ct.nii.gz', 'COVID::volume-covid19-A-0237_ct.nii.gz', 'COVID::volume-covid19-A-0360_ct.nii.gz', 'COVID::volume-covid19-A-0354_ct.nii.gz', 'COVID::volume-covid19-A-0252_ct.nii.gz', 'COVID::volume-covid19-A-0361_ct.nii.gz', 'COVID::volume-covid19-A-0154_ct.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0213.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0058.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0185.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0035.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0192.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0057.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0113.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0114.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0096.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0091.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0032.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0047.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0146.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0087.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0078.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0059.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0042.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0005.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0190.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0084.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0148.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0010.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0052.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0075.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0065.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0120.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0093.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0069.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0144.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0160.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0076.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0194.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0083.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0145.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0015.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0068.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0012.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0143.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0016.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0111.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0001.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0131.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0037.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0128.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0013.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0138.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0077.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0095.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0152.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0099.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0030.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0182.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0098.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0044.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0186.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0123.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0041.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0092.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0027.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0167.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0183.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0199.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0082.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0163.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0024.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0049.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0081.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0141.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0085.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0151.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0060.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0112.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0026.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0018.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0127.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0125.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0033.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0066.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0165.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0055.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0089.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0203.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0159.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0006.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0048.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0105.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0045.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0208.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0051.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0107.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0036.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0004.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0019.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0124.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0064.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0202.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0104.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0067.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0029.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0207.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0116.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0133.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0200.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0110.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0028.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0206.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0205.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0053.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0106.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0204.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0031.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0142.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0023.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0090.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0121.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0062.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0080.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0132.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0162.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0017.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0071.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0174.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0008.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0153.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0020.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0156.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0021.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0014.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0034.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0209.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0168.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0179.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0115.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0181.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0193.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0187.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0166.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0088.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0178.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0164.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0079.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0201.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0126.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0056.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0139.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0175.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0210.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0009.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0038.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0150.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0050.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0177.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0002.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0119.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0063.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0118.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0188.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0197.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0007.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0103.nii.gz'], 'val_volumes': ['HNSCC::HN_P009.nii.gz', 'HNSCC::HN_P012.nii.gz', 'HNSCC::HN_P016.nii.gz', 'MSD::liver_198.nii.gz', 'MSD::liver_135.nii.gz', 'MSD::liver_165.nii.gz', 'MSD::liver_105.nii.gz', 'MSD::liver_72.nii.gz', 'MSD::liver_161.nii.gz', 'MSD::liver_190.nii.gz', 'MSD::liver_87.nii.gz', 'MSD::liver_6.nii.gz', 'MSD::liver_33.nii.gz', 'MSD::liver_24.nii.gz', 'MSD::liver_80.nii.gz', 'MSD::liver_172.nii.gz', 'MSD::liver_31.nii.gz', 'MSD::liver_106.nii.gz', 'COVID::volume-covid19-A-0270_ct.nii.gz', 'COVID::volume-covid19-A-0338_ct.nii.gz', 'COVID::volume-covid19-A-0073_ct.nii.gz', 'COVID::volume-covid19-A-0301_ct.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0043.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0170.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0161.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0011.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0137.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0109.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0086.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0040.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0136.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0180.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0172.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0054.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0100.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0191.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0189.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0094.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0157.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0176.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0158.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0003.nii.gz'], 'test_volumes': ['HNSCC::HN_P033.nii.gz', 'HNSCC::HN_P022.nii.gz', 'HNSCC::HN_P008.nii.gz', 'MSD::liver_111.nii.gz', 'MSD::liver_18.nii.gz', 'MSD::liver_3.nii.gz', 'MSD::liver_82.nii.gz', 'MSD::liver_115.nii.gz', 'MSD::liver_193.nii.gz', 'MSD::liver_118.nii.gz', 'MSD::liver_85.nii.gz', 'MSD::liver_98.nii.gz', 'MSD::liver_81.nii.gz', 'MSD::liver_183.nii.gz', 'MSD::liver_174.nii.gz', 'MSD::liver_57.nii.gz', 'MSD::liver_170.nii.gz', 'MSD::liver_112.nii.gz', 'COVID::volume-covid19-A-0070_ct.nii.gz', 'COVID::volume-covid19-A-0025_ct.nii.gz', 'COVID::volume-covid19-A-0013_ct.nii.gz', 'COVID::volume-covid19-A-0120_ct.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0097.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0074.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0130.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0101.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0154.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0149.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0102.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0122.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0073.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0025.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0039.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0155.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0129.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0196.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0195.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0072.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0212.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0211.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0171.nii.gz', 'COLONOG::1.3.6.1.4.1.9328.50.4.0184.nii.gz']}
MAIN_CONFIG = {'split_unit': 'volume', 'split_seed': 42, 'model_seed': 42, 'train_volume_count': 337, 'val_volume_count': 42, 'test_volume_count': 42, 'train_slice_count': 5055, 'val_slice_count': 630, 'test_slice_count': 630, 'interslice_K': 5, 'use_3slice_input': True, 'use_zpos': True, 'use_interslice_mamba': True, 'use_residual_mamba': True, 'model_base': 48, 'mamba_d_state': 16, 'mamba_d_conv': 4, 'mamba_expand': 2, 'loss': {'main': 'CE-weighted-25cls', 'aux2': 'CE-uniform-2cls', 'aux4': 'CE-uniform-4cls', 'lam2': 0.3, 'lam4': 0.3, 'dice_weight': 0.0}, 'optimizer': 'AdamW', 'lr': 3e-05, 'weight_decay': 0.0001, 'scheduler': 'CosineAnnealingLR', 'batch_size': 8, 'epochs': 100, 'amp': True}

#!/usr/bin/env python3
"""
train_laa_mamba_volume_split_v2.py
──────────────────────────────────
LAA-Mamba U-Net training with volume-level 80/10/10 train/val/test split.

Key features vs v1:
  - 3-way split: train / val / test (all strictly disjoint at volume level)
  - val set  → best.pth selection (early stopping criterion)
  - test set → final evaluation only, NEVER used during training
  - AMP mixed precision (torch.cuda.amp) – no change to loss/metrics/network
  - --split_seed (default 42) and --seed/--seeds remain independent
  - Old checkpoints from slice-level experiments are refused
  - batch_size=8 (unchanged)

Usage – generate split only:
    python volume_level_experiments_v1/scripts/train_laa_mamba_volume_split_v2.py \\
        --generate_split_only

Usage – dry run (3 epochs):
    python volume_level_experiments_v1/scripts/train_laa_mamba_volume_split_v2.py \\
        --seed 42 --split_seed 42 --epochs 3 --batch_size 8 \\
        --use_3slice_input --use_interslice_mamba --use_zpos \\
        --interslice_K 5 --dry_run

Usage – full training (after dry-run confirmed):
    python volume_level_experiments_v1/scripts/train_laa_mamba_volume_split_v2.py \\
        --seeds 42 123 3407 --split_seed 42 --epochs 100 --batch_size 8 \\
        --lr 2e-4 --lam2 0.3 --lam4 0.3 \\
        --use_3slice_input --use_interslice_mamba --use_zpos --interslice_K 5 \\
        --output_dir volume_level_experiments_v1/results/Full_V17

Usage – test-set evaluation (after training):
    python volume_level_experiments_v1/scripts/train_laa_mamba_volume_split_v2.py \\
        --seed 42 --split_seed 42 --eval_test_only \\
        --resume_ckpt volume_level_experiments_v1/results/Full_V17/seed_42/best.pth \\
        --use_3slice_input --use_interslice_mamba --use_zpos --interslice_K 5 \\
        --output_dir volume_level_experiments_v1/results/Full_V17
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom
from torch.utils.data import WeightedRandomSampler

REPO            = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = REPO
DATA_ROOT = os.environ.get("LAMNET_DATA_ROOT", os.path.join(REPO, "data"))
sys.path.insert(0, REPO)

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] nibabel not installed."); sys.exit(1)


_lmap = os.path.join(REPO, "comparison_experiments")
if _lmap not in sys.path:
    sys.path.insert(0, _lmap)

# ── constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES_25 = 26
SLICES_PER_VOL = 15
VERTEBRA_NAMES = {
    1:"C1",2:"C2",3:"C3",4:"C4",5:"C5",6:"C6",7:"C7",
    8:"T1",9:"T2",10:"T3",11:"T4",12:"T5",13:"T6",
    14:"T7",15:"T8",16:"T9",17:"T10",18:"T11",19:"T12",
    20:"L1",21:"L2",22:"L3",23:"L4",24:"L5",25:"L6",
}
HARD_CLASSES = {1,3,4,5,8,13,14,15,25}

_FORBIDDEN_CKPT_DIRS = [
    "LAA_Mamba_UNet_multiseed_protocolA",
    "LAA_Mamba_UNet_multiseed",
    "paper_outputs",
    "ablation_results_protocolA",
    "archive_legacy_ablation_20260602",
    "protocolA_final_eval",
    "protocolA_retrain",
    "v17_laa_mamba_experiment",
    "final_seed42_v16_official_results",
    "results_volume_seed42",
    "runs_volume_seed42",
]

SOURCES_CFG = {
    "HNSCC": {
        "vol_path":   os.path.join(DATA_ROOT, 'HNSCC-3DCT-RT', 'volume'),
        "lbl_path":   os.path.join(DATA_ROOT, 'HNSCC-3DCT-RT', 'label'),
        "vol_suffix": ".nii.gz", "lbl_suffix": "_seg.nii.gz", "max_cases": 31,
    },
    "MSD": {
        "vol_path":   os.path.join(DATA_ROOT, 'CTSpineMSD', 'volume'),
        "lbl_path":   os.path.join(DATA_ROOT, 'CTSpineMSD', 'label'),
        "vol_suffix": ".nii.gz", "lbl_suffix": "_seg.nii.gz", "max_cases": 150,
    },
    "COVID": {
        "vol_path":   os.path.join(DATA_ROOT, 'COVID-19', 'volume'),
        "lbl_path":   os.path.join(DATA_ROOT, 'COVID-19', 'label'),
        "vol_suffix": "_ct.nii.gz", "lbl_suffix": "_ct_seg.nii.gz", "max_cases": 40,
    },
    "COLONOG": {
        "vol_path":   os.path.join(DATA_ROOT, 'COLONOG', 'volume'),
        "lbl_path":   os.path.join(DATA_ROOT, 'COLONOG', 'label'),
        "vol_suffix": ".nii.gz", "lbl_suffix": "_seg.nii.gz", "max_cases": 200,
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _check_forbidden_ckpt(path: str):
    if not path:
        return
    abs_path = os.path.abspath(path)
    for forbidden in _FORBIDDEN_CKPT_DIRS:
        if forbidden in abs_path:
            print(
                f"\n[ERROR] This checkpoint belongs to the previous slice-level protocol "
                f"and cannot be used for volume-level validation.\n"
                f"        Path: {abs_path}\n"
                f"        Forbidden pattern: {forbidden}\n"
            )
            sys.exit(1)


def _write_txt(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(str(i) for i in items) + "\n")


# ── split ──────────────────────────────────────────────────────────────────────
def _scan_source_files(src, cfg):
    vp = cfg["vol_path"]
    if not os.path.isdir(vp):
        print(f"  [SKIP] {src}: directory not found: {vp}", flush=True)
        return []
    files = sorted(f for f in os.listdir(vp) if f.endswith(".nii.gz"))[: cfg["max_cases"]]
    return [f"{src}::{f}" for f in files]


def load_split(split_json_path: str, split_seed: int):
    if os.path.isfile(split_json_path):
        with open(split_json_path) as f:
            split = json.load(f)
        assert split["split_seed"] == split_seed, \
            f"Cached split seed={split['split_seed']}, requested {split_seed}"
        print(f"  Loaded split: {split_json_path}", flush=True)
        print(f"  train={len(split['train_volumes'])}  "
              f"val={len(split['val_volumes'])}  "
              f"test={len(split['test_volumes'])}", flush=True)
        return split
    if os.path.abspath(split_json_path) == os.path.join(REPO, "split_seed42.json") and split_seed == 42:
        print("Using embedded paper split: 337/42/42 volumes.")
        return PAPER_SPLIT
    raise FileNotFoundError(
        f"Split JSON not found: {split_json_path}\n"
        f"Run: python volume_level_experiments_v1/scripts/generate_split_80_10_10.py"
    )


# ── volume data loading ────────────────────────────────────────────────────────
def load_volume_slices(vol_id: str):
    """Load one CT volume → slices, labels, z-positions, sampled z-indices."""
    src, fname = vol_id.split("::", 1)
    cfg      = SOURCES_CFG[src]
    img_path = os.path.join(cfg["vol_path"], fname)
    lbl_fname= fname.replace(cfg["vol_suffix"], cfg["lbl_suffix"])
    lbl_path = os.path.join(cfg["lbl_path"], lbl_fname)

    if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
        print(f"  [SKIP] Missing file for {vol_id}", flush=True)
        return None
    try:
        vol = nib.load(img_path).get_fdata()
        lbl = np.clip(nib.load(lbl_path).get_fdata().astype(np.int64), 0, 25)
    except Exception as e:
        print(f"  [ERROR] Load failed {vol_id}: {e}", flush=True)
        return None

    eligible = [z for z in range(vol.shape[2]) if int((lbl[:,:,z]>0).sum()) >= 50]
    if not eligible:
        eligible = list(range(vol.shape[2]))

    pos       = np.linspace(0, len(eligible)-1, num=SLICES_PER_VOL)
    sampled_z = [eligible[int(p)] for p in pos]

    slices, m25s, m4s, m2s, zpos_list = [], [], [], [], []
    for rank, z in enumerate(sampled_z):
        sl  = vol[:,:,z].astype(np.float32)
        m25 = lbl[:,:,z]
        p1, p99 = np.percentile(sl, [1, 99])
        sl  = np.clip((sl-p1)/(p99-p1+1e-8), 0, 1).astype(np.float32)
        h, w = sl.shape
        slices.append(zoom(sl,  (256/h, 256/w), order=1).astype(np.float32))
        m25s.append(  zoom(m25, (256/h, 256/w), order=0).astype(np.int16))
        m4s.append(   zoom(label_25_to_4class(m25),(256/h,256/w),order=0).astype(np.int8))
        m2s.append(   zoom(label_25_to_2class(m25),(256/h,256/w),order=0).astype(np.int8))
        zpos_list.append(float(rank / (SLICES_PER_VOL - 1)))
    del vol, lbl
    return slices, m25s, m4s, m2s, zpos_list, sampled_z


def build_dataset_from_volumes(vol_ids: list, K: int = 5, desc: str = ""):
    """Build (N,K,H,W) sample arrays from a list of volume IDs."""
    half = K // 2
    all_imgs, all_m25, all_m4, all_m2, all_zpos = [], [], [], [], []
    all_vol_ids, all_ranks, all_z = [], [], []
    failed = []

    for vi, vol_id in enumerate(vol_ids):
        result = load_volume_slices(vol_id)
        if result is None:
            failed.append(vol_id); continue
        slices, m25s, m4s, m2s, zpos_list, sampled_z = result
        assert len(slices) == SLICES_PER_VOL, \
            f"{vol_id}: got {len(slices)} slices, expected {SLICES_PER_VOL}"

        for target_rank in range(SLICES_PER_VOL):
            ctx_ranks = [max(0, min(SLICES_PER_VOL-1, target_rank+(ki-half)))
                         for ki in range(K)]
            assert all(0 <= cr < SLICES_PER_VOL for cr in ctx_ranks), \
                f"Cross-volume context in {vol_id} rank={target_rank}"
            all_imgs.append(np.stack([slices[cr]    for cr in ctx_ranks]))   # (K,H,W)
            all_zpos.append(np.array([zpos_list[cr] for cr in ctx_ranks], dtype=np.float32))
            all_m25.append(m25s[target_rank])
            all_m4.append(m4s[target_rank])
            all_m2.append(m2s[target_rank])
            all_vol_ids.append(vol_id)
            all_ranks.append(target_rank)
            all_z.append(sampled_z[target_rank])

        if (vi+1) % 50 == 0:
            print(f"    [{desc}] {vi+1}/{len(vol_ids)} vols  "
                  f"{len(all_imgs)} samples", flush=True)

    if failed:
        print(f"\n[WARN] {len(failed)} volumes failed ({desc}): {failed[:5]}", flush=True)
    if not all_imgs:
        raise RuntimeError(f"No samples built for {desc}. Check data paths.")

    print(f"  [{desc}] {len(all_imgs)} samples from "
          f"{len(vol_ids)-len(failed)} volumes", flush=True)
    return (np.stack(all_imgs).astype(np.float32),   # (N,K,H,W)
            np.stack(all_m25).astype(np.int16),       # (N,H,W)
            np.stack(all_m4).astype(np.int8),
            np.stack(all_m2).astype(np.int8),
            np.stack(all_zpos).astype(np.float32),    # (N,K)
            all_vol_ids, all_ranks, all_z)


# ── pre-training leak audit ────────────────────────────────────────────────────
def run_leak_audit(train_vol_ids, val_vol_ids, test_vol_ids,
                   tr_sample_vids, va_sample_vids, te_sample_vids):
    """Abort if any pairwise volume overlap is detected."""
    print("\n" + "=" * 65)
    print("  Volume-level split audit")
    tr = set(train_vol_ids); va = set(val_vol_ids); te = set(test_vol_ids)

    ov_tr_va = tr & va
    ov_tr_te = tr & te
    ov_va_te = va & te

    print(f"  Train volumes            : {len(tr)}")
    print(f"  Validation volumes       : {len(va)}")
    print(f"  Test volumes             : {len(te)}")
    print(f"  Train target slices      : {len(tr_sample_vids)}")
    print(f"  Validation target slices : {len(va_sample_vids)}")
    print(f"  Test target slices       : {len(te_sample_vids)}")
    print(f"  Train/validation overlap : {len(ov_tr_va)}")
    print(f"  Train/test overlap       : {len(ov_tr_te)}")
    print(f"  Validation/test overlap  : {len(ov_va_te)}")
    print(f"  Cross-volume ctx windows : 0")          # guaranteed by clamping
    # test images in training contexts: impossible when volumes are disjoint
    test_in_train = 0 if not (ov_tr_te or ov_va_te) else -1
    print(f"  Test images in train ctx : {test_in_train}")
    print("=" * 65)

    errors = []
    if ov_tr_va: errors.append(f"Train/val overlap: {len(ov_tr_va)}")
    if ov_tr_te: errors.append(f"Train/test overlap: {len(ov_tr_te)}")
    if ov_va_te: errors.append(f"Val/test overlap: {len(ov_va_te)}")
    if errors:
        print("\n[ABORT] Leak audit FAILED:")
        for e in errors:
            print(f"        {e}")
        sys.exit(1)
    print("\n  Leak audit: PASS – proceeding\n")


# ── input tensor construction ──────────────────────────────────────────────────
def make_input_tensor(imgs_K, zpos_K, indices, args, device):
    """Build (B,K,4,H,W) tensor for interslice-Mamba path."""
    raw = torch.from_numpy(imgs_K[indices].astype(np.float32))   # (B,K,H,W)
    zp  = torch.from_numpy(zpos_K[indices].astype(np.float32))   # (B,K)
    B, K, H, W = raw.shape

    if args.use_interslice_mamba:
        if args.use_3slice_input:
            prev_ = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
            next_ = torch.cat([raw[:, 1:], raw[:, -1:]], dim=1)
            imgs_ch = torch.stack([prev_, raw, next_], dim=2)     # (B,K,3,H,W)
        else:
            imgs_ch = raw.unsqueeze(2)                             # (B,K,1,H,W)
        zp_ch = zp.view(B, K, 1, 1, 1).expand(-1, -1, 1, H, W)
        x = torch.cat([imgs_ch, zp_ch], dim=2).to(device)        # (B,K,4,H,W)
    elif args.use_3slice_input:
        x = raw.to(device)
        if args.use_zpos:
            zpm = zp.view(B, 1, 1, 1).expand(-1, 1, H, W).to(device)
            x = torch.cat([x, zpm], dim=1)
    else:
        x = raw.unsqueeze(1).to(device)
        if args.use_zpos:
            zpm = zp.view(B, 1, 1, 1).expand(-1, 1, H, W).to(device)
            x = torch.cat([x, zpm], dim=1)
    return x


# ── class / sample weighting ───────────────────────────────────────────────────
def compute_class_weights(tr_m25, num_classes=26, device="cpu", power=0.5, cap=None, cervical_boost=1.0):
    """
    Compute class weights from training labels.
    
    Args:
        cervical_boost: Multiplier for cervical vertebrae (C1-C7, classes 1-7).
                       Default 1.0 (no boost). Use 2.0-3.0 to improve C1-C5 performance.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for m in tr_m25:
        vals, cnts = np.unique(m, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            if 0 <= v < num_classes:
                counts[v] += c
    counts = np.maximum(counts.astype(np.float64), 1)
    freq   = counts / counts.sum()
    w      = 1.0 / (freq ** power)
    w[0]  *= 0.1
    
    # Apply cervical boost (C1-C7 are classes 1-7)
    if cervical_boost != 1.0:
        w[1:8] *= cervical_boost
    
    w      = w / w.sum() * num_classes
    if cap is not None:
        w = np.minimum(w, cap); w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32).to(device)


def compute_sample_weights(tr_m25, hard_weight=2.0, hard_cap=3.0):
    weights = np.ones(len(tr_m25), dtype=np.float64)
    for idx, mask in enumerate(tr_m25):
        hard_count = sum(1 for cls in HARD_CLASSES if np.any(mask == cls))
        if hard_count > 0:
            weights[idx] = min(hard_cap, hard_weight + 0.5*(hard_count-1))
    return weights


def multiclass_dice_loss(logits, targets, smooth=1e-5):
    B, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)
    t_oh  = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
    inter = (probs * t_oh).sum(dim=(0,2,3))
    denom = probs.sum(dim=(0,2,3)) + t_oh.sum(dim=(0,2,3))
    dice  = (2.0*inter + smooth) / (denom + smooth)
    return 1.0 - dice[1:].mean()


# ── metrics ────────────────────────────────────────────────────────────────────
def compute_mov_and_perclass(p25, g25):
    dices = {}
    for c in range(1, NUM_CLASSES_25):
        gc = (g25 == c).float()
        if gc.sum() == 0: continue
        pc = (p25 == c).float()
        dices[c] = float(2.0*(pc*gc).sum()/(pc.sum()+gc.sum()+1e-8))
    mov = float(np.mean(list(dices.values()))) if dices else 0.0
    return mov, dices


def compute_miou(p25, g25):
    ious = []
    for c in range(1, NUM_CLASSES_25):
        gm=(g25==c); pm=(p25==c)
        if gm.sum()==0: continue
        ious.append((gm&pm).sum().item() / ((gm|pm).sum().item()+1e-8))
    return float(np.mean(ious)) if ious else 0.0


def compute_hd95(p_bin, g_bin):
    if p_bin.sum()==0 or g_bin.sum()==0: return 999.0
    dt_g = distance_transform_edt(~g_bin)
    dt_p = distance_transform_edt(~p_bin)
    return float(max(np.percentile(dt_g[p_bin], 95), np.percentile(dt_p[g_bin], 95)))


def compute_asd(p_bin, g_bin):
    if p_bin.sum()==0 or g_bin.sum()==0: return 999.0
    try:
        ps = p_bin ^ binary_erosion(p_bin)
        gs = g_bin ^ binary_erosion(g_bin)
        if ps.sum()==0 or gs.sum()==0: return 999.0
        dt_g = distance_transform_edt(~g_bin)
        dt_p = distance_transform_edt(~p_bin)
        return float((dt_g[ps].mean()+dt_p[gs].mean())/2.0)
    except Exception:
        return 999.0


def evaluate(model, data_tuple, device, batch_size, args, compute_asd_flag=False):
    """Evaluate model on a dataset tuple. Returns metrics dict."""
    model.eval()
    imgs_K, m25_np = data_tuple[0], data_tuple[1]
    zpos_K         = data_tuple[4]
    n = len(imgs_K)
    mov_list, miou_list, hd_list, asd_list = [], [], [], []
    per_class_acc = {c: [] for c in range(1, NUM_CLASSES_25)}
    first_batch_diag_done = False

    with torch.no_grad():
        for i in range(0, n, batch_size*2):
            idx = np.arange(i, min(i+batch_size*2, n))
            x   = make_input_tensor(imgs_K, zpos_K, idx, args, device)
            y25 = torch.from_numpy(m25_np[idx].astype(np.int64))
            logits25 = model(x)["logits_25"].cpu()
            p25 = logits25.argmax(1)

            # First-batch diagnostics (printed to stdout once per evaluate call)
            if not first_batch_diag_done:
                first_batch_diag_done = True
                t_uniq, t_cnt = torch.unique(y25, return_counts=True)
                p_uniq, p_cnt = torch.unique(p25, return_counts=True)
                print(f"  [VAL DIAG] logits25 shape={tuple(logits25.shape)}, "
                      f"min={logits25.min().item():.4f}, max={logits25.max().item():.4f}, "
                      f"finite={torch.isfinite(logits25).all().item()}", flush=True)
                print(f"  [VAL DIAG] target classes: "
                      f"{list(zip(t_uniq.tolist(), t_cnt.tolist()))[:15]}", flush=True)
                print(f"  [VAL DIAG] pred   classes: "
                      f"{list(zip(p_uniq.tolist(), p_cnt.tolist()))[:15]}", flush=True)

            for j in range(p25.shape[0]):
                mov, dices = compute_mov_and_perclass(p25[j], y25[j])
                mov_list.append(mov)
                miou_list.append(compute_miou(p25[j], y25[j]))
                for c, d in dices.items():
                    per_class_acc[c].append(d)
                pb = (p25[j]>0).numpy().astype(bool)
                gb = (y25[j]>0).numpy().astype(bool)
                hd_list.append(compute_hd95(pb, gb))
                if compute_asd_flag:
                    asd_list.append(compute_asd(pb, gb))

    hd_clean  = [v for v in hd_list  if v < 900]
    asd_clean = [v for v in asd_list if v < 900]
    per_class_mean = {c: float(np.mean(v)) if v else 0.0
                      for c, v in per_class_acc.items()}
    return {
        "MoV":  round(float(np.mean(mov_list)) *100, 4),
        "mIoU": round(float(np.mean(miou_list))*100, 4),
        "HD95": round(float(np.mean(hd_clean))  if hd_clean  else 999.0, 4),
        "ASD":  round(float(np.mean(asd_clean)) if asd_clean else -1.0,  4),
        "per_class_dice": {VERTEBRA_NAMES[c]: round(v*100, 4)
                           for c, v in per_class_mean.items()},
    }


# ── config saving ──────────────────────────────────────────────────────────────
def save_run_config(save_dir, args, split_json_path, split_data, tr_n, va_n, te_n):
    os.makedirs(save_dir, exist_ok=True)
    cfg = {
        "split_unit": "volume", "split_seed": args.split_seed,
        "model_seed": getattr(args, "_current_seed", "?"),
        "train_volume_count": len(split_data["train_volumes"]),
        "val_volume_count":   len(split_data["val_volumes"]),
        "test_volume_count":  len(split_data["test_volumes"]),
        "train_slice_count": tr_n, "val_slice_count": va_n, "test_slice_count": te_n,
        "interslice_K": args.interslice_K,
        "use_3slice_input": args.use_3slice_input,
        "use_zpos": args.use_zpos,
        "use_interslice_mamba": args.use_interslice_mamba,
        "use_residual_mamba": args.use_residual_mamba,
        "model_base": 48, "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
        "loss": {"main": "CE-weighted-25cls", "aux2": "CE-uniform-2cls",
                 "aux4": "CE-uniform-4cls",
                 "lam2": args.lam2, "lam4": args.lam4,
                 "dice_weight": args.dice_weight},
        "optimizer": "AdamW", "lr": args.lr, "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
        "batch_size": args.batch_size, "epochs": args.epochs, "amp": True,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(save_dir, "split_file_used.txt"), "w") as f:
        f.write(os.path.abspath(split_json_path) + "\n")
    env = {"python": sys.version, "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import nibabel; env["nibabel"] = nibabel.__version__
    except Exception:
        pass
    with open(os.path.join(save_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)


# ── training loop (AMP) ────────────────────────────────────────────────────────
def train_one_seed(seed, train_data, val_data, args, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt    = os.path.join(save_dir, "best.pth")
    last_ckpt    = os.path.join(save_dir, "last.pth")
    metrics_path = os.path.join(save_dir, "metrics.json")
    log_path     = os.path.join(save_dir, "training_log.txt")

    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()

    set_seed(seed)
    model = LAA_Mamba_SpineSeg(
        base=48, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        force_s4d=args.force_s4d, use_zpos=args.use_zpos,
        use_residual_mamba=args.use_residual_mamba,
        use_3slice_input=args.use_3slice_input,
        use_interslice_mamba=args.use_interslice_mamba,
        interslice_K=args.interslice_K,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for attr in ("use_interslice_mamba","use_3slice_input","use_zpos",
                 "use_residual_mamba","interslice_K"):
        assert hasattr(model, attr), f"model missing: {attr}"

    # Shape check
    with torch.no_grad():
        dummy_idx = np.arange(min(2, len(train_data[0])))
        dx = make_input_tensor(train_data[0], train_data[4], dummy_idx, args, device)
        do = model(dx)
        print(f"  [SHAPE] input={tuple(dx.shape)}  "
              f"logits_25={tuple(do['logits_25'].shape)}  "
              f"logits_4={tuple(do['logits_4'].shape)}  "
              f"logits_2={tuple(do['logits_2'].shape)}", flush=True)
        assert dx.shape[1:] == (args.interslice_K, 4, 256, 256)
        assert do["logits_25"].shape[1:] == (26, 256, 256)
        assert do["logits_4"].shape[1:]  == (5,  256, 256)
        assert do["logits_2"].shape[1:]  == (2,  256, 256)
    del dx, do

    use_amp = device.type == "cuda" and not args.no_amp
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    t_max = args.scheduler_T_max if args.scheduler_T_max is not None else args.epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6)

    imgs_K, m25_np, m4_np, m2_np, zpos_K = train_data[:5]
    n_train = len(imgs_K)

    sample_weights = None
    if args.hard_class_sampler:
        sample_weights = compute_sample_weights(m25_np,
                                                hard_weight=args.hard_weight,
                                                hard_cap=args.hard_cap)
    # Class weights from TRAINING set only — test set never touches this.
    # Cap at 10.0 to prevent float16 overflow under AMP.
    _cap = args.weight_cap if args.weight_cap is not None else 10.0
    class_weights = compute_class_weights(m25_np, device=device,
                                          power=args.weight_power, cap=_cap,
                                          cervical_boost=args.cervical_boost)
    ce_25 = nn.CrossEntropyLoss(weight=class_weights)
    ce    = nn.CrossEntropyLoss()

    history = []; best_mov = 0.0

    def _log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")

    _log(f"\n{'='*65}")
    _log(f"  LAA-Mamba U-Net (80/10/10)  seed={seed}  "
         f"params={n_params/1e6:.2f}M  epochs={args.epochs}  AMP={use_amp}")
    _log(f"  train_slices={n_train}  val_slices={len(val_data[0])}")
    _log(f"  class_weight_cap={_cap}  weight_power={args.weight_power}")
    if args.cervical_boost != 1.0:
        _log(f"  cervical_boost={args.cervical_boost}  (C1-C7 weights × {args.cervical_boost})")
    _log(f"{'='*65}")

    for ep in range(args.epochs):
        model.train()
        if args.hard_class_sampler and sample_weights is not None:
            sampler   = WeightedRandomSampler(sample_weights, n_train, replacement=True)
            epoch_idx = torch.tensor(list(sampler), dtype=torch.int64)
        else:
            epoch_idx = torch.randperm(n_train)

        ep_loss = 0.0; n_b = 0; t0 = time.time()
        # Detailed step tracking
        total_batches = 0
        successful_steps = 0
        skipped_steps = 0
        nonfinite_losses = 0
        nonfinite_grads = 0

        for i in range(0, n_train, args.batch_size):
            bi  = epoch_idx[i:i+args.batch_size].numpy()
            x   = make_input_tensor(imgs_K, zpos_K, bi, args, device)
            y25 = torch.from_numpy(m25_np[bi].astype(np.int64)).to(device)
            y4  = torch.from_numpy(m4_np[bi].astype(np.int64)).to(device)
            y2  = torch.from_numpy(m2_np[bi].astype(np.int64)).to(device)

            total_batches += 1

            # First batch diagnostics
            if ep == 0 and i == 0:
                _log(f"\n[DIAGNOSTICS] First training batch:")
                _log(f"  Input shape={tuple(x.shape)}, "
                     f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                     f"mean={x.mean().item():.4f}, std={x.std().item():.4f}, "
                     f"finite={torch.isfinite(x).all().item()}")
                _log(f"  Target25 shape={tuple(y25.shape)}, "
                     f"min={y25.min().item()}, max={y25.max().item()}, "
                     f"unique={torch.unique(y25).cpu().tolist()[:20]}")
                _log(f"  Target4 min={y4.min().item()}, max={y4.max().item()}")
                _log(f"  Target2 min={y2.min().item()}, max={y2.max().item()}")

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(x)
                l25  = ce_25(out["logits_25"], y25)
                if args.dice_weight > 0:
                    l25 = l25 + args.dice_weight * multiclass_dice_loss(
                        out["logits_25"], y25)
                loss = (l25
                        + args.lam2 * ce(out["logits_2"], y2)
                        + args.lam4 * ce(out["logits_4"], y4))

            # Check label/output consistency (first batch only)
            if ep == 0 and i == 0:
                num_classes_25 = out["logits_25"].shape[1]
                num_classes_4  = out["logits_4"].shape[1]
                num_classes_2  = out["logits_2"].shape[1]
                _log(f"  Logits25 shape={tuple(out['logits_25'].shape)}, "
                     f"channels={num_classes_25}")
                _log(f"  Logits4 shape={tuple(out['logits_4'].shape)}, "
                     f"channels={num_classes_4}")
                _log(f"  Logits2 shape={tuple(out['logits_2'].shape)}, "
                     f"channels={num_classes_2}")
                assert y25.max().item() < num_classes_25, \
                    f"Label/output mismatch: target25 max={y25.max().item()}, " \
                    f"logits25 channels={num_classes_25}"
                assert y4.max().item() < num_classes_4, \
                    f"Label/output mismatch: target4 max={y4.max().item()}, " \
                    f"logits4 channels={num_classes_4}"
                assert y2.max().item() < num_classes_2, \
                    f"Label/output mismatch: target2 max={y2.max().item()}, " \
                    f"logits2 channels={num_classes_2}"
                _log(f"  Loss values: l25={l25.item():.4f}, "
                     f"l2={ce(out['logits_2'],y2).item():.4f}, "
                     f"l4={ce(out['logits_4'],y4).item():.4f}, "
                     f"total={loss.item():.4f}")

            # Check for non-finite loss before backward
            if not torch.isfinite(loss):
                nonfinite_losses += 1
                if nonfinite_losses <= 3:
                    _log(f"  [NONFINITE LOSS] batch {i}: "
                         f"l25={l25.item():.4f}, "
                         f"l2={ce(out['logits_2'],y2).item():.4f}, "
                         f"l4={ce(out['logits_4'],y4).item():.4f}")
                continue

            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                nonfinite_grads += 1

            scaler.step(opt)
            scaler.update()
            scale_after = scaler.get_scale()

            # Detect if scaler skipped the step
            if scale_after < scale_before:
                skipped_steps += 1
            else:
                successful_steps += 1
                ep_loss += loss.item()
                n_b += 1

        # Only advance lr scheduler when actual optimizer steps occurred
        if n_b > 0:
            sch.step()

        ep_loss /= max(n_b, 1)
        ep_time  = time.time() - t0

        _log(f"  [OPTIMIZER] total_batches={total_batches}, "
             f"successful={successful_steps}, skipped={skipped_steps}, "
             f"nonfinite_loss={nonfinite_losses}, nonfinite_grad={nonfinite_grads}, "
             f"scaler_scale={scaler.get_scale():.1f}, "
             f"lr={opt.param_groups[0]['lr']:.8f}")

        do_asd  = ((ep+1) % 5 == 0) or (ep == args.epochs - 1)
        # Evaluate on VAL set — test set is NEVER touched here
        metrics = evaluate(model, val_data, device, args.batch_size, args, do_asd)
        metrics["epoch"]  = ep + 1
        metrics["loss"]   = round(ep_loss, 6)
        metrics["time_s"] = round(ep_time, 1)
        history.append(metrics)

        torch.save({"epoch": ep, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sch": sch.state_dict(),
                    "best_mov": best_mov}, last_ckpt)

        if metrics["MoV"] >= best_mov:   # best.pth selected by val MoV only
            best_mov = metrics["MoV"]
            torch.save(model.state_dict(), best_ckpt)

        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        _log(f"  Ep{ep+1:3d} | loss={ep_loss:.4f} | "
             f"val MoV={metrics['MoV']:.2f}% mIoU={metrics['mIoU']:.2f}% "
             f"HD95={metrics['HD95']:.2f}mm ASD={metrics['ASD']:.2f}mm | {ep_time:.1f}s")

    _log(f"\n  Done seed={seed}. Best val MoV={best_mov:.3f}%\n")
    return history, best_ckpt


# ── independent test-set evaluation ───────────────────────────────────────────
def eval_test_set(seed, test_data, args, device, save_dir, ckpt_path):
    """Load best.pth, evaluate on held-out test set. Never runs training."""
    _check_forbidden_ckpt(ckpt_path)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}"); sys.exit(1)

    set_seed(seed)
    model = LAA_Mamba_SpineSeg(
        base=48, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
        force_s4d=args.force_s4d, use_zpos=args.use_zpos,
        use_residual_mamba=args.use_residual_mamba,
        use_3slice_input=args.use_3slice_input,
        use_interslice_mamba=args.use_interslice_mamba,
        interslice_K=args.interslice_K,
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    print(f"  Loaded checkpoint: {ckpt_path}", flush=True)

    print("  Evaluating on TEST SET (ASD computed) ...", flush=True)
    metrics = evaluate(model, test_data, device, args.batch_size, args,
                       compute_asd_flag=True)
    metrics["checkpoint"] = ckpt_path
    metrics["split"] = "test"

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  TEST SET RESULTS  seed={seed}")
    print(f"  MoV={metrics['MoV']:.3f}%  mIoU={metrics['mIoU']:.3f}%  "
          f"HD95={metrics['HD95']:.3f}mm  ASD={metrics['ASD']:.3f}mm")
    print(f"  Saved → {out_path}", flush=True)
    return metrics


# ── argument parser ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LAA-Mamba vol-level 80/10/10")
    p.add_argument("--seeds",       type=int, nargs="+", default=[42, 123, 3407])
    p.add_argument("--seed",        type=int, default=None,
                   help="Single seed (overrides --seeds)")
    p.add_argument("--split_seed",  type=int, default=42)
    p.add_argument("--output_dir",  type=str,
                   default=os.path.join(EXPERIMENT_ROOT, "results", "Full_V17"))
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--lam2",        type=float, default=0.3)
    p.add_argument("--lam4",        type=float, default=0.3)
    p.add_argument("--weight_power",type=float, default=0.5)
    p.add_argument("--weight_cap",  type=float, default=None)
    p.add_argument("--cervical_boost", type=float, default=1.0,
                   help="Boost multiplier for cervical vertebrae (C1-C7) class weights. "
                        "Default 1.0 (no boost). Try 2.0-3.0 to improve C1-C5 performance.")
    p.add_argument("--dice_weight", type=float, default=0.0)
    p.add_argument("--force_s4d",   action="store_true")
    p.add_argument("--use_zpos",    action="store_true")
    p.add_argument("--use_3slice_input",     action="store_true")
    p.add_argument("--use_interslice_mamba", action="store_true")
    p.add_argument("--interslice_K",  type=int, default=5)
    p.add_argument("--no_residual_mamba", dest="use_residual_mamba",
                   action="store_false")
    p.set_defaults(use_residual_mamba=True)
    p.add_argument("--hard_class_sampler", action="store_true")
    p.add_argument("--hard_weight",  type=float, default=2.0)
    p.add_argument("--hard_cap",     type=float, default=3.0)
    p.add_argument("--resume_ckpt",  type=str, default="",
                   help="Resume/finetune from a NEW-experiment checkpoint only")
    p.add_argument("--generate_split_only", action="store_true")
    p.add_argument("--dry_run",      action="store_true",
                   help="3-epoch sanity run → results/_dryrun/")
    p.add_argument("--no_amp",       action="store_true",
                   help="Disable AMP mixed precision (use FP32 for debugging)")
    p.add_argument("--scheduler_T_max", type=int, default=None,
                   help="T_max for CosineAnnealingLR; defaults to --epochs if not set")
    p.add_argument("--eval_test_only", action="store_true",
                   help="Evaluate test set with --resume_ckpt, no training")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def train_main():
    args = parse_args()

    if args.resume_ckpt:
        _check_forbidden_ckpt(args.resume_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.seed is not None:
        args.seeds = [args.seed]
    if args.dry_run:
        args.output_dir = os.path.join(EXPERIMENT_ROOT, "results", "_dryrun")
        print(f"[DRY RUN] epochs={args.epochs}  out={args.output_dir}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    split_json_path = os.environ.get("LAMNET_SPLIT", os.path.join(REPO, "split_seed42.json"))

    print("\n--- Volume split (80/10/10) ---", flush=True)
    split_data = load_split(split_json_path, args.split_seed)

    if args.generate_split_only:
        print("Split loaded. Exiting (--generate_split_only)."); return

    # Load all three splits independently – never mix before splitting
    print("\n--- Loading training volumes ---", flush=True)
    tr = build_dataset_from_volumes(
        split_data["train_volumes"], K=args.interslice_K, desc="train")

    print("\n--- Loading validation volumes ---", flush=True)
    va = build_dataset_from_volumes(
        split_data["val_volumes"], K=args.interslice_K, desc="val")

    print("\n--- Loading test volumes ---", flush=True)
    te = build_dataset_from_volumes(
        split_data["test_volumes"], K=args.interslice_K, desc="test")

    tr_imgs_K, tr_m25, tr_m4, tr_m2, tr_zpos_K, tr_vids = (
        tr[0], tr[1], tr[2], tr[3], tr[4], tr[5])
    va_imgs_K, va_m25, va_m4, va_m2, va_zpos_K, va_vids = (
        va[0], va[1], va[2], va[3], va[4], va[5])
    te_imgs_K, te_m25, te_m4, te_m2, te_zpos_K, te_vids = (
        te[0], te[1], te[2], te[3], te[4], te[5])

    # Triple-disjoint leak audit – aborts on any violation
    run_leak_audit(
        split_data["train_volumes"], split_data["val_volumes"],
        split_data["test_volumes"],
        tr_vids, va_vids, te_vids,
    )

    # Per-volume sample count check
    for vids, label in [(tr_vids, "train"), (va_vids, "val"), (te_vids, "test")]:
        wrong = {v: c for v, c in Counter(vids).items() if c != SLICES_PER_VOL}
        if wrong:
            print(f"[WARN] {label}: volumes with != {SLICES_PER_VOL} samples: {wrong}")

    assert tr_imgs_K.ndim == 4 and tr_imgs_K.shape[1] == args.interslice_K
    assert tr_imgs_K.shape[2] == 256 and tr_imgs_K.shape[3] == 256
    print(f"\n[SHAPE] train={tr_imgs_K.shape}  val={va_imgs_K.shape}  "
          f"test={te_imgs_K.shape}", flush=True)

    train_data = (tr_imgs_K, tr_m25, tr_m4, tr_m2, tr_zpos_K, tr_vids)
    val_data   = (va_imgs_K, va_m25, va_m4, va_m2, va_zpos_K, va_vids)
    test_data  = (te_imgs_K, te_m25, te_m4, te_m2, te_zpos_K, te_vids)

    # Test-only path
    if args.eval_test_only:
        if not args.resume_ckpt:
            print("[ERROR] --eval_test_only requires --resume_ckpt"); sys.exit(1)
        for seed in args.seeds:
            args._current_seed = seed
            save_dir = os.path.join(args.output_dir, f"seed_{seed}")
            eval_test_set(seed, test_data, args, device, save_dir, args.resume_ckpt)
        return

    # Training
    for seed in args.seeds:
        args._current_seed = seed
        save_dir = os.path.join(args.output_dir, f"seed_{seed}")
        print(f"\n--- Training seed={seed} ---", flush=True)
        save_run_config(save_dir, args, split_json_path, split_data,
                        len(tr_imgs_K), len(va_imgs_K), len(te_imgs_K))
        if args.resume_ckpt:
            _check_forbidden_ckpt(args.resume_ckpt)
        train_one_seed(seed, train_data, val_data, args, device, save_dir)

    print("\n=== All seeds complete ===\n", flush=True)




"""Physical-distance functions used by the paper evaluation; empty masks return None."""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_hd95_mm(pred_bool, gt_bool, sampling):
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return None
    try:
        pred_dist = distance_transform_edt(~pred_bool, sampling=sampling)
        gt_dist = distance_transform_edt(~gt_bool, sampling=sampling)
        return float(max(np.percentile(pred_dist[gt_bool], 95),
                         np.percentile(gt_dist[pred_bool], 95)))
    except Exception:
        return None


def compute_asd_mm(pred_bool, gt_bool, sampling):
    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return None
    try:
        pred_surf = pred_bool.astype(np.uint8) ^ binary_erosion(pred_bool).astype(np.uint8)
        gt_surf = gt_bool.astype(np.uint8) ^ binary_erosion(gt_bool).astype(np.uint8)
        if pred_surf.sum() == 0 or gt_surf.sum() == 0:
            return None
        d_to_pred = distance_transform_edt(~pred_bool, sampling=sampling)
        d_to_gt = distance_transform_edt(~gt_bool, sampling=sampling)
        a = d_to_pred[gt_surf > 0]
        b = d_to_gt[pred_surf > 0]
        return float((a.mean() + b.mean()) / 2.0)
    except Exception:
        return None




"""Evaluate a main LAM-NET checkpoint; physical metrics follow the paper convention."""
import argparse,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import torch

def physical_evaluate_main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='test_results.json');p.add_argument('--device',default='cuda');a=p.parse_args()
    set_seed(42);torch.set_num_threads(4)
    split=load_split(str(Path(__file__).with_name('split_seed42.json')),42)
    data=build_dataset_from_volumes(split['test_volumes'],K=5,desc='test')
    model=LAA_Mamba_SpineSeg(base=48,use_zpos=True,use_3slice_input=True,use_interslice_mamba=True,interslice_K=5).to(a.device)
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
    args=SimpleNamespace(use_zpos=True,use_3slice_input=True,use_interslice_mamba=True,interslice_K=5)
    spacings={}
    for vid in split['test_volumes']:
        src,name=vid.split('::',1);nii=nib.load(str(Path(SOURCES_CFG[src]['vol_path'])/name));z=nii.header.get_zooms();shape=nii.shape
        spacings[vid]=(float(z[0])*shape[0]/256,float(z[1])*shape[1]/256)
    rows=[]
    with torch.inference_mode():
        for i in range(0,len(data[0]),16):
            idx=np.arange(i,min(i+16,len(data[0])));x=make_input_tensor(data[0],data[4],idx,args,a.device);pred=model(x)['logits_25'].argmax(1).cpu()
            for j,mask in enumerate(pred):
                n=int(idx[j]);gt=torch.from_numpy(data[1][n].astype(np.int64));dice,_=compute_mov_and_perclass(mask,gt);iou=compute_miou(mask,gt)
                pb=mask.numpy()>0;gb=gnumpy()>0;vid=data[5][n]
                rows.append(dict(volume_id=vid,MoV=float(dice),mIoU=float(iou),HD95_mm=compute_hd95_mm(pb,gb,spacings[vid]),ASD_mm=compute_asd_mm(pb,gb,spacings[vid])))
    result={'n_slices':len(rows),'MoV_percent':float(np.mean([r['MoV'] for r in rows])*100),'mIoU_percent':float(np.mean([r['mIoU'] for r in rows])*100)}
    for key in ['HD95_mm','ASD_mm']:
        vals=[r[key] for r in rows if r[key] is not None and math.isfinite(r[key])];result[key]=float(np.mean(vals)) if vals else None;result[key+'_valid_count']=len(vals)
    Path(a.output).write_text(json.dumps({'summary':result,'per_slice':rows},indent=2));print(json.dumps(result,indent=2))



def smoke_test():
    import torch,numpy as np
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    m=LAA_Mamba_SpineSeg(base=48,use_zpos=True,use_3slice_input=True,use_interslice_mamba=True,interslice_K=5).to(device).eval()
    assert m.interslice_mamba._use_mamba, 'Paper reproduction requires the real mamba_ssm backend.'
    with torch.inference_mode():out=m(torch.zeros(1,5,4,64,64,device=device))
    for name,c in [('logits_25',26),('logits_4',5),('logits_2',2)]:
        assert out[name].shape==(1,c,64,64) and torch.isfinite(out[name]).all()
    pred=np.zeros((8,8),dtype=bool);gt=pred.copy();pred[2,2]=True;gt[4,2]=True
    assert compute_hd95_mm(pred,gt,(2.,1.))==4. and compute_asd_mm(pred,gt,(2.,1.))==4.
    assert compute_hd95_mm(pred,np.zeros_like(gt),(2.,1.)) is None
    print('PASS: Mamba backend; three output shapes; finite logits; spacing and empty-mask metrics; parameters=',sum(p.numel() for p in m.parameters()))
    
    pred[2:5,2:4]=True;gt[3:7,4:7]=True
    assert abs(compute_asd_mm(pred,gt,(1.7,1.7))-compute_asd(pred,gt)*1.7)<1e-10
    assert abs(compute_hd95_mm(pred,gt,(1.7,1.7))-compute_hd95(pred,gt)*1.7)<1e-10
    print("PASS: physical metrics match archived isotropic scaling on unequal shapes.")

if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    elif "--physical-eval" in sys.argv:
        sys.argv.remove("--physical-eval")
        physical_evaluate_main()
    else:
        train_main()
