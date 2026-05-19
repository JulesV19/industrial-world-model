import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shape Encoder ──────────────────────────────────────────────────────────────
class ShapeEncoder(nn.Module):
    """
    Transformer encoder over variable-length waypoints → fixed shape embedding.

    Waypoints: (x, y, is_cutting) — 3 dims, variable N_wp per piece.
    A Transformer captures global geometric structure better than a GRU
    (each waypoint can attend to all others regardless of order distance).
    """

    def __init__(self, wp_dim: int = 3, embed_dim: int = 256,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Project raw waypoints to transformer dimension
        self.input_proj = nn.Linear(wp_dim, embed_dim)

        # Learnable positional encoding (max 64 waypoints per piece)
        self.pos_emb = nn.Embedding(64, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool all tokens → single embedding
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, waypoints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        waypoints : (B, N_max, 3) — zero-padded
        lengths   : (B,)
        returns   : (B, embed_dim)
        """
        B, N, _ = waypoints.shape
        pos = torch.arange(N, device=waypoints.device).unsqueeze(0)  # (1, N)

        x = self.input_proj(waypoints) + self.pos_emb(pos)           # (B, N, D)

        # Key padding mask: True = position to IGNORE (padding)
        mask = torch.arange(N, device=waypoints.device)[None, :] >= lengths[:, None]

        x = self.transformer(x, src_key_padding_mask=mask)           # (B, N, D)

        # Masked mean pooling over valid tokens
        valid = (~mask).float().unsqueeze(-1)                         # (B, N, 1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)      # (B, D)

        return self.pool_proj(pooled)


# ── Residual block ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


# ── RSSM ───────────────────────────────────────────────────────────────────────
class RSSM(nn.Module):
    """
    Fully-vectorized RSSM with:
      - 3-layer GRU for the deterministic path
      - Free-bits KL to prevent posterior collapse
      - Residual decoder
    """

    def __init__(
        self,
        shape_embed_dim: int = 256,
        h_dim: int = 512,
        z_dim: int = 64,
        obs_dim: int = 15,
        dropout: float = 0.1,
        free_bits: float = 0.5,     # min KL per latent dim (prevents collapse)
        gru_layers: int = 3,
    ):
        super().__init__()
        self.h_dim     = h_dim
        self.z_dim     = z_dim
        self.free_bits = free_bits
        self.gru_layers = gru_layers

        # Temporal GRU — input is shape_embed broadcast over T
        self.gru = nn.GRU(shape_embed_dim, h_dim, num_layers=gru_layers,
                          batch_first=True, dropout=dropout)

        # Initialise all GRU layers from shape embedding
        self.h_init = nn.Sequential(
            nn.Linear(shape_embed_dim, h_dim * gru_layers),
            nn.Tanh(),
        )

        # Prior  p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 512), nn.ELU(),
            nn.Linear(512, z_dim * 2),
        )

        # Posterior  q(z_t | h_t, y_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, 512), nn.ELU(),
            nn.Linear(512, z_dim * 2),
        )

        # Residual decoder: (h_t, z_t) → obs_dim
        dec_dim = h_dim + z_dim
        self.decoder_in  = nn.Linear(dec_dim, h_dim)
        self.decoder_res = nn.Sequential(
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
        )
        self.decoder_out = nn.Linear(h_dim, obs_dim)

    # ------------------------------------------------------------------
    def _dist(self, raw: torch.Tensor):
        mu, log_s = raw.chunk(2, dim=-1)
        return mu, F.softplus(log_s) + 1e-4

    def _kl_free_bits(self, q_mu, q_sigma, p_mu, p_sigma) -> torch.Tensor:
        """
        KL(q || p) per latent dimension, clamped to free_bits minimum.
        Prevents posterior collapse by guaranteeing a minimum KL per dim.
        Averaged over (B, T, z_dim).
        """
        kl_per_dim = 0.5 * (
            (q_sigma / p_sigma).pow(2)
            + ((p_mu - q_mu) / p_sigma).pow(2)
            - 1
            + 2 * p_sigma.log()
            - 2 * q_sigma.log()
        )                                           # (B, T, z_dim)
        return kl_per_dim.clamp(min=self.free_bits).mean()

    def _decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_in(torch.cat([h, z], dim=-1))
        x = self.decoder_res(x)
        return self.decoder_out(x)

    # ------------------------------------------------------------------
    def forward(
        self,
        shape_embed: torch.Tensor,
        targets: torch.Tensor | None = None,
        max_len: int = 1300,
    ):
        B, E = shape_embed.shape
        T = targets.shape[1] if targets is not None else max_len

        # Initialise hidden state
        h0_flat = self.h_init(shape_embed)                          # (B, h_dim * L)
        h0 = h0_flat.view(B, self.gru_layers, self.h_dim) \
                     .permute(1, 0, 2).contiguous()                  # (L, B, h_dim)

        # Deterministic path — fully vectorized
        x = shape_embed.unsqueeze(1).expand(B, T, E).contiguous()
        h_seq, _ = self.gru(x, h0)                                   # (B, T, h_dim)

        # Prior
        p_mu, p_sigma = self._dist(self.prior_net(h_seq))

        if targets is not None:
            q_mu, q_sigma = self._dist(
                self.posterior_net(torch.cat([h_seq, targets], dim=-1))
            )
            z = q_mu + q_sigma * torch.randn_like(q_mu)
            kl_loss = self._kl_free_bits(q_mu, q_sigma, p_mu, p_sigma)
        else:
            z = p_mu + p_sigma * torch.randn_like(p_mu)
            kl_loss = torch.tensor(0.0, device=shape_embed.device)

        preds = self._decode(h_seq, z)                               # (B, T, obs_dim)
        return preds, kl_loss


# ── WorldModel ─────────────────────────────────────────────────────────────────
class WorldModel(nn.Module):
    """Top-level module: ShapeEncoder (Transformer) + RSSM."""

    def __init__(
        self,
        shape_embed_dim: int = 256,
        h_dim:           int = 512,
        z_dim:           int = 64,
        obs_dim:         int = 15,
        dropout:         float = 0.1,
        free_bits:       float = 0.5,
        gru_layers:      int  = 3,
    ):
        super().__init__()
        self.encoder = ShapeEncoder(embed_dim=shape_embed_dim, dropout=dropout)
        self.rssm    = RSSM(shape_embed_dim, h_dim, z_dim, obs_dim,
                            dropout, free_bits, gru_layers)

    def forward(self, waypoints, wp_lengths, targets=None, max_len=1300):
        shape_embed = self.encoder(waypoints, wp_lengths)
        return self.rssm(shape_embed, targets, max_len)

    @torch.no_grad()
    def predict(self, waypoints, wp_lengths, max_len=1300):
        self.eval()
        shape_embed = self.encoder(waypoints, wp_lengths)
        preds, _ = self.rssm(shape_embed, targets=None, max_len=max_len)
        return preds
