import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeEncoder(nn.Module):
    """
    Encode a variable-length waypoint sequence into a fixed-size shape embedding.
    Waypoints: (x, y, is_cutting) — 3 dims, variable N_wp per piece.
    """

    def __init__(self, wp_dim: int = 3, hidden: int = 128, embed_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(wp_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, waypoints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        waypoints : (B, N_max, 3) — padded
        lengths   : (B,)          — actual number of waypoints per sample
        returns   : (B, embed_dim)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            waypoints, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)          # h: (num_layers, B, hidden)
        return self.proj(h[-1])          # last layer → (B, embed_dim)


class RSSM(nn.Module):
    """
    Recurrent State Space Model — fully vectorized (no Python loop over T).

    Deterministic path:
        h_seq = GRU(shape_embed repeated T times)   → (B, T, h_dim)

    Stochastic path:
        prior     : p(z_t | h_t)
        posterior : q(z_t | h_t, y_t)   [training only]
        z sampled via reparameterization

    Decoder:
        ŷ_t = MLP(h_t, z_t)             → (B, T, obs_dim)
    """

    def __init__(
        self,
        shape_embed_dim: int = 128,
        h_dim: int = 256,
        z_dim: int = 32,
        obs_dim: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        # Temporal GRU — input is shape_embed broadcast over T
        self.gru = nn.GRU(shape_embed_dim, h_dim, num_layers=2,
                          batch_first=True, dropout=dropout)

        # Initialise hidden state from shape embedding
        self.h_init = nn.Sequential(
            nn.Linear(shape_embed_dim, h_dim * 2),  # *2 for num_layers
            nn.Tanh(),
        )

        # Prior  p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, 256), nn.ELU(),
            nn.Linear(256, z_dim * 2),
        )

        # Posterior  q(z_t | h_t, y_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(h_dim + obs_dim, 256), nn.ELU(),
            nn.Linear(256, z_dim * 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, 512), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, obs_dim),
        )

    # ------------------------------------------------------------------
    def _dist(self, raw: torch.Tensor):
        """Split raw output into (mu, sigma) with softplus for sigma."""
        mu, log_s = raw.chunk(2, dim=-1)
        return mu, F.softplus(log_s) + 1e-4

    def _kl(self, q_mu, q_sigma, p_mu, p_sigma) -> torch.Tensor:
        """KL(q || p) summed over z_dim, averaged over (B, T)."""
        kl = (
            (q_sigma / p_sigma).pow(2)
            + ((p_mu - q_mu) / p_sigma).pow(2)
            - 1
            + 2 * p_sigma.log()
            - 2 * q_sigma.log()
        ).sum(-1)                               # (B, T)
        return 0.5 * kl.mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        shape_embed: torch.Tensor,
        targets: torch.Tensor | None = None,
        max_len: int = 1300,
    ):
        """
        shape_embed : (B, E)
        targets     : (B, T, obs_dim) or None  — normalized observations
        max_len     : used only when targets is None

        Returns
        -------
        preds    : (B, T, obs_dim)   — raw logits (apply sigmoid to last dim for is_cutting)
        kl_loss  : scalar
        """
        B, E = shape_embed.shape
        T = targets.shape[1] if targets is not None else max_len

        # ---------- deterministic path (fully vectorized) ----------
        h0_flat = self.h_init(shape_embed)           # (B, h_dim*2)
        # reshape to (num_layers, B, h_dim)
        h0 = h0_flat.view(B, 2, self.h_dim).permute(1, 0, 2).contiguous()

        x = shape_embed.unsqueeze(1).expand(B, T, E).contiguous()   # (B, T, E)
        h_seq, _ = self.gru(x, h0)                                   # (B, T, h_dim)

        # ---------- stochastic path ----------
        prior_raw = self.prior_net(h_seq)                # (B, T, z_dim*2)
        p_mu, p_sigma = self._dist(prior_raw)

        if targets is not None:
            post_raw = self.posterior_net(torch.cat([h_seq, targets], dim=-1))
            q_mu, q_sigma = self._dist(post_raw)
            eps = torch.randn_like(q_mu)
            z = q_mu + q_sigma * eps                     # (B, T, z_dim)
            kl_loss = self._kl(q_mu, q_sigma, p_mu, p_sigma)
        else:
            eps = torch.randn_like(p_mu)
            z = p_mu + p_sigma * eps
            kl_loss = torch.tensor(0.0, device=shape_embed.device)

        # ---------- decode ----------
        preds = self.decoder(torch.cat([h_seq, z], dim=-1))   # (B, T, obs_dim)
        return preds, kl_loss


class WorldModel(nn.Module):
    """Top-level module: ShapeEncoder + RSSM."""

    def __init__(
        self,
        shape_embed_dim: int = 128,
        h_dim: int = 256,
        z_dim: int = 32,
        obs_dim: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ShapeEncoder(embed_dim=shape_embed_dim)
        self.rssm = RSSM(shape_embed_dim, h_dim, z_dim, obs_dim, dropout)

    def forward(
        self,
        waypoints: torch.Tensor,
        wp_lengths: torch.Tensor,
        targets: torch.Tensor | None = None,
        max_len: int = 1300,
    ):
        shape_embed = self.encoder(waypoints, wp_lengths)
        return self.rssm(shape_embed, targets, max_len)

    @torch.no_grad()
    def predict(
        self,
        waypoints: torch.Tensor,
        wp_lengths: torch.Tensor,
        max_len: int = 1300,
    ) -> torch.Tensor:
        """Pure prior rollout — no observations needed. Returns (B, T, obs_dim)."""
        self.eval()
        shape_embed = self.encoder(waypoints, wp_lengths)
        preds, _ = self.rssm(shape_embed, targets=None, max_len=max_len)
        return preds
