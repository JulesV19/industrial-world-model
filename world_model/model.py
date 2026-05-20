import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shape Encoder ──────────────────────────────────────────────────────────────
class ShapeEncoder(nn.Module):
    """
    Transformer encoder over variable-length waypoints → fixed shape embedding.
    Pre-LN Transformer for training stability.
    """

    def __init__(self, wp_dim: int = 3, embed_dim: int = 256,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim  = embed_dim
        self.input_proj = nn.Linear(wp_dim, embed_dim)
        self.pos_emb    = nn.Embedding(64, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_proj   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, waypoints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, N, _ = waypoints.shape
        pos  = torch.arange(N, device=waypoints.device).unsqueeze(0)
        x    = self.input_proj(waypoints) + self.pos_emb(pos)
        mask = torch.arange(N, device=waypoints.device)[None, :] >= lengths[:, None]
        x    = self.transformer(x, src_key_padding_mask=mask)
        valid  = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.pool_proj(pooled)                              # (B, embed_dim)


# ── Sinusoidal positional encoding ────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    """
    Donne au GRU un signal temporel qui varie à chaque pas.
    Sans ça, le GRU reçoit la même entrée à chaque step → converge
    vers un point fixe → sorties constantes ou aléatoires.
    """

    def __init__(self, dim: int, max_len: int = 4000):
        super().__init__()
        pe       = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div      = torch.exp(torch.arange(0, dim, 2).float()
                             * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)                             # (max_len, dim)

    def forward(self, T: int) -> torch.Tensor:
        return self.pe[:T]                                         # (T, dim)


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


# ── Temporal decoder ───────────────────────────────────────────────────────────
class TemporalDecoder(nn.Module):
    """
    Décodeur déterministe : shape_embed + encodage positionnel → trajectoire.

    À chaque step t, le GRU reçoit :
        [shape_embed, pos_enc(t)]
    ce qui lui donne un signal temporel qui varie → h_t évolue sur toute
    la durée de la séquence et encode "quelle phase de la trajectoire".

    On supprime le composant stochastique z : sur ce système la trajectoire
    est >95 % déterministe depuis la forme, le RSSM stochastique n'apportait
    que du bruit (KL collapsait au plancher free_bits).
    """

    def __init__(
        self,
        shape_embed_dim: int = 256,
        h_dim:           int = 512,
        obs_dim:         int = 15,
        pe_dim:          int = 64,
        gru_layers:      int = 3,
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.h_dim      = h_dim
        self.gru_layers = gru_layers
        self.obs_dim    = obs_dim

        self.pe = SinusoidalPE(pe_dim, max_len=20000)

        gru_input = shape_embed_dim + pe_dim + obs_dim   # +obs_dim : q_prev autorégressif
        self.gru  = nn.GRU(gru_input, h_dim, num_layers=gru_layers,
                           batch_first=True, dropout=dropout)

        self.h_init = nn.Sequential(
            nn.Linear(shape_embed_dim, h_dim * gru_layers),
            nn.Tanh(),
        )

        # Décodeur résiduel
        self.dec_in  = nn.Linear(h_dim, h_dim)
        self.dec_res = nn.Sequential(
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
        )
        self.dec_out = nn.Linear(h_dim, obs_dim)

    def forward(
        self,
        shape_embed: torch.Tensor,
        T:           int,
        targets:     torch.Tensor | None = None,
        p_teacher:   float               = 1.0,
        q_init:      torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        shape_embed : (B, E)
        T           : longueur de séquence à générer
        targets     : (B, T, obs_dim) — ground truth pour teacher forcing
        p_teacher   : probabilité d'utiliser le ground truth à chaque step (1.0 = pur TF)
        q_init      : (B, obs_dim) ou (obs_dim,) — état initial normalisé
        returns     : (B, T, obs_dim)
        """
        B, E   = shape_embed.shape
        device = shape_embed.device
        dtype  = shape_embed.dtype

        h0 = self.h_init(shape_embed) \
                 .view(B, self.gru_layers, self.h_dim) \
                 .permute(1, 0, 2).contiguous()          # (L, B, h_dim)

        pe  = self.pe(T).unsqueeze(0).expand(B, T, -1)  # (B, T, pe_dim)
        ctx = shape_embed.unsqueeze(1).expand(B, T, E)  # (B, T, E)

        # État initial : IK(HOME) normalisé (fourni par train.py), sinon zéros
        if q_init is None:
            q_prev = torch.zeros(B, self.obs_dim, device=device, dtype=dtype)
        else:
            q_prev = q_init.to(device=device, dtype=dtype)
            if q_prev.dim() == 1:
                q_prev = q_prev.unsqueeze(0).expand(B, -1)

        # ── Chemin rapide : teacher forcing pur → un seul appel GRU batché ──
        if p_teacher >= 1.0 and targets is not None:
            # Décaler les targets d'un pas : [q_init, t0, t1, …, t_{T-2}]
            q_shifted = torch.cat([q_prev.unsqueeze(1), targets[:, :-1, :]], dim=1)
            x = torch.cat([ctx, pe, q_shifted], dim=-1).contiguous()
            h_seq, _ = self.gru(x, h0)
            out = self.dec_in(h_seq)
            out = self.dec_res(out)
            return self.dec_out(out)                     # (B, T, obs_dim)

        # ── Chemin lent : scheduled sampling step-by-step ──
        outputs = []
        h = h0
        for t in range(T):
            x_t = torch.cat([ctx[:, t, :], pe[:, t, :], q_prev], dim=-1).unsqueeze(1)
            h_t, h = self.gru(x_t, h)
            pred_t = self.dec_out(self.dec_res(self.dec_in(h_t.squeeze(1))))
            outputs.append(pred_t)

            if targets is not None and t < T - 1:
                use_gt = (torch.rand(1, device=device).item() < p_teacher)
                q_prev = targets[:, t, :] if use_gt else pred_t.detach()
            else:
                q_prev = pred_t.detach()

        return torch.stack(outputs, dim=1)               # (B, T, obs_dim)


# ── WorldModel ─────────────────────────────────────────────────────────────────
class WorldModel(nn.Module):

    def __init__(
        self,
        shape_embed_dim: int   = 256,
        h_dim:           int   = 512,
        z_dim:           int   = 64,    # conservé pour compatibilité CLI, non utilisé
        obs_dim:         int   = 15,
        dropout:         float = 0.1,
        free_bits:       float = 0.5,   # idem
        gru_layers:      int   = 3,
        pe_dim:          int   = 64,
    ):
        super().__init__()
        self.encoder    = ShapeEncoder(embed_dim=shape_embed_dim, dropout=dropout)
        self.speed_proj = nn.Sequential(
            nn.Linear(1, shape_embed_dim),
            nn.GELU(),
            nn.Linear(shape_embed_dim, shape_embed_dim),
        )
        self.decoder = TemporalDecoder(
            shape_embed_dim=shape_embed_dim,
            h_dim=h_dim, obs_dim=obs_dim,
            pe_dim=pe_dim, gru_layers=gru_layers, dropout=dropout,
        )

    def forward(self, waypoints, wp_lengths, speed, targets=None, max_len=1300,
                p_teacher=1.0, q_init=None):
        """
        speed     : (B, 1) float — duration_per_segment en secondes
        p_teacher : probabilité teacher forcing (1.0 = pur TF, 0.0 = free-running)
        q_init    : (obs_dim,) ou (B, obs_dim) — état initial normalisé du décodeur
        """
        shape_embed = self.encoder(waypoints, wp_lengths)
        shape_embed = shape_embed + self.speed_proj(speed)
        T = targets.shape[1] if targets is not None else max_len
        preds = self.decoder(shape_embed, T, targets=targets,
                             p_teacher=p_teacher, q_init=q_init)
        return preds, torch.tensor(0.0, device=preds.device)

    @torch.no_grad()
    def predict(self, waypoints, wp_lengths, speed, max_len=1300, q_init=None):
        self.eval()
        shape_embed = self.encoder(waypoints, wp_lengths)
        shape_embed = shape_embed + self.speed_proj(speed)
        return self.decoder(shape_embed, max_len, p_teacher=0.0, q_init=q_init)
