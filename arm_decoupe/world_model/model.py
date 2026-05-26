import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HISTORY_K        = 100
HISTORY_DIM      = 3
PIECE_COUNT_NORM = 1000.0
CADENCE_NORM     = 60.0
TEMP_NORM        = 80.0
DEVIATION_NORM   = 0.02
TAU_RMS_NORM     = 50.0
Q_ERROR_RMS_NORM = 0.01


# ── History Encoder ────────────────────────────────────────────────────────────
class HistoryEncoder(nn.Module):
    def __init__(self, K: int = HISTORY_K, hidden: int = 64,
                 input_dim: int = HISTORY_DIM):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(history)
        return h.squeeze(0)


# ── Context Encoder ────────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, out_dim)
        )

    def forward(self, piece_count, cadence, temperature):
        return self.net(torch.cat([piece_count, cadence, temperature], dim=-1))


# ── Shape Encoder ──────────────────────────────────────────────────────────────
class ShapeEncoder(nn.Module):
    """
    Transformer encoder sur les waypoints.
    Retourne (pooled, wp_embeds, wp_mask) :
      pooled    : (B, embed_dim) — embedding global pour le contexte
      wp_embeds : (B, N_wp, embed_dim) — embeddings par waypoint pour la cross-attention
      wp_mask   : (B, N_wp) bool — True = padding à ignorer
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

    def forward(self, waypoints: torch.Tensor, lengths: torch.Tensor):
        B, N, _ = waypoints.shape
        pos  = torch.arange(N, device=waypoints.device).unsqueeze(0)
        x    = self.input_proj(waypoints) + self.pos_emb(pos)
        mask = torch.arange(N, device=waypoints.device)[None, :] >= lengths[:, None]
        x    = self.transformer(x, src_key_padding_mask=mask)
        valid  = (~mask).float().unsqueeze(-1)
        pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.pool_proj(pooled), x, mask   # pooled, wp_embeds, wp_mask


# ── Sinusoidal positional encoding ────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 20000):
        super().__init__()
        pe       = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div      = torch.exp(torch.arange(0, dim, 2).float()
                             * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, T: int) -> torch.Tensor:
        return self.pe[:T]


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


# ── Temporal decoder (cross-attention) ────────────────────────────────────────
class TemporalDecoder(nn.Module):
    """
    Décodeur par cross-attention : PE(t) attend les waypoints → trajectoire.

    À chaque step t :
        query    = proj(pe(t)) + context_embed   # "à quel moment, dans quel contexte ?"
        attended = CrossAttention(query, wp_embeds)  # "quels waypoints sont actifs ?"
        output   = MLP(attended)

    Avantages vs GRU :
    - Pas d'attracteur fixe → impossible de coller à la position initiale
    - Le lien PE(t) ↔ waypoints est explicite et appris directement
    - Parallélisable sur T (pas de recurrence)
    """

    def __init__(
        self,
        shape_embed_dim: int = 256,
        h_dim:           int = 512,
        obs_dim:         int = 2,
        pe_dim:          int = 64,
        gru_layers:      int = 3,   # réutilisé comme n_layers cross-attention
        dropout:         float = 0.1,
    ):
        super().__init__()
        self.pe     = SinusoidalPE(pe_dim, max_len=20000)
        n_layers    = max(1, gru_layers - 1)
        n_heads     = 4

        # Projette PE → espace des waypoints (shape_embed_dim)
        self.pe_proj = nn.Sequential(
            nn.Linear(pe_dim, shape_embed_dim),
            nn.LayerNorm(shape_embed_dim),
        )

        # Cross-attention : queries = PE(t) + context, keys/values = waypoints
        layer = nn.TransformerDecoderLayer(
            d_model=shape_embed_dim, nhead=n_heads,
            dim_feedforward=shape_embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.cross_attn = nn.TransformerDecoder(layer, num_layers=n_layers)

        # MLP de décodage
        self.dec_in  = nn.Linear(shape_embed_dim, h_dim)
        self.dec_res = nn.Sequential(
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
        )
        self.dec_out = nn.Linear(h_dim, obs_dim)

    def forward(self, wp_embeds: torch.Tensor, context_embed: torch.Tensor,
                T: int, wp_mask: torch.Tensor | None = None,
                targets: torch.Tensor | None = None) -> torch.Tensor:
        """
        wp_embeds    : (B, N_wp, E) — embeddings waypoints individuels
        context_embed: (B, E)       — contexte global (vitesse + historique + état machine)
        T            : longueur de séquence à générer
        wp_mask      : (B, N_wp) bool — True = padding
        """
        B = wp_embeds.shape[0]

        # Queries : PE(t) projeté + contexte global
        pe      = self.pe(T).unsqueeze(0).expand(B, T, -1)   # (B, T, pe_dim)
        queries = self.pe_proj(pe) + context_embed.unsqueeze(1)  # (B, T, E)

        # Cross-attention : chaque step t attend les waypoints
        attended = self.cross_attn(
            tgt=queries,
            memory=wp_embeds,
            memory_key_padding_mask=wp_mask,
        )  # (B, T, E)

        return self.dec_out(self.dec_res(self.dec_in(attended)))


# ── WorldModel ─────────────────────────────────────────────────────────────────
class WorldModel(nn.Module):

    def __init__(
        self,
        shape_embed_dim: int   = 256,
        h_dim:           int   = 512,
        z_dim:           int   = 64,
        obs_dim:         int   = 2,
        dropout:         float = 0.1,
        free_bits:       float = 0.5,
        gru_layers:      int   = 3,
        pe_dim:          int   = 64,
        history_hidden:  int   = 64,
        context_dim:     int   = 64,
    ):
        super().__init__()
        self.encoder         = ShapeEncoder(embed_dim=shape_embed_dim, dropout=dropout)
        self.speed_proj      = nn.Sequential(
            nn.Linear(1, shape_embed_dim),
            nn.GELU(),
            nn.Linear(shape_embed_dim, shape_embed_dim),
        )
        self.history_encoder = HistoryEncoder(K=HISTORY_K, hidden=history_hidden)
        self.context_encoder = ContextEncoder(out_dim=context_dim)
        self.context_proj    = nn.Sequential(
            nn.Linear(shape_embed_dim + history_hidden + context_dim, shape_embed_dim),
            nn.GELU(),
        )
        self.decoder = TemporalDecoder(
            shape_embed_dim=shape_embed_dim,
            h_dim=h_dim, obs_dim=obs_dim,
            pe_dim=pe_dim, gru_layers=gru_layers, dropout=dropout,
        )

    def _encode(self, waypoints, wp_lengths, speed, deviation_history,
                piece_count, cadence, temperature):
        """→ (context_embed (B,256), wp_embeds (B,N,256), wp_mask (B,N))"""
        pooled, wp_embeds, wp_mask = self.encoder(waypoints, wp_lengths)
        shape_embed = pooled + self.speed_proj(speed)

        hist_norm = deviation_history.clone()
        hist_norm[:, :, 0] /= DEVIATION_NORM
        hist_norm[:, :, 1] /= TAU_RMS_NORM
        hist_norm[:, :, 2] /= Q_ERROR_RMS_NORM
        pc_norm   = piece_count / PIECE_COUNT_NORM
        cad_norm  = cadence     / CADENCE_NORM
        temp_norm = temperature / TEMP_NORM

        hist_embed = self.history_encoder(hist_norm)
        ctx_embed  = self.context_encoder(pc_norm, cad_norm, temp_norm)

        combined      = torch.cat([shape_embed, hist_embed, ctx_embed], dim=-1)
        context_embed = self.context_proj(combined)
        return context_embed, wp_embeds, wp_mask

    def forward(self, waypoints, wp_lengths, speed,
                deviation_history=None, piece_count=None, cadence=None,
                temperature=None, targets=None, max_len=1300,
                p_teacher=1.0, q_init=None):
        B = waypoints.shape[0]
        dev  = _default_zeros(deviation_history, (B, HISTORY_K, HISTORY_DIM), waypoints.device)
        pc   = _default_zeros(piece_count,       (B, 1),                      waypoints.device)
        cad  = _default_zeros(cadence,           (B, 1),                      waypoints.device)
        temp = _default_zeros(temperature,       (B, 1),                      waypoints.device)

        context_embed, wp_embeds, wp_mask = self._encode(
            waypoints, wp_lengths, speed, dev, pc, cad, temp)
        T = targets.shape[1] if targets is not None else max_len
        return self.decoder(wp_embeds, context_embed, T, wp_mask)

    @torch.no_grad()
    def predict(self, waypoints, wp_lengths, speed,
                deviation_history=None, piece_count=None, cadence=None,
                temperature=None, max_len=1300, q_init=None):
        self.eval()
        B = waypoints.shape[0]
        dev  = _default_zeros(deviation_history, (B, HISTORY_K, HISTORY_DIM), waypoints.device)
        pc   = _default_zeros(piece_count,       (B, 1),                      waypoints.device)
        cad  = _default_zeros(cadence,           (B, 1),                      waypoints.device)
        temp = _default_zeros(temperature,       (B, 1),                      waypoints.device)
        context_embed, wp_embeds, wp_mask = self._encode(
            waypoints, wp_lengths, speed, dev, pc, cad, temp)
        return self.decoder(wp_embeds, context_embed, max_len, wp_mask)


def _default_zeros(tensor, shape, device):
    if tensor is not None:
        return tensor
    return torch.zeros(shape, dtype=torch.float32, device=device)
