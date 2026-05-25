import math
import torch
import torch.nn as nn

HISTORY_K        = 100
PIECE_COUNT_NORM = 1000.0
CADENCE_NORM     = 60.0
DEVIATION_NORM   = 0.02


# ── History Encoder ────────────────────────────────────────────────────────────
class HistoryEncoder(nn.Module):
    """GRU sur K erreurs de perçage passées (normalisées) → embedding 64D."""

    def __init__(self, K: int = HISTORY_K, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(history)
        return h.squeeze(0)   # (B, 64)


# ── Context Encoder ────────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    """(piece_count_norm, cadence_norm) → 64D."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(), nn.Linear(64, out_dim)
        )

    def forward(self, piece_count: torch.Tensor, cadence: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([piece_count, cadence], dim=-1))  # (B, 64)


def _default_zeros(tensor, shape, device):
    if tensor is not None:
        return tensor
    return torch.zeros(shape, dtype=torch.float32, device=device)


class SinusoidalPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 4000):
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


class DrillWorldModel(nn.Module):
    """
    Prédit la trajectoire articulaire q_real du bras et, par dérivation,
    les offsets de perçage et les défauts pour les 4 coins.

    Entrée  : corner_targets (B, 4, 2) + speed (B, 1)
    Sortie  :
      traj          (B, T, 2)  — trajectoire q_real normalisée
      offsets       (B, 4, 2)  — drill_hit - corner_target, normalisé
      defect_logits (B, 4)     — logits de défaut

    Architecture :
      1. Encodage MLP coins + vitesse → embedding global (B, embed_dim)
      2. GRU + PE sinusoïdale → hidden states h (B, T, h_dim) → q_real
      3. Cross-attention : 4 queries apprenables sur h → features par coin
      4. Têtes offset et défaut depuis les features par coin
    """

    def __init__(
        self,
        corner_embed_dim: int   = 64,
        embed_dim:        int   = 256,
        h_dim:            int   = 512,
        pe_dim:           int   = 64,
        gru_layers:       int   = 2,
        n_attn_heads:     int   = 4,
        dropout:          float = 0.1,
        history_hidden:   int   = 64,
        context_dim:      int   = 64,
    ):
        super().__init__()
        self.h_dim      = h_dim
        self.gru_layers = gru_layers

        self.corner_enc = nn.Sequential(
            nn.Linear(2, corner_embed_dim),
            nn.GELU(),
            nn.Linear(corner_embed_dim, corner_embed_dim),
        )
        self.speed_enc = nn.Sequential(
            nn.Linear(1, corner_embed_dim),
            nn.GELU(),
            nn.Linear(corner_embed_dim, corner_embed_dim),
        )
        self.history_encoder = HistoryEncoder(K=HISTORY_K, hidden=history_hidden)
        self.context_encoder = ContextEncoder(out_dim=context_dim)
        # Trunk élargi : 4×coins + speed + hist + ctx
        trunk_in = 4 * corner_embed_dim + corner_embed_dim + history_hidden + context_dim
        self.global_trunk = nn.Sequential(
            nn.Linear(trunk_in, embed_dim),
            nn.GELU(),
            ResBlock(embed_dim, dropout),
            ResBlock(embed_dim, dropout),
        )

        self.pe  = SinusoidalPE(pe_dim, max_len=4000)
        self.gru = nn.GRU(
            embed_dim + pe_dim, h_dim,
            num_layers=gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.h_init = nn.Sequential(
            nn.Linear(embed_dim, h_dim * gru_layers),
            nn.Tanh(),
        )
        self.traj_head = nn.Sequential(
            ResBlock(h_dim, dropout),
            nn.Linear(h_dim, 2),
        )

        # 4 queries apprenables pour lire les hidden states du GRU
        self.corner_queries = nn.Embedding(4, h_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h_dim, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )

        per_coin_in = h_dim + corner_embed_dim
        self.offset_head = nn.Sequential(
            nn.Linear(per_coin_in, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )
        self.defect_head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

    def forward(
        self,
        corners:           torch.Tensor,               # (B, 4, 2)
        speed:             torch.Tensor,               # (B, 1)
        T:                 int | None        = None,
        lengths:           torch.Tensor | None = None, # (B,) nombre de pas valides
        deviation_history: torch.Tensor | None = None, # (B, K, 1) erreurs passées
        piece_count:       torch.Tensor | None = None, # (B, 1)
        cadence:           torch.Tensor | None = None, # (B, 1)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = corners.shape[0]

        dev = _default_zeros(deviation_history, (B, HISTORY_K, 1), corners.device)
        pc  = _default_zeros(piece_count,       (B, 1),            corners.device)
        cad = _default_zeros(cadence,           (B, 1),            corners.device)

        coin_emb   = self.corner_enc(corners)                   # (B, 4, corner_embed_dim)
        spd_emb    = self.speed_enc(speed)                      # (B, corner_embed_dim)
        hist_emb   = self.history_encoder(dev / DEVIATION_NORM) # (B, 64)
        ctx_emb    = self.context_encoder(pc / PIECE_COUNT_NORM,
                                          cad / CADENCE_NORM)  # (B, 64)
        ctx = self.global_trunk(
            torch.cat([coin_emb.reshape(B, -1), spd_emb,
                       hist_emb, ctx_emb], dim=-1)
        )                                                       # (B, embed_dim)

        if T is None:
            T = 512
        h0  = (self.h_init(ctx)
                   .view(B, self.gru_layers, self.h_dim)
                   .permute(1, 0, 2).contiguous())          # (L, B, h_dim)
        pe  = self.pe(T).unsqueeze(0).expand(B, T, -1)
        inp = torch.cat(
            [ctx.unsqueeze(1).expand(B, T, -1), pe], dim=-1
        ).contiguous()                                      # (B, T, embed_dim+pe_dim)
        h_raw, _ = self.gru(inp, h0)                       # (B, T, h_dim)

        traj = self.traj_head(h_raw)                       # (B, T, 2)

        # Masque les positions paddées dans la cross-attention
        key_mask = None
        if lengths is not None:
            idx      = torch.arange(T, device=corners.device).unsqueeze(0)
            key_mask = idx >= lengths.unsqueeze(1)         # (B, T)  True = ignoré

        queries = (self.corner_queries.weight
                       .unsqueeze(0).expand(B, -1, -1))    # (B, 4, h_dim)
        coin_feat, _ = self.cross_attn(
            queries, h_raw, h_raw, key_padding_mask=key_mask
        )                                                   # (B, 4, h_dim)

        per_coin      = torch.cat([coin_feat, coin_emb], dim=-1)
        offsets       = self.offset_head(per_coin)          # (B, 4, 2)
        defect_logits = self.defect_head(coin_feat).squeeze(-1)  # (B, 4)

        return traj, offsets, defect_logits
