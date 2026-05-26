import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HISTORY_K        = 100    # nombre de pièces précédentes dans l'historique
HISTORY_DIM      = 3      # [mean_cut_deviation, tau_cut_rms, q_error_cut_rms]
PIECE_COUNT_NORM = 1000.0
CADENCE_NORM     = 60.0
TEMP_NORM        = 80.0   # °C — écart max attendu au-dessus de T_AMBIENT (~100°C à cadence max)
DEVIATION_NORM   = 0.02   # m  — seuil de défaut
TAU_RMS_NORM     = 50.0   # Nm — effort moteur typique en découpe
Q_ERROR_RMS_NORM = 0.01   # rad — erreur de suivi typique


# ── History Encoder ────────────────────────────────────────────────────────────
class HistoryEncoder(nn.Module):
    """GRU sur K pas d'historique (normalisés) → embedding 64D.
    Chaque pas : [mean_cut_deviation, tau_cut_rms, q_error_cut_rms]."""

    def __init__(self, K: int = HISTORY_K, hidden: int = 64,
                 input_dim: int = HISTORY_DIM):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history : (B, K, HISTORY_DIM) — signaux normalisés
        _, h = self.gru(history)
        return h.squeeze(0)   # (B, 64)


# ── Context Encoder ────────────────────────────────────────────────────────────
class ContextEncoder(nn.Module):
    """(piece_count_norm, cadence_norm, temperature_norm) → 64D."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, out_dim)
        )

    def forward(self, piece_count: torch.Tensor, cadence: torch.Tensor,
                temperature: torch.Tensor) -> torch.Tensor:
        # piece_count, cadence, temperature : (B, 1) chacun, déjà normalisés
        return self.net(torch.cat([piece_count, cadence, temperature], dim=-1))  # (B, 64)


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


# ── Quality head ──────────────────────────────────────────────────────────────
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

        gru_input = shape_embed_dim + pe_dim
        self.gru  = nn.GRU(gru_input, h_dim, num_layers=gru_layers,
                           batch_first=True, dropout=dropout)

        self.h_init = nn.Sequential(
            nn.Linear(shape_embed_dim, h_dim * gru_layers),
            nn.Tanh(),
        )

        # Décodeur résiduel
        self.dec_in      = nn.Linear(h_dim, h_dim)
        self.dec_res     = nn.Sequential(
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
            ResBlock(h_dim, dropout),
        )
        self.dec_out     = nn.Linear(h_dim, obs_dim)

    def forward(
        self,
        shape_embed: torch.Tensor,
        T:           int,
        targets:     torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        shape_embed : (B, E)
        T           : longueur de séquence à générer
        targets     : ignoré (conservé pour compatibilité avec l'appelant)
        returns     : traj: B×T×obs_dim
        """
        B, E = shape_embed.shape

        h0  = self.h_init(shape_embed) \
                  .view(B, self.gru_layers, self.h_dim) \
                  .permute(1, 0, 2).contiguous()         # (L, B, h_dim)

        pe  = self.pe(T).unsqueeze(0).expand(B, T, -1)  # (B, T, pe_dim)

        ctx = shape_embed.unsqueeze(1).expand(B, T, E)  # (B, T, E)

        x       = torch.cat([ctx, pe], dim=-1).contiguous()
        h_raw, _ = self.gru(x, h0)
        traj    = self.dec_out(self.dec_res(self.dec_in(h_raw)))
        return traj


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
        # Projette [shape_embed(256) + hist(64) + ctx(64)] → 256
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
        """Fusionne les quatre sources d'information → embedding 256D."""
        shape_embed = self.encoder(waypoints, wp_lengths) + self.speed_proj(speed)

        # Normalisation des inputs : chaque canal de l'historique a sa propre échelle
        hist_norm = deviation_history.clone()                   # (B, K, 3)
        hist_norm[:, :, 0] = hist_norm[:, :, 0] / DEVIATION_NORM
        hist_norm[:, :, 1] = hist_norm[:, :, 1] / TAU_RMS_NORM
        hist_norm[:, :, 2] = hist_norm[:, :, 2] / Q_ERROR_RMS_NORM
        pc_norm   = piece_count / PIECE_COUNT_NORM              # (B, 1)
        cad_norm  = cadence     / CADENCE_NORM                  # (B, 1)
        temp_norm = temperature / TEMP_NORM                     # (B, 1)

        hist_embed = self.history_encoder(hist_norm)                       # (B, 64)
        ctx_embed  = self.context_encoder(pc_norm, cad_norm, temp_norm)   # (B, 64)

        combined = torch.cat([shape_embed, hist_embed, ctx_embed], dim=-1)
        return self.context_proj(combined)                      # (B, 256)

    def forward(self, waypoints, wp_lengths, speed,
                deviation_history=None, piece_count=None, cadence=None,
                temperature=None, targets=None, max_len=1300,
                p_teacher=1.0, q_init=None):
        """
        deviation_history : (B, K, 3) — historique des K pièces précédentes :
                            [:,:,0] = mean_cut_deviation (m)
                            [:,:,1] = tau_cut_rms (Nm)
                            [:,:,2] = q_error_cut_rms (rad)
                            Si None, remplacé par des zéros (machine à neuf / mode single).
        piece_count       : (B, 1) — nombre de pièces produites avant cette pièce.
        cadence           : (B, 1) — pièces/heure.
        temperature       : (B, 1) — température courante de la machine (°C).
        speed             : (B, 1) — duration_per_segment en secondes.
        """
        B = waypoints.shape[0]
        dev  = _default_zeros(deviation_history, (B, HISTORY_K, HISTORY_DIM), waypoints.device)
        pc   = _default_zeros(piece_count,       (B, 1),                      waypoints.device)
        cad  = _default_zeros(cadence,           (B, 1),                      waypoints.device)
        temp = _default_zeros(temperature,       (B, 1),                      waypoints.device)

        embed = self._encode(waypoints, wp_lengths, speed, dev, pc, cad, temp)
        T     = targets.shape[1] if targets is not None else max_len
        return self.decoder(embed, T, targets=targets)

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
        embed = self._encode(waypoints, wp_lengths, speed, dev, pc, cad, temp)
        return self.decoder(embed, max_len)


def _default_zeros(tensor, shape, device):
    """Retourne tensor s'il est fourni, sinon un tenseur de zéros."""
    if tensor is not None:
        return tensor
    return torch.zeros(shape, dtype=torch.float32, device=device)
