import torch
import torch.nn as nn


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


class DrillModel(nn.Module):
    """
    Prédit les offsets de perçage et les défauts pour les 4 coins.

    Entrée  : corner_targets (B, 4, 2) + speed (B, 1)
    Sortie  : offsets (B, 4, 2)  — drill_hit - corner_target, normalisé
              defect_logits (B, 4)

    Architecture : encodage global de la pièce (MLP sur les 4 coins + vitesse),
    puis pour chaque coin une tête qui combine contexte global + position du coin.
    Cela permet au modèle de capturer des biais spécifiques à chaque zone du
    workspace (certains coins sont plus difficiles à atteindre).
    """

    def __init__(
        self,
        corner_embed_dim: int   = 64,
        global_dim:       int   = 256,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.corner_embed_dim = corner_embed_dim
        self.global_dim       = global_dim

        # Encodage de chaque coin individuellement (partagé entre les 4)
        self.corner_enc = nn.Sequential(
            nn.Linear(2, corner_embed_dim),
            nn.GELU(),
            nn.Linear(corner_embed_dim, corner_embed_dim),
        )

        # Encodage de la vitesse
        self.speed_enc = nn.Sequential(
            nn.Linear(1, corner_embed_dim),
            nn.GELU(),
            nn.Linear(corner_embed_dim, corner_embed_dim),
        )

        # Contexte global : agrégation des 4 coins + vitesse
        self.global_trunk = nn.Sequential(
            nn.Linear(4 * corner_embed_dim + corner_embed_dim, global_dim),
            nn.GELU(),
            ResBlock(global_dim, dropout),
            ResBlock(global_dim, dropout),
            ResBlock(global_dim, dropout),
        )

        # Tête par coin : contexte global + embedding du coin → offset + logit défaut
        per_coin_in = global_dim + corner_embed_dim
        self.offset_head = nn.Sequential(
            nn.Linear(per_coin_in, global_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(global_dim // 2, 2),   # (dx, dy)
        )
        self.defect_head = nn.Sequential(
            nn.Linear(per_coin_in, global_dim // 4),
            nn.GELU(),
            nn.Linear(global_dim // 4, 1),   # logit
        )

    def forward(
        self,
        corners: torch.Tensor,   # (B, 4, 2)
        speed:   torch.Tensor,   # (B, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = corners.shape[0]

        # Encode chaque coin : (B, 4, corner_embed_dim)
        coin_emb = self.corner_enc(corners)

        # Encode la vitesse : (B, corner_embed_dim)
        spd_emb = self.speed_enc(speed)

        # Contexte global : flatten les 4 coins + vitesse
        flat = torch.cat([coin_emb.reshape(B, -1), spd_emb], dim=-1)
        ctx  = self.global_trunk(flat)   # (B, global_dim)

        # Prédiction par coin
        ctx_exp = ctx.unsqueeze(1).expand(B, 4, -1)          # (B, 4, global_dim)
        per_coin = torch.cat([ctx_exp, coin_emb], dim=-1)    # (B, 4, global_dim + corner_embed_dim)

        offsets       = self.offset_head(per_coin)            # (B, 4, 2)
        defect_logits = self.defect_head(per_coin).squeeze(-1)  # (B, 4)

        return offsets, defect_logits
