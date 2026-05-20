# World Model — Documentation détaillée

## 1. Contexte et objectif

### Le système physique

On simule un bras robotique à **2 degrés de liberté** (2-DOF) qui découpe des pièces selon des trajectoires planifiées. Le bras est décrit par deux angles articulaires `q = (q1, q2)` (en radians). À chaque instant, le contrôleur calcule un couple moteur pour suivre une trajectoire désirée `q_des(t)`, mais le bras réel s'en écarte à cause des frottements, du bruit moteur, de la zone morte, etc.

### Le rôle du world model

Le world model est un réseau de neurones qui **prédit la trajectoire réelle complète** d'une pièce à partir de sa description géométrique et de la vitesse d'exécution, **sans simuler** la physique étape par étape. Concrètement :

```
Entrée  : forme de la pièce (waypoints) + vitesse de coupe
Sortie  : q_real(t) pour t = 0, 1, …, T   (≈ 1300 pas à 100 Hz)
```

L'intérêt est d'obtenir cette prédiction **en une seule passe**, bien plus rapide que la simulation physique complète.

---

## 2. Vue d'ensemble de l'architecture

```
Waypoints (N×3) ──► ShapeEncoder (Transformer) ──► shape_embed (256d)
                                                          │
Vitesse (1d) ──────► speed_proj (MLP) ─────────────────► + ──► contexte (256d)
                                                          │
                                                          ▼
                                               TemporalDecoder (GRU)
                                                          │
                                                          ▼
                                               q̂(0), q̂(1), …, q̂(T)
```

Le modèle se décompose en trois blocs : un **encodeur de forme**, une **projection de vitesse**, et un **décodeur temporel** autorégressif.

---

## 3. Encodeur de forme — `ShapeEncoder`

### Entrée

Les waypoints sont les points-clés de la trajectoire : position `(x, y)` et un booléen `is_cutting` (laser actif ou non). Chaque waypoint est donc un vecteur de dimension 3. Le nombre de waypoints `N` varie selon la pièce.

### Architecture : Transformer encoder

```
waypoints (B, N, 3)
       │
  Linear(3 → 256)  +  Embedding de position (appris, jusqu'à 64 positions)
       │
  ┌────┴────────────────────────┐
  │  TransformerEncoderLayer ×3 │  (Pre-LN, 4 têtes, FFN dim 1024, dropout 0.1)
  └────────────────────────────┘
       │
  Mean pooling (masqué : on ignore les positions paddées)
       │
  Linear(256 → 256) + LayerNorm
       │
  shape_embed (B, 256)
```

**Pre-LN** (LayerNorm avant l'attention, pas après) : rend l'entraînement plus stable qu'avec le Transformer original de Vaswani et al.

**Mean pooling masqué** : comme les séquences ont des longueurs variables, on pad jusqu'à la longueur maximale du batch. Le masque empêche les positions paddées de contribuer à la moyenne. Résultat : un vecteur fixe de 256 dimensions qui résume toute la géométrie de la pièce.

### Pourquoi un Transformer et pas un MLP ?

Les waypoints sont une **séquence non ordonnée de points géométriques**. Le Transformer peut apprendre des relations entre waypoints non-adjacents (ex. "ce coin est proche de ce bord"), ce qu'un MLP appliqué à une concaténation ne peut pas faire proprement avec des longueurs variables.

---

## 4. Projection de vitesse — `speed_proj`

```
vitesse (B, 1)
       │
  Linear(1 → 256) → GELU → Linear(256 → 256)
       │
  speed_embed (B, 256)
```

La vitesse de coupe (durée par segment, en secondes) est projetée dans le même espace que `shape_embed`, puis **additionnée** :

```
contexte = shape_embed + speed_embed
```

Ce vecteur de 256 dimensions condense toute l'information statique disponible : la forme de la pièce ET la cadence d'exécution.

---

## 5. Décodeur temporel — `TemporalDecoder`

C'est le cœur du modèle. Il génère la séquence `q̂(0), q̂(1), …, q̂(T-1)` en se basant sur le contexte et sur ses propres prédictions précédentes.

### 5.1 Encodage positionnel sinusoïdal

```python
pe(t, 2i)   = sin(t / 10000^(2i/d))
pe(t, 2i+1) = cos(t / 10000^(2i/d))
```

Un vecteur de 64 dimensions qui encode le **temps absolu** dans la trajectoire. Sans lui, le GRU recevrait exactement la même entrée à chaque pas et produirait une sortie constante.

### 5.2 Initialisation de l'état caché

```
contexte (B, 256)
       │
  Linear(256 → 512 × 3) + Tanh
       │
  reshape → h0 (3, B, 512)    ← état initial des 3 couches du GRU
```

L'état caché initial n'est pas zéro : il est calculé depuis le contexte. Cela permet au GRU de "savoir" dès le premier pas quelle pièce il génère.

### 5.3 Le GRU autorégressif

À chaque pas `t`, le GRU reçoit une entrée assemblée :

```
x(t) = [ contexte (256d) | pe(t) (64d) | q̂(t-1) (2d) ]
                                                  ↑
                                        prédiction précédente
```

Dimension d'entrée totale : 256 + 64 + 2 = **322**. L'état caché a une dimension de **512**, sur **3 couches**.

La **cellule GRU** (Gated Recurrent Unit) maintient une mémoire interne `h(t)` qui évolue à chaque pas :

```
z(t) = σ(W_z · [h(t-1), x(t)])          ← porte de mise à jour
r(t) = σ(W_r · [h(t-1), x(t)])          ← porte de remise à zéro
h̃(t) = tanh(W · [r(t)⊙h(t-1), x(t)])  ← candidat état caché
h(t) = (1 - z(t)) ⊙ h(t-1) + z(t) ⊙ h̃(t)
```

La porte `z` décide combien garder de l'ancien état vs. intégrer le nouveau. La porte `r` contrôle l'influence du passé sur le candidat. Contrairement au LSTM, le GRU n'a pas de cellule séparée — il est plus léger pour des résultats comparables.

### 5.4 Décodeur résiduel

```
h(t) (B, 512)
       │
  Linear(512 → 512)
       │
  ResBlock × 3
       │
  Linear(512 → 2)
       │
  q̂(t) (B, 2)
```

Chaque **ResBlock** est :
```
x → LayerNorm → Linear(512→1024) → GELU → Dropout → Linear(1024→512) → + x
```

La connexion résiduelle (`+ x`) permet au gradient de circuler directement sans passer par toutes les non-linéarités, ce qui stabilise l'entraînement en profondeur.

### 5.5 L'autorégression : pourquoi c'est important

**Avant** : le décodeur ne recevait pas `q̂(t-1)`. Il ne pouvait apprendre que le comportement **moyen** de la trajectoire — la dynamique d'accumulation des erreurs était invisible pour lui.

**Maintenant** : en recevant sa propre prédiction précédente, le modèle peut apprendre comment les erreurs **se propagent dans le temps**. Par exemple : si à `t-1` il a prédit que le bras était légèrement en retard, il peut anticiper que le contrôleur va sur-corriger à `t`.

---

## 6. Entraînement

### 6.1 Normalisation des données

Toutes les cibles (`q_real`, etc.) sont **z-scorées** signal par signal :

```
x_norm = (x - μ) / (σ + ε)
```

Les statistiques `μ` et `σ` sont calculées sur tout le dataset d'entraînement et sauvegardées dans `normalizer.npz`. Le modèle apprend et prédit dans cet espace normalisé. La loss est calculée en espace normalisé.

### 6.2 Loss pondérée

La loss de base est la **MSE masquée** (on ignore les pas paddés) :

```
L = Σ_{b,t,d} [ (ŷ_btd - y_btd)² × mask_bt × w_b ] / Σ mask
```

Le poids `w_b` de chaque séquence dans le batch est proportionnel au **RMS de ses targets** :

```
w_b = RMS(y_b)^γ / mean(RMS(y)^γ)
```

`γ = error_weight_gamma` (défaut 1.0). Les séquences avec de grandes erreurs (rares dans le dataset) contribuent plus à la loss, compensant le déséquilibre.

### 6.3 Teacher forcing et scheduled sampling

C'est la partie la plus délicate de l'entraînement autorégressif.

**Le problème** : à l'entraînement, si on donne toujours le vrai `q(t-1)` au décodeur (*teacher forcing*), le modèle ne s'habitue jamais à ses propres erreurs. À l'inférence, quand on lui donne `q̂(t-1)` (qui est imparfait), la distribution d'entrée est différente de ce qu'il a vu pendant l'entraînement → **erreur de covariate shift** qui s'accumule.

**La solution — Scheduled Sampling** : on fait décroître progressivement la probabilité d'utiliser le ground truth :

```
Epoch 1 → ss_start  :  p_teacher = 1.0   (teacher forcing pur)
Epoch ss_start → ss_end :  p_teacher décroît linéairement de 1.0 → ss_p_min
Epoch ss_end → fin   :  p_teacher = ss_p_min  (ex. 0.0 = free-running)
```

À chaque pas `t`, on tire au sort :
```
avec probabilité p_teacher : q_prev = q_real(t-1)   ← ground truth
avec probabilité 1-p_teacher : q_prev = q̂(t-1)      ← prédiction du modèle
```

**Optimisation** : quand `p_teacher = 1.0`, on n'a pas besoin de boucler pas-à-pas. On construit la séquence décalée `[q_init, q(0), q(1), …, q(T-2)]` et on l'envoie au GRU en un seul appel batché (beaucoup plus rapide sur GPU). La boucle step-by-step n'est activée que quand `p_teacher < 1.0`.

### 6.4 État initial `q_init`

Au premier pas, le décodeur a besoin de `q̂(-1)`. On utilise la position initiale du bras, toujours la même : la position HOME `(x=0.1, y=0.1)`. On calcule les angles articulaires correspondants via la cinématique inverse :

```
D  = (x² + y² - l1² - l2²) / (2·l1·l2)
q2 = -arccos(D)
q1 = atan2(y, x) - atan2(l2·sin(q2), l1 + l2·cos(q2))
```

Ce vecteur est ensuite normalisé avec les mêmes stats que le signal cible, et passé au décodeur.

### 6.5 Optimiseur et scheduler

- **Adam** (`lr=3e-4`, `weight_decay=1e-4`)
- **CosineAnnealingLR** : le learning rate suit une décroissance cosinus de `3e-4` à `1e-5` sur toute la durée de l'entraînement
- **Gradient clipping** à 5.0 : évite les explosions de gradient lors des phases de scheduled sampling

### 6.6 Mixed precision BF16

Sur GPU Nvidia (A100/T4), on utilise `torch.autocast` en BFloat16. BF16 a la même dynamique que FP32 (8 bits d'exposant) mais seulement 7 bits de mantisse (vs. 23). Cela divise par ~2 la mémoire et accélère les matmuls sans instabilité numérique, contrairement à FP16 qui nécessite un GradScaler plus agressif.

---

## 7. Inférence

À l'inférence, `p_teacher = 0.0` : le modèle est entièrement **free-running**. La boucle :

```
q_prev ← IK(HOME)  (normalisé)
h      ← h_init(contexte)

pour t = 0, 1, …, T-1 :
    x(t) ← [contexte, pe(t), q_prev]
    h    ← GRU(x(t), h)
    q̂(t) ← décodeur(h)
    q_prev ← q̂(t)
```

La prédiction se construit causalement : chaque pas dépend uniquement des pas précédents, pas du futur.

---

## 8. Compte de paramètres

| Composant | Paramètres (approx.) |
|---|---|
| ShapeEncoder (Transformer 3L) | ~2.5 M |
| speed_proj | ~130 K |
| GRU (3L, h=512, in=322) | ~3.4 M |
| h_init | ~400 K |
| Décodeur résiduel (3 ResBlocks) | ~3.2 M |
| dec_out | ~1 K |
| **Total** | **~10 M** |

---

## 9. Hyperparamètres clés

| Paramètre | Défaut | Rôle |
|---|---|---|
| `shape_embed_dim` | 256 | Dimension du vecteur de forme |
| `h_dim` | 512 | Dimension de l'état caché GRU |
| `gru_layers` | 3 | Profondeur du GRU |
| `pe_dim` | 64 | Dimension de l'encodage positionnel |
| `ss_start_epoch` | 10 | Début du scheduled sampling |
| `ss_end_epoch` | 60 | Fin de la décroissance de p_teacher |
| `ss_p_min` | 0.0 | p_teacher minimum (0 = free-running) |
| `error_weight_gamma` | 1.0 | Force de la pondération par amplitude d'erreur |
| `subsample_factor` | 1 | Sous-échantillonnage temporel (1 = aucun) |

---

## 10. Limitations actuelles

**Déterminisme** : le modèle prédit une trajectoire unique pour une pièce donnée. Il ne modélise pas l'incertitude sur les erreurs stochastiques (bruit moteur, glitches capteur). Pour obtenir une distribution sur les trajectoires possibles, il faudrait ajouter un composant stochastique (VAE, flow, ou diffusion).

**Autorégression lente à l'inférence** : la boucle step-by-step ne peut pas être parallélisée. Pour T=1300, cela représente 1300 passes séquentielles dans le GRU. Les Transformers causaux (type GPT) permettraient la parallélisation à l'entraînement via le masque causal, mais restent séquentiels à l'inférence.

**Généralisation hors-distribution** : si une pièce a une géométrie très différente de celles vues à l'entraînement, le `shape_embed` peut être dans une région de l'espace latent jamais vue par le décodeur.
