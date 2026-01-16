# LOBMAX MoE Model Parameter Analysis

Date: 2026-01-16

## Model Architecture Summary

LOBMAX uses a LLaMA-2 style Transformer with optional MoE (Mixture of Experts).

### Architecture Components
1. **Message Encoder**: Embedding + N transformer layers
2. **Book Encoder**: Projection + M transformer layers
3. **Fused Transformer**: Main transformer backbone with L layers
4. **Fusion Projection**: 2*d_model -> d_model
5. **Output Decoder**: d_model -> n_classes

### MoE vs Dense Difference
- **Dense**: Each layer has 1 FFN (SwiGLU)
- **MoE**: Each layer has `num_experts` FFNs + 1 router
- Only FFN parameters scale with experts; Attention stays the same

---

## Parameter Calculation Formulas

### Per Transformer Layer

#### Attention (unchanged between Dense/MoE)
```
Attention = 4 * d_model^2
          = Q + K + V + O projections
          = (d_model * d_model) * 4
```

#### FFN (SwiGLU)
```
FFN_single = 3 * d_model * mlp_dim
           = W_gate (d_model -> mlp_dim)
           + W_up   (d_model -> mlp_dim)
           + W_down (mlp_dim -> d_model)
```

For MoE with `E` experts:
```
FFN_total  = 3 * d_model * mlp_dim * E
FFN_active = 3 * d_model * mlp_dim * k  (where k = top-k experts per token)
```

#### Router (MoE only)
```
Router = d_model * num_experts
```

#### Layer Total
```
Dense Layer  = Attention + FFN
             = 4 * d^2 + 3 * d * m

MoE Layer    = Attention + (FFN * E) + Router
Total:       = 4 * d^2 + 3 * d * m * E + d * E
Active:      = 4 * d^2 + 3 * d * m * k + d * E
```

### Non-Layer Parameters
```
Embedding       = vocab_size * d_model
Book Projection = d_book * d_model
Fusion          = 2 * d_model * d_model
Output          = d_model * n_classes
RMSNorm         = d_model per layer (negligible)
```

---

## Configuration Assumptions

For LOBMAX model:
- `vocab_size = 128`
- `d_book = 40`
- `n_classes = 128`
- `n_message_layers = 2`
- `n_book_pre_layers = 1`
- `n_book_post_layers = 1`
- **Total encoder layers** = n_message + n_book_pre + n_book_post = 4
- **MoE experts** = 16
- **Top-k** = 1

---

## Case 1: 1.25B-A125M Configuration

### Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| n_layers | 12 (fused) |
| mlp_dim | 3072 |
| num_heads | 12 |
| num_experts | 16 |
| num_experts_per_tok | 1 |

### Calculation

Let d = 768, m = 3072, E = 16, k = 1, L_fused = 12, L_enc = 4

#### Per Fused Layer (MoE)
```
Attention = 4 * d^2
          = 4 * 768^2
          = 4 * 589,824
          = 2,359,296

FFN_total = 3 * d * m * E
          = 3 * 768 * 3072 * 16
          = 113,246,208

FFN_active = 3 * d * m * k
           = 3 * 768 * 3072 * 1
           = 7,077,888

Router = d * E
       = 768 * 16
       = 12,288

Layer_total  = 2,359,296 + 113,246,208 + 12,288 = 115,617,792
Layer_active = 2,359,296 + 7,077,888 + 12,288   = 9,449,472
```

#### Per Encoder Layer (Dense)
```
Attention = 4 * d^2 = 2,359,296
FFN       = 3 * d * m = 7,077,888
Layer     = 9,437,184
```

#### Non-Layer Parameters
```
Embedding       = 128 * 768 = 98,304
Book Projection = 40 * 768  = 30,720
Fusion          = 2 * 768^2 = 1,179,648
Output          = 768 * 128 = 98,304
Total Non-Layer = 1,406,976
```

#### Total Parameters

```
Total = Non-Layer + (L_fused * Layer_total) + (L_enc * Layer_enc)
      = 1,406,976 + (12 * 115,617,792) + (4 * 9,437,184)
      = 1,406,976 + 1,387,413,504 + 37,748,736
      = 1,426,569,216

Active = Non-Layer + (L_fused * Layer_active) + (L_enc * Layer_enc)
       = 1,406,976 + (12 * 9,449,472) + (4 * 9,437,184)
       = 1,406,976 + 113,393,664 + 37,748,736
       = 152,549,376
```

### Summary: 1.25B-A125M
| Metric | Value |
|--------|-------|
| **Total Parameters** | **1.43B** (1,426,569,216) |
| **Active Parameters** | **153M** (152,549,376) |
| Target Total | 1.25B |
| Target Active | 125M |

**Note**: Actual total is ~1.43B (14% over target), active is ~153M (22% over target).

---

## Case 2: 3.6B-A360M Configuration

### Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| n_layers | 24 (fused) |
| mlp_dim | 4096 |
| num_heads | 16 |
| num_experts | 16 |
| num_experts_per_tok | 1 |

### Calculation

Let d = 1024, m = 4096, E = 16, k = 1, L_fused = 24, L_enc = 4

#### Per Fused Layer (MoE)
```
Attention = 4 * d^2
          = 4 * 1024^2
          = 4,194,304

FFN_total = 3 * d * m * E
          = 3 * 1024 * 4096 * 16
          = 201,326,592

FFN_active = 3 * d * m * k
           = 3 * 1024 * 4096 * 1
           = 12,582,912

Router = d * E
       = 1024 * 16
       = 16,384

Layer_total  = 4,194,304 + 201,326,592 + 16,384 = 205,537,280
Layer_active = 4,194,304 + 12,582,912 + 16,384  = 16,793,600
```

#### Per Encoder Layer (Dense)
```
Attention = 4 * d^2 = 4,194,304
FFN       = 3 * d * m = 12,582,912
Layer     = 16,777,216
```

#### Non-Layer Parameters
```
Embedding       = 128 * 1024 = 131,072
Book Projection = 40 * 1024  = 40,960
Fusion          = 2 * 1024^2 = 2,097,152
Output          = 1024 * 128 = 131,072
Total Non-Layer = 2,400,256
```

#### Total Parameters

```
Total = Non-Layer + (L_fused * Layer_total) + (L_enc * Layer_enc)
      = 2,400,256 + (24 * 205,537,280) + (4 * 16,777,216)
      = 2,400,256 + 4,932,894,720 + 67,108,864
      = 5,002,403,840

Active = Non-Layer + (L_fused * Layer_active) + (L_enc * Layer_enc)
       = 2,400,256 + (24 * 16,793,600) + (4 * 16,777,216)
       = 2,400,256 + 403,046,400 + 67,108,864
       = 472,555,520
```

### Summary: 3.6B-A360M
| Metric | Value |
|--------|-------|
| **Total Parameters** | **5.00B** (5,002,403,840) |
| **Active Parameters** | **473M** (472,555,520) |
| Target Total | 3.6B |
| Target Active | 360M |

**Note**: Actual total is ~5.0B (39% over target), active is ~473M (31% over target).

---

## Case 3: 30B Total Parameters Configuration

### Target
- Total Parameters: 30B
- Experts: 16
- Top-k: 1

### Approach

We need to find (d_model, n_layers, mlp_dim) such that total params = 30B.

Standard ratio for LLaMA-style: `mlp_dim = 4 * d_model` (can use ~2.7x for SwiGLU efficiency)

Let's derive the formula:

```
Total = Non-Layer + L_fused * (4d^2 + 3dm*E + d*E) + L_enc * (4d^2 + 3dm)
```

Simplifying (Non-Layer is small, L_enc = 4):
```
Total ≈ L_fused * (4d^2 + 3dm*E + d*E) + 4 * (4d^2 + 3dm)
```

For MoE, FFN dominates:
```
Total ≈ L_fused * 3 * d * m * E
```

With m = 4d and E = 16:
```
Total ≈ L_fused * 3 * d * 4d * 16
      = L_fused * 192 * d^2
```

For 30B:
```
30B = L_fused * 192 * d^2
```

### Proposed Configurations

#### Option A: d_model=2048, n_layers=32
```
d = 2048, m = 8192, E = 16, L_fused = 32, L_enc = 4

Layer_total  = 4*d^2 + 3*d*m*E + d*E
             = 4*2048^2 + 3*2048*8192*16 + 2048*16
             = 16,777,216 + 805,306,368 + 32,768
             = 822,116,352

Encoder_layer = 4*d^2 + 3*d*m
              = 16,777,216 + 50,331,648
              = 67,108,864

Non-Layer = 128*2048 + 40*2048 + 2*2048^2 + 2048*128
          = 262,144 + 81,920 + 8,388,608 + 262,144
          = 8,994,816

Total = 8,994,816 + 32*822,116,352 + 4*67,108,864
      = 8,994,816 + 26,307,723,264 + 268,435,456
      = 26,585,153,536 ≈ 26.6B
```

Need more layers. Try n_layers=36:
```
Total = 8,994,816 + 36*822,116,352 + 4*67,108,864
      = 8,994,816 + 29,596,188,672 + 268,435,456
      = 29,873,618,944 ≈ 29.9B ✓
```

#### Option A (Refined): d_model=2048, n_layers=36, mlp_dim=8192

```
Active_layer = 4*d^2 + 3*d*m*1 + d*E
             = 16,777,216 + 50,331,648 + 32,768
             = 67,141,632

Active = 8,994,816 + 36*67,141,632 + 4*67,108,864
       = 8,994,816 + 2,417,098,752 + 268,435,456
       = 2,694,529,024 ≈ 2.7B
```

| Configuration | Value |
|--------------|-------|
| d_model | 2048 |
| num_heads | 16 |
| head_dim | 128 |
| mlp_dim | 8192 |
| n_layers (fused) | 36 |
| n_message_layers | 2 |
| n_book_pre_layers | 1 |
| n_book_post_layers | 1 |
| num_experts | 16 |
| num_experts_per_tok | 1 |
| **Total Parameters** | **29.9B** |
| **Active Parameters** | **2.7B** |

---

#### Option B: d_model=2560, n_layers=24, mlp_dim=10240

```
d = 2560, m = 10240, E = 16, L_fused = 24, L_enc = 4

Layer_total  = 4*d^2 + 3*d*m*E + d*E
             = 4*2560^2 + 3*2560*10240*16 + 2560*16
             = 26,214,400 + 1,258,291,200 + 40,960
             = 1,284,546,560

Encoder_layer = 4*d^2 + 3*d*m
              = 26,214,400 + 78,643,200
              = 104,857,600

Non-Layer = 128*2560 + 40*2560 + 2*2560^2 + 2560*128
          = 327,680 + 102,400 + 13,107,200 + 327,680
          = 13,864,960

Total = 13,864,960 + 24*1,284,546,560 + 4*104,857,600
      = 13,864,960 + 30,829,117,440 + 419,430,400
      = 31,262,412,800 ≈ 31.3B
```

Slightly over. Try n_layers=23:
```
Total = 13,864,960 + 23*1,284,546,560 + 4*104,857,600
      = 13,864,960 + 29,544,570,880 + 419,430,400
      = 29,977,866,240 ≈ 30.0B ✓
```

#### Option B (Refined): d_model=2560, n_layers=23, mlp_dim=10240

```
Active_layer = 4*d^2 + 3*d*m*1 + d*E
             = 26,214,400 + 78,643,200 + 40,960
             = 104,898,560

Active = 13,864,960 + 23*104,898,560 + 4*104,857,600
       = 13,864,960 + 2,412,666,880 + 419,430,400
       = 2,845,962,240 ≈ 2.8B
```

| Configuration | Value |
|--------------|-------|
| d_model | 2560 |
| num_heads | 20 |
| head_dim | 128 |
| mlp_dim | 10240 |
| n_layers (fused) | 23 |
| n_message_layers | 2 |
| n_book_pre_layers | 1 |
| n_book_post_layers | 1 |
| num_experts | 16 |
| num_experts_per_tok | 1 |
| **Total Parameters** | **30.0B** |
| **Active Parameters** | **2.8B** |

---

#### Option C: d_model=3072, n_layers=18, mlp_dim=12288

```
d = 3072, m = 12288, E = 16, L_fused = 18, L_enc = 4

Layer_total  = 4*d^2 + 3*d*m*E + d*E
             = 4*3072^2 + 3*3072*12288*16 + 3072*16
             = 37,748,736 + 1,811,939,328 + 49,152
             = 1,849,737,216

Encoder_layer = 4*d^2 + 3*d*m
              = 37,748,736 + 113,246,208
              = 150,994,944

Non-Layer = 128*3072 + 40*3072 + 2*3072^2 + 3072*128
          = 393,216 + 122,880 + 18,874,368 + 393,216
          = 19,783,680

Total = 19,783,680 + 18*1,849,737,216 + 4*150,994,944
      = 19,783,680 + 33,295,269,888 + 603,979,776
      = 33,919,033,344 ≈ 33.9B
```

Too high. Try n_layers=16:
```
Total = 19,783,680 + 16*1,849,737,216 + 4*150,994,944
      = 19,783,680 + 29,595,795,456 + 603,979,776
      = 30,219,558,912 ≈ 30.2B ✓
```

#### Option C (Refined): d_model=3072, n_layers=16, mlp_dim=12288

```
Active_layer = 4*d^2 + 3*d*m*1 + d*E
             = 37,748,736 + 113,246,208 + 49,152
             = 151,044,096

Active = 19,783,680 + 16*151,044,096 + 4*150,994,944
       = 19,783,680 + 2,416,705,536 + 603,979,776
       = 3,040,468,992 ≈ 3.0B
```

| Configuration | Value |
|--------------|-------|
| d_model | 3072 |
| num_heads | 24 |
| head_dim | 128 |
| mlp_dim | 12288 |
| n_layers (fused) | 16 |
| n_message_layers | 2 |
| n_book_pre_layers | 1 |
| n_book_post_layers | 1 |
| num_experts | 16 |
| num_experts_per_tok | 1 |
| **Total Parameters** | **30.2B** |
| **Active Parameters** | **3.0B** |

---

## Summary: 30B Model Options

| Option | d_model | n_layers | mlp_dim | Total | Active | Active Ratio |
|--------|---------|----------|---------|-------|--------|--------------|
| A | 2048 | 36 | 8192 | 29.9B | 2.7B | 9.0% |
| B | 2560 | 23 | 10240 | 30.0B | 2.8B | 9.3% |
| C | 3072 | 16 | 12288 | 30.2B | 3.0B | 9.9% |

### Recommendation

**Option B (d_model=2560, n_layers=23)** provides a good balance:
- Exactly 30B total parameters
- 2.8B active parameters (top-1, 16 experts)
- Reasonable depth (23 layers) for good gradient flow
- Standard 4x MLP ratio

**Option A** is also excellent if you prefer a deeper model with smaller width.

---

## Appendix: Aligned Equation Derivation (AED)

### Total Parameters for MoE Layer

```
Layer_total = Attention + FFN_total + Router
            = 4 * d^2 + 3 * d * m * E + d * E
            = d * (4d + 3mE + E)
            = d * (4d + E(3m + 1))
```

### Active Parameters for MoE Layer (top-k)

```
Layer_active = Attention + FFN_active + Router
             = 4 * d^2 + 3 * d * m * k + d * E
             = d * (4d + 3mk + E)
```

### 30B Target Derivation (Option B)

```
Total = Non-Layer + L_fused * Layer_total + L_enc * Layer_enc
      = 13,864,960 + 23 * 1,284,546,560 + 4 * 104,857,600
      = 13,864,960 + 29,544,570,880 + 419,430,400
      = 29,977,866,240
      ≈ 30.0B
```

### Active/Total Ratio

For MoE with E experts and top-k routing:
```
Ratio_layer ≈ (4d^2 + 3dmk) / (4d^2 + 3dmE)
            ≈ (4d + 3mk) / (4d + 3mE)
```

When m >> d (typical):
```
Ratio_layer ≈ 3mk / 3mE = k / E = 1/16 ≈ 6.25%
```

But attention is shared, so actual ratio is higher:
```
Ratio_actual = (4d^2 + 3dmk) / (4d^2 + 3dmE)
             = (4d + 3mk) / (4d + 3mE)
             = (4*2560 + 3*10240*1) / (4*2560 + 3*10240*16)
             = (10240 + 30720) / (10240 + 491520)
             = 40960 / 501760
             ≈ 8.2%
```

Including router and non-layer params, final active ratio is ~9-10%.
