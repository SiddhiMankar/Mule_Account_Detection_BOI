# Bank of India Highlighted Features Audit

This report details the characteristics and predictive signal of the 18 specific features highlighted by the Bank of India.

## Master Summary Table

| Feature | Type | Missing % | Target Correlation | RF Importance Rank | MI Rank | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `F115` | Continuous | 3.94% | - | - | - | ❌ Dropped (Redundant) |
| `F321` | Continuous | 0.94% | 0.0097 | 317 | 521 | ✅ Kept |
| `F527` | Continuous | 8.68% | - | - | - | ❌ Dropped (Redundant) |
| `F531` | Continuous | 7.06% | - | - | - | ❌ Dropped (Redundant) |
| `F670` | Binary | 0.18% | 0.0471 | 446 | 289 | ✅ Kept |
| `F1692` | Continuous | 0.08% | 0.0181 | 286 | 700 | ✅ Kept |
| `F2082` | Continuous | 0.08% | 0.0242 | 530 | 1074 | ✅ Kept |
| `F2122` | Continuous | 0.04% | - | - | - | ❌ Dropped (Redundant) |
| `F2582` | Continuous | 37.48% | 0.0039 | 481 | 816 | ✅ Kept |
| `F2678` | Continuous | 28.34% | - | - | - | ❌ Dropped (Redundant) |
| `F2737` | Continuous | 1.23% | 0.0008 | 204 | 616 | ✅ Kept |
| `F2956` | Continuous | 11.29% | - | - | - | ❌ Dropped (Redundant) |
| `F3043` | - | >40% | - | - | - | ❌ Dropped in Phase 1 (Missingness > 40%) |
| `F3836` | Continuous | 0.00% | - | - | - | ❌ Dropped (Redundant) |
| `F3887` | Continuous | 0.00% | - | - | - | ❌ Dropped (Redundant) |
| `F3889` | Categorical | 0.00% | 0.0200 (max) | 531 (best) | 216 (best) | ✅ Kept & Encoded |
| `F3891` | Categorical | 0.00% | 0.0432 (max) | 552 (best) | 204 (best) | ✅ Kept & Encoded |
| `F3894` | Continuous | 0.03% | 0.0081 | 218 | 1251 | ✅ Kept |

## Detailed Class Distribution Profiles

Below are the class-specific statistical profiles for key continuous BOI features that show predictive signal.

### Feature `F321` Profile

| Target Class | Count | Mean | Std | Min | Median (50%) | Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (0)** | 9001 | 0.2822 | 1.3670 | -3.6897 | 0.0000 | 32.8621 |
| **Mule (1)** | 81 | 0.1413 | 1.1331 | -3.3448 | -0.1034 | 3.3103 |

### Feature `F670` Profile

| Target Class | Count | Mean | Std | Min | Median (50%) | Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (0)** | 9001 | 0.0901 | 0.2863 | 0.0000 | 0.0000 | 1.0000 |
| **Mule (1)** | 81 | 0.2346 | 0.4264 | 0.0000 | 0.0000 | 1.0000 |

### Feature `F1692` Profile

| Target Class | Count | Mean | Std | Min | Median (50%) | Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (0)** | 9001 | 0.2609 | 0.7818 | 0.0000 | 0.0000 | 14.0000 |
| **Mule (1)** | 81 | 0.1111 | 0.5244 | 0.0000 | 0.0000 | 4.0000 |

### Feature `F2082` Profile

| Target Class | Count | Mean | Std | Min | Median (50%) | Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (0)** | 9001 | 0.0209 | 0.0813 | 0.0000 | 0.0000 | 1.0000 |
| **Mule (1)** | 81 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Feature `F2582` Profile

| Target Class | Count | Mean | Std | Min | Median (50%) | Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (0)** | 9001 | 0.9530 | 8.2595 | -17.6000 | 0.0000 | 377.8000 |
| **Mule (1)** | 81 | 1.2988 | 5.5766 | -6.6000 | 0.0000 | 39.4000 |

