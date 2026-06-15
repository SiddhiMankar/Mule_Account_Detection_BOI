# Redundancy Report

This report documents pairs of numerical features with Pearson correlation coefficients greater than **0.95**. To prevent collinearity and reduce feature dimension, the weaker feature in each pair (the one with the lower correlation to the target `F3924`) is dropped.

## Summary
- **Total Highly Correlated Pairs (>0.95)**: 3932
- **Unique Columns Dropped**: 1214
- **Unique Columns Retained**: 1299

## Redundant Pairs Details (Top 25 Pairs)

| Keep Feature | Drop Feature | Correlation | Keep Target Corr | Drop Target Corr |
| :--- | :--- | :---: | :---: | :---: |
| `customer_segment_retail` | `customer_segment_corporate` | 1.0000 | 0.0479 | 0.0479 |
| `F25` | `F19` | 1.0000 | 0.0369 | 0.0369 |
| `F3227` | `F3659` | 1.0000 | 0.0010 | 0.0010 |
| `F26` | `F20` | 1.0000 | 0.0322 | 0.0322 |
| `F3551` | `F3659` | 1.0000 | 0.0010 | 0.0010 |
| `F3659` | `F3554` | 1.0000 | 0.0010 | 0.0010 |
| `F27` | `F21` | 1.0000 | 0.0379 | 0.0379 |
| `F430` | `F424` | 1.0000 | 0.0013 | 0.0013 |
| `F335` | `F329` | 1.0000 | 0.0011 | 0.0011 |
| `F334` | `F328` | 1.0000 | 0.0210 | 0.0210 |
| `F333` | `F327` | 1.0000 | 0.0091 | 0.0091 |
| `F332` | `F326` | 1.0000 | 0.0059 | 0.0059 |
| `F331` | `F325` | 1.0000 | 0.0060 | 0.0060 |
| `F330` | `F324` | 1.0000 | 0.0037 | 0.0037 |
| `F234` | `F228` | 1.0000 | 0.0583 | 0.0583 |
| `F233` | `F227` | 1.0000 | 0.0410 | 0.0410 |
| `F232` | `F226` | 1.0000 | 0.0438 | 0.0438 |
| `F231` | `F225` | 1.0000 | 0.0568 | 0.0568 |
| `F230` | `F224` | 1.0000 | 0.0623 | 0.0623 |
| `F229` | `F223` | 1.0000 | 0.0559 | 0.0559 |
| `F809` | `F811` | 1.0000 | 0.0010 | 0.0010 |
| `F808` | `F811` | 1.0000 | 0.0010 | 0.0010 |
| `F809` | `F808` | 1.0000 | 0.0010 | 0.0010 |
| `F3282` | `F3168` | 1.0000 | 0.0022 | 0.0022 |
| `F3276` | `F3270` | 1.0000 | 0.0022 | 0.0022 |
