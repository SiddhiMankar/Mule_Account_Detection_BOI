# Missing Value Distribution Report

This report audits the missing values in the remaining features of the dataset (after dropping high-missing features above 40%).

## Missing Value Summary Table

| Missing % Range | Number of Columns | Percentage of Features | Imputation Strategy |
| :--- | :---: | :---: | :--- |
| **0%** (Complete) | 87 | 3.51% | None |
| **0% – 5%** | 2078 | 83.82% | Median (Continuous) / Most Frequent (Binary & Cat) |
| **5% – 20%** | 164 | 6.62% | Median (Continuous) / Most Frequent (Binary & Cat) |
| **20% – 40%** | 151 | 6.09% | Median (Continuous) / Most Frequent (Binary & Cat) |
| **Total Features** | **2479** | **100.00%** | |

## Key Insights
- **No high missingness remains**: The maximum missing percentage in the remaining features is **37.8991%**, which is well below our 40% threshold.
- **High completeness**: Approximately **3.51%** of features are completely non-missing, minimizing imputation distortion.
- **Imputation selection**: A ColumnTransformer utilizing median imputation for continuous features and most-frequent imputation for binary and categorical features will be used.
