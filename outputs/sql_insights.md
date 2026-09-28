### Churn rate by contract type

| Contract       |   customers |   churn_pct |
|:---------------|------------:|------------:|
| Month-to-month |        3875 |        42.7 |
| One year       |        1472 |        11.3 |
| Two year       |        1685 |         2.8 |

### Churn rate by tenure bucket

| tenure_group   |   customers |   churn_pct |
|:---------------|------------:|------------:|
| 0-12 months    |        2175 |        47.7 |
| 13-24 months   |        1024 |        28.7 |
| 25-48 months   |        1594 |        20.4 |
| 49+ months     |        2239 |         9.5 |

### Churn rate by internet service and payment method

| InternetService   | PaymentMethod             |   customers |   churn_pct |
|:------------------|:--------------------------|------------:|------------:|
| Fiber optic       | Electronic check          |        1595 |        53.2 |
| Fiber optic       | Mailed check              |         258 |        42.6 |
| DSL               | Electronic check          |         648 |        31.9 |
| Fiber optic       | Bank transfer (automatic) |         646 |        28.9 |
| Fiber optic       | Credit card (automatic)   |         597 |        25.3 |

### Average monthly charge: churned vs retained

| Churn   |   avg_monthly_charge |   avg_tenure_months |
|:--------|---------------------:|--------------------:|
| No      |                61.31 |                37.7 |
| Yes     |                74.44 |                18   |

### Revenue at risk (monthly) from churned customers by contract

| Contract       |   monthly_revenue_lost |
|:---------------|-----------------------:|
| Month-to-month |                 120847 |
| One year       |                  14118 |
| Two year       |                   4165 |
