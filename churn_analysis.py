"""
Customer Churn Analysis: SQL + Machine Learning
Dataset: Telco Customer Churn (Kaggle, IBM sample data)
Download: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Save the file as 'WA_Fn-UseC_-Telco-Customer-Churn.csv' next to this script
(or change CSV_PATH below). Works in Google Colab, Jupyter, or VS Code.
"""
import os
import sqlite3
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings("ignore")
CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD AND CLEAN
# ---------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print("Raw shape:", df.shape)

# TotalCharges has blank strings for brand-new customers -> convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
missing = df["TotalCharges"].isna().sum()
df = df.dropna(subset=["TotalCharges"]).drop_duplicates().reset_index(drop=True)
print(f"Dropped {missing} rows with missing TotalCharges. Clean shape: {df.shape}")

df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
print(f"Overall churn rate: {df['ChurnFlag'].mean():.1%}")

# ---------------------------------------------------------------
# 2. SQL ANALYSIS (SQLite, in memory)
# ---------------------------------------------------------------
conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")

queries = {
    "Churn rate by contract type": """
        SELECT Contract,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1) AS churn_pct
        FROM customers
        GROUP BY Contract
        ORDER BY churn_pct DESC;""",
    "Churn rate by tenure bucket": """
        SELECT CASE WHEN tenure <= 12 THEN '0-12 months'
                    WHEN tenure <= 24 THEN '13-24 months'
                    WHEN tenure <= 48 THEN '25-48 months'
                    ELSE '49+ months' END AS tenure_group,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1) AS churn_pct
        FROM customers
        GROUP BY tenure_group
        ORDER BY MIN(tenure);""",
    "Churn rate by internet service and payment method": """
        SELECT InternetService, PaymentMethod,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(ChurnFlag) / COUNT(*), 1) AS churn_pct
        FROM customers
        GROUP BY InternetService, PaymentMethod
        HAVING COUNT(*) >= 30
        ORDER BY churn_pct DESC
        LIMIT 5;""",
    "Average monthly charge: churned vs retained": """
        SELECT Churn,
               ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charge,
               ROUND(AVG(tenure), 1) AS avg_tenure_months
        FROM customers
        GROUP BY Churn;""",
    "Revenue at risk (monthly) from churned customers by contract": """
        SELECT Contract,
               ROUND(SUM(MonthlyCharges), 0) AS monthly_revenue_lost
        FROM customers
        WHERE ChurnFlag = 1
        GROUP BY Contract
        ORDER BY monthly_revenue_lost DESC;""",
}

sql_results = []
for title, q in queries.items():
    res = pd.read_sql_query(q, conn)
    print(f"\n--- {title} ---\n{res.to_string(index=False)}")
    sql_results.append(f"### {title}\n\n{res.to_markdown(index=False)}\n")
with open(f"{OUT_DIR}/sql_insights.md", "w") as f:
    f.write("\n".join(sql_results))

# ---------------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df, x="Contract", y="ChurnFlag", errorbar=None, ax=ax)
ax.set_ylabel("Churn rate"); ax.set_title("Churn Rate by Contract Type")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/churn_by_contract.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack", ax=ax)
ax.set_title("Customer Tenure Distribution by Churn")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/tenure_distribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
ax.set_title("Monthly Charges: Churned vs Retained")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/monthly_charges_box.png", dpi=150); plt.close()

# ---------------------------------------------------------------
# 4. MACHINE LEARNING
# ---------------------------------------------------------------
X = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
y = df["ChurnFlag"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42, n_jobs=-1),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "CV F1 (train)": round(cv_f1, 3),
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "Precision": round(precision_score(y_test, pred), 3),
        "Recall": round(recall_score(y_test, pred), 3),
        "F1": round(f1_score(y_test, pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 3),
    })
    fitted[name] = (pipe, pred)

results = pd.DataFrame(rows)
print("\n=== MODEL COMPARISON (held-out test set) ===")
print(results.to_string(index=False))
results.to_csv(f"{OUT_DIR}/model_results.csv", index=False)

# Best model by test F1 (churn is imbalanced, so F1 beats plain accuracy)
best_name = results.sort_values("F1", ascending=False).iloc[0]["Model"]
best_pipe, best_pred = fitted[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_test, best_pred, target_names=["Retained", "Churned"]))

cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix: {best_name}")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150); plt.close()

# Feature importance / coefficients
feat_names = best_pipe.named_steps["prep"].get_feature_names_out()
est = best_pipe.named_steps["model"]
if hasattr(est, "feature_importances_"):
    imp = pd.Series(est.feature_importances_, index=feat_names)
else:
    imp = pd.Series(est.coef_[0], index=feat_names)
top = imp.reindex(imp.abs().sort_values(ascending=False).head(10).index).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
top.plot(kind="barh", ax=ax)
ax.set_title(f"Top 10 Drivers of Churn ({best_name})")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150); plt.close()

print(f"\nAll charts and tables saved in the '{OUT_DIR}/' folder.")
