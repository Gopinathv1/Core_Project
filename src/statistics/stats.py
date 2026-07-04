from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_PATH = Path(__file__).resolve().parents[1] / "Data_Preparation" / "src" / "data.csv"
data = pd.read_csv(r'Data_Preparation\src\data.csv')

print(data.head(5))

numeric_cols = data.select_dtypes(include="number").columns

print("\nDescriptive Statistics:")
descriptive_stats = (
    data[numeric_cols]
    .agg(["mean", "median", "std", "var", "skew", "kurt"])
    .T
)

mode_values = data[numeric_cols].mode().iloc[0]
descriptive_stats["mode"] = mode_values

print(descriptive_stats[["mean", "median", "mode", "var", "std", "skew", "kurt"]])

print("\nCorrelation Analysis:")
pearson_corr = data[numeric_cols].corr(method="pearson")
print("\nPearson Correlation:")
print(pearson_corr)

spearman_corr = data[numeric_cols].corr(method="spearman")
print("\nSpearman Correlation:")
print(spearman_corr)

print("\nHypothesis Testing:")
print("\nH0: Machine temperature has no effect on failures.")
print("H1: Machine temperature affects failures.")

# T-test: compare failure vs non-failure groups by Process_Temp
failure_group = data.loc[data["Failure"] == 1, "Process_Temp"]
non_failure_group = data.loc[data["Failure"] == 0, "Process_Temp"]
t_test_result = stats.ttest_ind(failure_group, non_failure_group, equal_var=False)
print("\nT-test (Process_Temp vs Failure):")
print(f"t-statistic: {t_test_result.statistic:.4f}")
print(f"p-value: {t_test_result.pvalue:.6f}")

# Chi-square: check association between temperature category and failure
# Create a simple categorical temperature variable
median_temp = data["Process_Temp"].median()
data["temp_high"] = (data["Process_Temp"] > median_temp).astype(int)
contingency_table = pd.crosstab(data["temp_high"], data["Failure"])
chi2_result = stats.chi2_contingency(contingency_table)
print("\nChi-square test (Temp High vs Failure):")
print(contingency_table)
print(f"chi-square statistic: {chi2_result.statistic:.4f}")
print(f"p-value: {chi2_result.pvalue:.6f}")

# ANOVA: compare Process_Temp across shifts
anova_result = stats.f_oneway(
    data.loc[data["Shift"] == "Morning", "Process_Temp"],
    data.loc[data["Shift"] == "Evening", "Process_Temp"],
    data.loc[data["Shift"] == "Night", "Process_Temp"],
)
print("\nANOVA (Process_Temp across shifts):")
print(f"F-statistic: {anova_result.statistic:.4f}")
print(f"p-value: {anova_result.pvalue:.6f}")

output_dir = Path(__file__).resolve().parent / "plots"
output_dir.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.heatmap(pearson_corr, annot=False, cmap="coolwarm")
plt.title("Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "pearson_corr_heatmap.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(spearman_corr, annot=False, cmap="coolwarm")
plt.title("Spearman Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "spearman_corr_heatmap.png")
plt.close()

print(f"\nPlots saved to: {output_dir}")