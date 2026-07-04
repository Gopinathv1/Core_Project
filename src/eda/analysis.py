from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parents[1] / "Data_Preparation" / "src" / "data.csv"
data = pd.read_csv(r'Data_Preparation\src\data.csv')

print(data.head(5))

# Machine Analysis
machine_defects = (
    data.groupby("Machine_ID")
    .agg(total_records=("Machine_ID", "size"), defects=("Defect", "sum"), failures=("Failure", "sum"))
    .sort_values("defects", ascending=False)
)
print("\nMachine Analysis:")
print(machine_defects.head(10))
print(f"Machine with most defects: {machine_defects.index[0]}")

# Shift Analysis
shift_summary = (
    data.groupby("Shift")
    .agg(total_records=("Shift", "size"), defects=("Defect", "sum"), failures=("Failure", "sum"))
    .assign(defect_rate=lambda x: x["defects"] / x["total_records"])
    .sort_values("defect_rate", ascending=False)
)
print("\nShift Analysis:")
print(shift_summary)
print(f"Highest defect shift: {shift_summary.index[0]}")

# Operator Analysis
operator_summary = (
    data.groupby("Operator")
    .agg(total_records=("Operator", "size"), defects=("Defect", "sum"), failures=("Failure", "sum"))
    .assign(
        defect_rate=lambda x: x["defects"] / x["total_records"],
        failure_rate=lambda x: x["failures"] / x["total_records"],
    )
    .sort_values("failure_rate", ascending=False)
)

print("\nOperator Analysis:")
print(operator_summary.head(10))

best_operator = operator_summary["failure_rate"].idxmin()
worst_operator = operator_summary["failure_rate"].idxmax()
print(f"Best operator: {best_operator}")
print(f"Worst operator: {worst_operator}")

# Production Analysis
production_corr_defect = data["Production_Count"].corr(data["Defect"])
production_corr_failure = data["Production_Count"].corr(data["Failure"])
print(f"\nCorrelation between Production_Count and Defect: {production_corr_defect:.3f}")
print(f"Correlation between Production_Count and Failure: {production_corr_failure:.3f}")

# Temperature Analysis
print("\nTemperature correlation matrix:")
print(data[["Air_Temp", "Process_Temp", "Failure"]].corr())

# Tool Wear Analysis
print("\nTool Wear by Failure status:")
print(data.groupby("Failure")["Tool_Wear"].agg(["mean", "median", "std"]))

# Visualizations
output_dir = Path(__file__).resolve().parent / "plots"
output_dir.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(data["Tool_Wear"], bins=30, kde=True)
plt.title("Tool Wear Distribution")
plt.xlabel("Tool Wear")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(output_dir / "tool_wear_hist.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="Failure", y="Tool_Wear", data=data)
plt.title("Tool Wear vs Failure")
plt.xlabel("Failure")
plt.ylabel("Tool Wear")
plt.tight_layout()
plt.savefig(output_dir / "tool_wear_boxplot.png")
plt.close()

plt.figure(figsize=(10, 8))
correlation_matrix = data.select_dtypes(include="number").corr()
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_dir / "feature_heatmap.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x="Production_Count", y="Defect", hue="Failure", alpha=0.4)
plt.title("Production Count vs Defects")
plt.xlabel("Production Count")
plt.ylabel("Defect")
plt.tight_layout()
plt.savefig(output_dir / "production_vs_defect.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x="Air_Temp", y="Process_Temp", hue="Failure", alpha=0.4)
plt.title("Air Temp vs Process Temp by Failure")
plt.xlabel("Air Temp")
plt.ylabel("Process Temp")
plt.tight_layout()
plt.savefig(output_dir / "temp_failure_scatter.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=data, x="Shift", hue="Failure")
plt.title("Failure Count by Shift")
plt.xlabel("Shift")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(output_dir / "failure_shift_countplot.png")
plt.close()

# Optional pairplot
# sns.pairplot(data[["Air_Temp", "Process_Temp", "Tool_Wear", "Production_Count", "Failure"]].assign(Failure=data["Failure"].astype(str)), hue="Failure")
# plt.savefig(output_dir / "pairplot.png")
# plt.close()

print(f"\nPlots saved to: {output_dir}")


