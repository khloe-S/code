import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load datasets (adjust path if necessary)
path_real = r'F:\DESKTOP(SUB-FOLDER)\python files\Automation analysis of multiple indicators\ZZZ_Sepsis_Data_From_R.csv'
path_synth = r'F:\DESKTOP(SUB-FOLDER)\python files\Automation analysis of multiple indicators\C001_FakeHypotension.csv'

real_data = pd.read_csv(path_real)
synthetic_data = pd.read_csv(path_synth)

# Remove 'Unnamed' columns if present
real_data = real_data.loc[:, ~real_data.columns.str.contains('Unnamed')]
synthetic_data = synthetic_data.loc[:, ~synthetic_data.columns.str.contains('Unnamed')]

# Variables to compare
variables_to_compare = [
    ('Vitl004_MeanBP', 'MAP'),
    ('Vitl003_SysBP', 'systolic_bp'),
    ('Vitl005_DiaBP', 'diastolic_bp'),
    ('Labs006_Creatinine', 'serum_creatinine'),
    ('Labs026_Lactate', 'lactic_acid')
]

# Normalize data using log transformation
for real_var, synth_var in variables_to_compare:
    real_data[real_var] = np.log1p(real_data[real_var])
    synthetic_data[synth_var] = np.log1p(synthetic_data[synth_var])

# Plot each histogram separately
for real_var, synth_var in variables_to_compare:
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(
        min(real_data[real_var].min(), synthetic_data[synth_var].min()),
        max(real_data[real_var].max(), synthetic_data[synth_var].max()),
        30
    )
    sns.histplot(real_data[real_var], bins=bin_edges, kde=True, color='blue', label='Real Data', alpha=0.6)
    sns.histplot(synthetic_data[synth_var], bins=bin_edges, kde=True, color='orange', label='Synthetic Data', alpha=0.6)
    
    plt.legend(loc='upper right', fontsize=12)
    plt.title(f'{real_var} (Real) vs {synth_var} (Synthetic)', fontsize=16, weight='bold')
    plt.xlabel('Log Transformed Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    plt.show()

# Prepare data for correlation heatmap
# Filter data for correlation
filtered_real = real_data[[rv for rv, _ in variables_to_compare]]
filtered_synth = synthetic_data[[sv for _, sv in variables_to_compare]]

# Calculate Spearman correlations
real_corr = filtered_real.corr(method='spearman')
synthetic_corr = filtered_synth.corr(method='spearman')

# Plot correlation heatmaps
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(real_corr, annot=True, cmap='coolwarm', ax=ax[0], cbar_kws={'label': 'Spearman Correlation'})
ax[0].set_title('Real Data Correlations', fontsize=16, weight='bold')
sns.heatmap(synthetic_corr, annot=True, cmap='coolwarm', ax=ax[1], cbar_kws={'label': 'Spearman Correlation'})
ax[1].set_title('Synthetic Data Correlations', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()
