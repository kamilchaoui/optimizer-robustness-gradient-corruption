"""Run this once results_persistent.csv is complete."""
import pandas as pd
import subprocess, sys

# Merge
df_p = pd.read_csv('results/results_persistent.csv')
df_p = df_p[df_p['optimizer'] != 'signsgd']
df_sign = pd.read_csv('results/results_signsgd_persistent_lr1e3.csv')
df_final = pd.concat([df_p, df_sign], ignore_index=True)
df_final.to_csv('results/results_persistent_final.csv', index=False)
print(f'Merged: {len(df_final)} rows')

# Plot
subprocess.run([sys.executable, 'plot_results.py',
                '--input', 'results/results_persistent_final.csv',
                '--output', 'figures/persistent/'])
