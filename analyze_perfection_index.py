"""
Analyze the perfection_index_predictions.csv file to identify perfectionist sessions
and generate summary statistics.
"""

import pandas as pd
import numpy as np

# Load the predictions CSV
df = pd.read_csv('perfection_index_predictions.csv')

print("=" * 70)
print("PERFECTION INDEX ANALYSIS")
print("=" * 70)

# Basic statistics
total_sessions = len(df)
print(f"\nTotal number of sessions: {total_sessions:,}")

# Define perfectionist threshold
perfectionist_threshold = 0.7

# Count perfectionist sessions
perfectionist_sessions = df[df['perfection_index'] >= perfectionist_threshold]
num_perfectionist = len(perfectionist_sessions)
percentage_perfectionist = (num_perfectionist / total_sessions) * 100

print(f"\n{'='*70}")
print(f"PERFECTIONIST SESSIONS (perfection_index >= {perfectionist_threshold})")
print(f"{'='*70}")
print(f"Number of perfectionist sessions: {num_perfectionist:,}")
print(f"Percentage of all sessions: {percentage_perfectionist:.2f}%")

# Summary statistics
print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")

avg_perfection_index = df['perfection_index'].mean()
median_perfection_index = df['perfection_index'].median()
std_perfection_index = df['perfection_index'].std()
min_perfection_index = df['perfection_index'].min()
max_perfection_index = df['perfection_index'].max()

print(f"Average perfection_index: {avg_perfection_index:.4f}")
print(f"Median perfection_index: {median_perfection_index:.4f}")
print(f"Standard deviation: {std_perfection_index:.4f}")
print(f"Minimum perfection_index: {min_perfection_index:.4f}")
print(f"Maximum perfection_index: {max_perfection_index:.4f}")

# Quartile analysis
print(f"\n{'='*70}")
print("QUARTILE DISTRIBUTION")
print(f"{'='*70}")

quartiles = df['perfection_index'].quantile([0.25, 0.5, 0.75])

q1_threshold = quartiles[0.25]
q2_threshold = quartiles[0.5]
q3_threshold = quartiles[0.75]

# Count sessions in each quartile
q1_sessions = df[df['perfection_index'] < q1_threshold]
q2_sessions = df[(df['perfection_index'] >= q1_threshold) & (df['perfection_index'] < q2_threshold)]
q3_sessions = df[(df['perfection_index'] >= q2_threshold) & (df['perfection_index'] < q3_threshold)]
q4_sessions = df[df['perfection_index'] >= q3_threshold]

# Create summary table
summary_data = {
    'Quartile': ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)'],
    'Range': [
        f'[{min_perfection_index:.4f}, {q1_threshold:.4f})',
        f'[{q1_threshold:.4f}, {q2_threshold:.4f})',
        f'[{q2_threshold:.4f}, {q3_threshold:.4f})',
        f'[{q3_threshold:.4f}, {max_perfection_index:.4f}]'
    ],
    'Count': [
        len(q1_sessions),
        len(q2_sessions),
        len(q3_sessions),
        len(q4_sessions)
    ],
    'Percentage': [
        f"{(len(q1_sessions)/total_sessions)*100:.2f}%",
        f"{(len(q2_sessions)/total_sessions)*100:.2f}%",
        f"{(len(q3_sessions)/total_sessions)*100:.2f}%",
        f"{(len(q4_sessions)/total_sessions)*100:.2f}%"
    ],
    'Average PI': [
        f"{q1_sessions['perfection_index'].mean():.4f}",
        f"{q2_sessions['perfection_index'].mean():.4f}",
        f"{q3_sessions['perfection_index'].mean():.4f}",
        f"{q4_sessions['perfection_index'].mean():.4f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Additional: Show distribution around perfectionist threshold
print(f"\n{'='*70}")
print(f"DISTRIBUTION AROUND PERFECTIONIST THRESHOLD (>= {perfectionist_threshold})")
print(f"{'='*70}")

# Calculate average perfection_index for perfectionist sessions
avg_perfectionist_pi = perfectionist_sessions['perfection_index'].mean()
avg_non_perfectionist_pi = df[df['perfection_index'] < perfectionist_threshold]['perfection_index'].mean()

print(f"Average perfection_index (perfectionist sessions): {avg_perfectionist_pi:.4f}")
print(f"Average perfection_index (non-perfectionist sessions): {avg_non_perfectionist_pi:.4f}")
print(f"Difference: {avg_perfectionist_pi - avg_non_perfectionist_pi:.4f}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}\n")

