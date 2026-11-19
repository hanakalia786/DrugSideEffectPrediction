
import matplotlib.pyplot as plt
import numpy as np

# Evaluation metrics
metrics = ['Precision', 'Recall', 'Micro F1', 'Macro F1', 'Hamming loss']

mlp = [0.2468, 0.3432, 0.2871, 0.0755, 0.1171]
baseline = [0.3280, 0.3168, 0.3223, 0.0319, 0.0916]

# Bar positions
x = np.arange(len(metrics))
width = 0.35

# Create graph
fig, ax = plt.subplots(figsize=(10, 5.5))

bars1 = ax.bar(
    x - width/2, mlp, width,
    label='Original MLP (threshold 0.2)',
    color='#806247'
)

bars2 = ax.bar(
    x + width/2, baseline, width,
    label='Most-frequent-label baseline',
    color='#C3AA8B'
)

# Add values above bars
ax.bar_label(bars1, fmt='%.4f', padding=3, fontsize=9)
ax.bar_label(bars2, fmt='%.4f', padding=3, fontsize=9)

ax.set_title('Test-set performance: MLP vs baseline')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 0.40)

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.figtext(
    0.5, 0.01,
    'Higher is better for precision, recall and F1; lower is better for Hamming loss.',
    ha='center', fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save the actual generated figure
plt.savefig(
    'mlp_vs_baseline_results.png',
    dpi=300,
    bbox_inches='tight'
)

print("Graph saved successfully as mlp_vs_baseline_results.png")
