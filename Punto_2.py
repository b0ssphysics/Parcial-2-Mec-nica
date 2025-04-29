import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set professional style
plt.style.use('seaborn-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': True,  # Enable LaTeX rendering
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.autolayout': True
})

# Experimental data
experimental_probs = np.array([
    [6, 4, 7, 2, 7, 4, 1],  # Side 1
    [6, 4, 1, 7, 5, 5, 8],  # Side 2
    [2, 1, 5, 4, 2, 4, 0],  # Side 3
    [2, 5, 5, 4, 1, 2, 4],  # Side 4
    [4, 5, 3, 7, 7, 4, 11], # Side 5
    [5, 6, 4, 1, 3, 6, 1]   # Side 6
]) / 25

experimental_error = 0.05
f_values = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4])

# Theoretical probabilities
def face_probabilities(f):
    denom = 2 + 4*f/(1 + f**2)
    P1 = P2 = 1 / denom
    P3 = P4 = P5 = P6 = (f/(1 + f**2)) / denom
    return np.array([P1, P2, P3, P4, P5, P6])

theoretical_probs = np.array([face_probabilities(f) for f in f_values])

# Create figure
fig, ax = plt.subplots(figsize=(6.5, 4))  # Single-column width for LaTeX

# Use a perceptually uniform colormap
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(f_values)))

# Plot parameters
face_labels = ['Face 1', 'Face 2', 'Face 3', 'Face 4', 'Face 5', 'Face 6']
x = np.arange(len(face_labels))
width = 0.3  # Spacing between groups

# Plot experimental and theoretical data
for i, f in enumerate(f_values):
    offset = width * (i - len(f_values)/2) / len(f_values)
    
    # Experimental data with error bars
    ax.errorbar(x + offset, experimental_probs[:,i], yerr=experimental_error,
               fmt='o', color=colors[i], markersize=5, capsize=2, capthick=1,
               elinewidth=1, label=f'Exp, f={f:.1f}' if i == 0 else "")
    
    # Theoretical predictions
    ax.plot(x + offset, theoretical_probs[i,:], 's', 
           markerfacecolor='none', markeredgecolor=colors[i],
           markersize=5, markeredgewidth=1, 
           label=f'Theo, f={f:.1f}' if i == 0 else "")

# Add connecting lines
for face in range(len(face_labels)):
    for f_idx in range(len(f_values)):
        x_pos = face + width * (f_idx - len(f_values)/2) / len(f_values)
        ax.plot([x_pos, x_pos], 
               [experimental_probs[face, f_idx] - experimental_error,
                experimental_probs[face, f_idx] + experimental_error],
               color=colors[f_idx], alpha=0.2, linestyle='-', linewidth=0.5)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(face_labels)
ax.set_ylabel('Probability', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
ax.set_axisbelow(True)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Experimental',
           markerfacecolor='k', markersize=6),
    Line2D([0], [0], marker='s', color='w', label='Theoretical',
           markerfacecolor='none', markeredgecolor='k', markersize=6)
]

ax.legend(handles=legend_elements, loc='upper right', framealpha=1)

# Add colorbar for f-values
norm = mpl.colors.Normalize(vmin=f_values.min(), vmax=f_values.max())
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
cbar.set_label('Truncation factor $f$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Adjust layout
plt.tight_layout(pad=1.0)

# Save for LaTeX (uncomment when ready)
# plt.savefig('die_probabilities.pdf', bbox_inches='tight', pad_inches=0.01)
plt.show()
