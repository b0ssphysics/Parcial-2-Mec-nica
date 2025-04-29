import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

# Experimental data (25 throws per die) with ±0.05 error
experimental_probs = np.array([
    [6, 4, 7, 2, 7, 4, 1],  # Side 1
    [6, 4, 1, 7, 5, 5, 8],  # Side 2
    [2, 1, 5, 4, 2, 4, 0],  # Side 3
    [2, 5, 5, 4, 1, 2, 4],  # Side 4
    [4, 5, 3, 7, 7, 4, 11], # Side 5
    [5, 6, 4, 1, 3, 6, 1]   # Side 6
]) / 25  # Convert to probabilities

experimental_error = 0.05  # Measurement uncertainty

# Truncation factors
f_values = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4])

# Theoretical probabilities
def face_probabilities(f):
    denom = 2 + 4*f/(1 + f**2)
    P1 = P2 = 1 / denom
    P3 = P4 = P5 = P6 = (f/(1 + f**2)) / denom
    return np.array([P1, P2, P3, P4, P5, P6])

theoretical_probs = np.array([face_probabilities(f) for f in f_values])

# Create figure
plt.figure(figsize=(14, 8))
plt.suptitle("Experimental vs Theoretical Face Probabilities with Measurement Error Bars", y=1.02)

# Main plot comparing probabilities
ax1 = plt.subplot(211)
x = np.arange(6)  # Faces 1-6
width = 0.1

# Color palette for different f-values
colors = plt.cm.viridis(np.linspace(0, 1, len(f_values)))

for i, f in enumerate(f_values):
    offset = width * (i - 3)
    # Experimental with error bars
    ax1.errorbar(x + offset, experimental_probs[:,i], 
                yerr=experimental_error, fmt='o',
                color=colors[i], markersize=8, capsize=5,
                label=f'f={f} Exp')
    # Theoretical predictions
    ax1.plot(x + offset, theoretical_probs[i,:], 's', 
            markerfacecolor='none', markeredgecolor=colors[i],
            markersize=8, markeredgewidth=2, label=f'f={f} Theo')

ax1.set_xticks(x)
ax1.set_xticklabels(['Face 1', 'Face 2', 'Face 3', 'Face 4', 'Face 5', 'Face 6'])
ax1.set_ylabel("Probability")
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# KL divergence plot
ax2 = plt.subplot(212)
kl_divergences = []
for i, f in enumerate(f_values):
    kl = np.sum(rel_entr(experimental_probs[:,i], theoretical_probs[i,:])) / np.log(2)
    kl_divergences.append(kl)

ax2.bar(range(len(f_values)), kl_divergences, color=colors)
ax2.set_xticks(range(len(f_values)))
ax2.set_xticklabels([f'f={f:.1f}' for f in f_values])
ax2.set_xlabel("Truncation Factor (f)")
ax2.set_ylabel("KL Divergence (bits)")
ax2.set_title("Goodness-of-Fit Between Experimental and Theoretical Distributions")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
