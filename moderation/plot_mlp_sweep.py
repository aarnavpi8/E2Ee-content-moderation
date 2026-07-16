import os
import json
import matplotlib.pyplot as plt

# Set style/font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "mlp", "models", "metrics.json")

with open(METRICS_PATH, "r") as f:
    data = json.load(f)

dims = ["64", "256", "1024"]

accuracy = []
precision = []
recall = []
f1 = []

for d in dims:
    m = data["dims"][d]["metrics_quantized"]
    accuracy.append(m["accuracy"])
    precision.append(m["spam_precision"])
    recall.append(m["spam_recall"])
    f1.append(m["spam_f1"])

# Define colors matching the user's plot style
c_acc = "#1d4e89"    # Dark blue
c_prec = "#27823b"   # Forest green
c_rec = "#c66917"    # Orange/Brown
c_f1 = "#732d8f"     # Purple

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

# Plot lines
x = [0, 1, 2]
ax.plot(x, accuracy, 'o-', color=c_acc, label="Accuracy", linewidth=2.5, markersize=8)
ax.plot(x, precision, 's-', color=c_prec, label="Spam precision", linewidth=2.5, markersize=8)
ax.plot(x, recall, '^-', color=c_rec, label="Spam recall", linewidth=2.5, markersize=8)
ax.plot(x, f1, 'd-', color=c_f1, label="Spam F1", linewidth=2.5, markersize=8)

# Grid
ax.grid(True, color="#e5e5e5", linestyle="-", linewidth=0.6)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')

# Ticks
ax.set_xticks(x)
ax.set_xticklabels([f"d={d}" for d in dims], fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_xlabel("Feature-hash dimension", fontsize=12)
ax.set_title("MLP classifier performance across feature dimensions", fontsize=13, fontweight='bold', pad=15)

# Y-limits and ticks (starting at 0.70 to match the original MLP plot)
ax.set_ylim(0.70, 1.01)
y_ticks = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=11)

# Annotations (with custom manual offsets to avoid overlaps)
# Accuracy: always above
for idx, val in enumerate(accuracy):
    ax.annotate(f"{val:.2f}", (x[idx], val), textcoords="offset points", 
                xytext=(0, 6), ha='center', va='bottom', color=c_acc, fontweight='bold', fontsize=9.5)

# Precision
prec_offsets = [6, 6, 6]
for idx, val in enumerate(precision):
    y_off = prec_offsets[idx]
    va_dir = 'bottom' if y_off > 0 else 'top'
    ax.annotate(f"{val:.2f}", (x[idx], val), textcoords="offset points", 
                xytext=(0, y_off), ha='center', va=va_dir, color=c_prec, fontweight='bold', fontsize=9.5)

# Recall
rec_offsets = [-14, 6, -14]
for idx, val in enumerate(recall):
    y_off = rec_offsets[idx]
    va_dir = 'bottom' if y_off > 0 else 'top'
    ax.annotate(f"{val:.2f}", (x[idx], val), textcoords="offset points", 
                xytext=(0, y_off), ha='center', va=va_dir, color=c_rec, fontweight='bold', fontsize=9.5)

# F1
f1_offsets = [6, 6, 6]
for idx, val in enumerate(f1):
    y_off = f1_offsets[idx]
    va_dir = 'bottom' if y_off > 0 else 'top'
    ax.annotate(f"{val:.2f}", (x[idx], val), textcoords="offset points", 
                xytext=(0, y_off), ha='center', va=va_dir, color=c_f1, fontweight='bold', fontsize=9.5)

# Legend (bottom right, 2 columns)
ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", edgecolor="#e5e5e5", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(HERE, "mlp_classifier_sweep.png"), dpi=300)
print("Saved mlp_classifier_sweep.png")
