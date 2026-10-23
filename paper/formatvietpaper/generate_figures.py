"""Generate reproducible figures for the red-light violation paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

MODELS = ["SSD300-VGG16", "YOLOv8s", "YOLO11x"]
MAP = np.array([0.257269, 0.318488, 0.309424])
F1 = np.array([0.445347, 0.529347, 0.514224])
FPS = np.array([166.793386, 119.899674, 65.163612])
PARAMS = np.array([24.280556, 11.137535, 56.879551])


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.facecolor": "white",
    }
)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
x = np.arange(len(MODELS))
width = 0.34
bars_map = axes[0].bar(x - width / 2, MAP, width, label="mAP@0.50:0.95", color="#2468a2")
bars_f1 = axes[0].bar(x + width / 2, F1, width, label="F1@IoU 0.50", color="#ef8a17")
axes[0].set_ylim(0, 0.62)
axes[0].set_ylabel("Điểm")
axes[0].set_title("Độ chính xác trên tập test")
axes[0].set_xticks(x, MODELS)
axes[0].grid(axis="y", alpha=0.25, linewidth=0.7)
axes[0].legend(frameon=False, loc="upper left")
axes[0].bar_label(bars_map, fmt="%.3f", padding=2, fontsize=8)
axes[0].bar_label(bars_f1, fmt="%.3f", padding=2, fontsize=8)

colors = ["#2a9d62", "#2a9d62", "#2a9d62"]
bars_fps = axes[1].bar(x, FPS, width=0.56, color=colors)
axes[1].set_ylabel("Khung hình/giây")
axes[1].set_title("Tốc độ detector trên NVIDIA H100")
axes[1].set_xticks(x, MODELS)
axes[1].set_ylim(0, 190)
axes[1].grid(axis="y", alpha=0.25, linewidth=0.7)
axes[1].bar_label(bars_fps, fmt="%.1f", padding=2, fontsize=8)

fig.suptitle("So sánh ba mô hình nhận diện phương tiện", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "model_comparison.png", dpi=240, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
scatter = ax.scatter(FPS, MAP, s=PARAMS * 22, c=["#4c78a8", "#f58518", "#54a24b"], alpha=0.82,
                     edgecolor="white", linewidth=1.2)
for model, fps, score in zip(MODELS, FPS, MAP):
    ax.annotate(model, (fps, score), xytext=(6, 6), textcoords="offset points", fontsize=9)
ax.set_xlabel("FPS (batch 1, FP32, H100)")
ax.set_ylabel("mAP@0.50:0.95")
ax.set_xlim(50, 180)
ax.set_ylim(0.24, 0.335)
ax.grid(alpha=0.25)
ax.set_title("Đánh đổi chính xác - tốc độ - độ phức tạp\n(kích thước điểm tỉ lệ với số tham số)")
fig.tight_layout()
fig.savefig(FIGURES / "accuracy_speed_tradeoff.png", dpi=240, bbox_inches="tight")
plt.close(fig)

print(f"Generated figures in {FIGURES}")
