import pandas as pd
import matplotlib.pyplot as plt

# Load your completed CSV
df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()  # Clean whitespace from headers

# Define the 4 standard YOLO metrics
metrics = [
    ('train/box_loss', 'val/box_loss', 'Box Loss'),
    ('train/cls_loss', 'val/cls_loss', 'Class Loss'),
    ('metrics/precision(B)', 'metrics/recall(B)', 'Precision/Recall'),
    ('metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'mAP Metrics')
]

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.ravel()

for i, (m1, m2, title) in enumerate(metrics):
    if m1 in df.columns:
        axs[i].plot(df['epoch'], df[m1], label=m1)
    if m2 in df.columns:
        axs[i].plot(df['epoch'], df[m2], label=m2)
    axs[i].set_title(title)
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.savefig('reconstructed_results.png')
print("Charts saved as reconstructed_results.png")
