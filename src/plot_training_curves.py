"""Plot training loss and learning rate curves from results/training_sailor2.csv."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path("results/training_sailor2.csv")
OUT_PATH = Path("results/training_sailor2_loss.png")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"File not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot Loss (Left Y-Axis)
    color_loss = "#1f77b4"
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Training Loss", color=color_loss, fontsize=12)
    line1 = ax1.plot(df["step"], df["loss"], color=color_loss, linewidth=2, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot Learning Rate (Right Y-Axis)
    ax2 = ax1.twinx()
    color_lr = "#ff7f0e"
    ax2.set_ylabel("Learning Rate", color=color_lr, fontsize=12)
    line2 = ax2.plot(df["step"], df["lr"], color=color_lr, linewidth=1.5, linestyle="--", label="LR")
    ax2.tick_params(axis="y", labelcolor=color_lr)
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title("Sailor2 Lemmatizer Training Progress", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    print(f"Plot saved successfully to: {OUT_PATH}")


if __name__ == "__main__":
    main()
