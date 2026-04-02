import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_training_log(filepath):
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Remove leading timestamp (e.g., "286.5s 209 ") if present
        line = re.sub(r"^\d+\.\d+s\s+\d+\s+", "", line)

        # Match training step block
        step_match = re.match(r"Step:\s+(\d+)/\d+", line)
        if step_match:
            step = int(step_match.group(1))
            # Look ahead for Training Loss within next ~5 lines
            for j in range(i + 1, min(i + 6, len(lines))):
                inner = re.sub(r"^\d+\.\d+s\s+\d+\s+", "", lines[j].strip())
                loss_match = re.match(r"Training Loss:\s+([\d.]+)", inner)
                if loss_match:
                    train_steps.append(step)
                    train_losses.append(float(loss_match.group(1)))
                    break

        # Match evaluation block
        eval_match = re.match(r"EVALUATION - Step:\s+(\d+)", line)
        if eval_match:
            step = int(eval_match.group(1))
            for j in range(i + 1, min(i + 6, len(lines))):
                inner = re.sub(r"^\d+\.\d+s\s+\d+\s+", "", lines[j].strip())
                val_match = re.match(r"Validation Loss:\s+([\d.]+)", inner)
                if val_match:
                    eval_steps.append(step)
                    eval_losses.append(float(val_match.group(1)))
                    break

        i += 1

    return train_steps, train_losses, eval_steps, eval_losses


def plot_losses(train_steps, train_losses, eval_steps, eval_losses, output_path="loss_plot.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(train_steps, train_losses, label="Training Loss", color="blue",
            linewidth=1.2, alpha=0.85)
    ax.plot(eval_steps, eval_losses, label="Validation Loss", color="red",
            linewidth=2.0)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training & Validation Loss", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    log_file = "train_test_loss.txt"  # Change this to your actual file path

    train_steps, train_losses, eval_steps, eval_losses = parse_training_log(log_file)

    print(f"Parsed {len(train_steps)} training steps and {len(eval_steps)} evaluation points.")
    print("\nEvaluation checkpoints:")
    for s, l in zip(eval_steps, eval_losses):
        print(f"  Step {s:>6}: Validation Loss = {l:.4f}")

    plot_losses(train_steps, train_losses, eval_steps, eval_losses,
                output_path="loss_plot.png")
