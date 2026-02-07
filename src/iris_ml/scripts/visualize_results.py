import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from iris_ml.data.factory import build_dataset
import iris_ml.data.datasets
from iris_ml.model import IrisModel
from iris_ml.data import DatasetSplit

OUTPUT_DIR = Path("outputs/visualizations/msd_pancreas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(metrics_path):
    print("Loading metrics from:", metrics_path)

    with open(metrics_path, "r") as f:
        data = json.load(f)

    metrics = data["metrics"]

    train_loss = [m["value"] for m in metrics["training_loss"]]
    train_dice = [m["value"] for m in metrics["training_dice"]]

    val_loss = [m["value"] for m in metrics["validation_loss"]]
    val_dice = [m["value"] for m in metrics["validation_dice"]]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    if val_loss:
        plt.plot(range(0, len(train_loss), len(train_loss)//len(val_loss)), val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_dice, label="Train Dice")
    if val_dice:
        plt.plot(range(0, len(train_dice), len(train_dice)//len(val_dice)), val_dice, label="Val Dice")
    plt.title("Dice Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Dice")
    plt.legend()

    plt.tight_layout()
    save_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    print(f"✓ Training curves saved to: {save_path}")
    plt.show()



def visualize_prediction(dataset_root, model_path):
    print("\nLoading validation dataset...")

    val_dataset = build_dataset(
        "msd_pancreas",
        root=Path(dataset_root),
        split=DatasetSplit.VALID
    )

    print("Loading trained model...")

    # SAME CONFIG AS TRAINING SCRIPT
    model = IrisModel(
        in_channels=1,
        base_channels=16,
        num_query_tokens=8,
        volume_shape=(128, 128, 128),
        use_memory_bank=True
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("\nRunning prediction on a validation sample...")

    # Get one sample from validation set
    sample = val_dataset[0]

    image = sample["image"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    mask = (mask > 0).float()

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)

    with torch.no_grad():
        support_out = model.encode_support(image, mask)
        outputs = model(image, support_out["task_embeddings"])
        logits = outputs["logits"]
        pred = torch.sigmoid(logits)

    img_np = image.squeeze().numpy()
    mask_np = mask.squeeze().numpy()
    pred_np = pred.squeeze().numpy()

    # Pick middle slice
    z = img_np.shape[0] // 2

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img_np[z], cmap="gray")
    plt.title("CT Slice")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_np[z], cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_np[z], cmap="hot")
    plt.title("Raw Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(img_np[z], cmap="gray")
    plt.imshow(pred_np[z] > 0.5, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    base = Path("outputs/training_with_metrics/msd_pancreas")

    metrics_path = base / "training_metrics.json"
    model_path = base / "checkpoints" / "final_model.pt"

    print("\n===== VISUALIZATION START =====\n")

    plot_training_curves(metrics_path)

    visualize_prediction("datasets/msd_pancreas", model_path)

    print("\n===== DONE =====\n")


if __name__ == "__main__":
    main()
