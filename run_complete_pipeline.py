"""
Complete IRIS training and evaluation pipeline.
Adapted for iris_ml_refactor structure.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a command and report status."""
    print("\n" + "="*60)
    print(f"{description}")
    print("="*60 + "\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def main():
    """Run complete pipeline for iris_ml_refactor."""
    
    print("\n" + "#"*60)
    print("# IRIS COMPLETE TRAINING & EVALUATION PIPELINE (REFACTOR VERSION)")
    print("#"*60)
    
    import argparse
    parser = argparse.ArgumentParser(description='Run IRIS training pipeline')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='msd_pancreas',
        choices=['msd_pancreas', 'acdc', 'amos', 'segthor'],
        help='Dataset to train on'
    )
    
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()
    
    dataset = args.dataset
    
    dataset_root = Path("datasets") / dataset
    iterations = args.iterations
    eval_every = 200
    num_vis_cases = 10
    
    print(f"\nUsing dataset: {dataset}")
    print(f"Dataset root: {dataset_root.resolve()}")
    
    # STEP 1 – TRAIN MODEL
    print("\n\n>>> STEP 1: Training Model with Metrics Tracking")
    
    train_cmd = [
        sys.executable,
        "-m",
        "iris_ml.scripts.train_with_metrics",
        "--dataset", dataset,
        "--dataset-root", str(dataset_root),
        "--iterations", str(iterations),
        "--eval-every", str(eval_every),
        "--lr", "0.001"
    ]
    
    if args.max_samples is not None:
        train_cmd.extend(["--max-samples", str(args.max_samples)])
    
    if not run_command(train_cmd, "Training"):
        print("\nTraining failed. Stopping pipeline.")
        return
    
    # STEP 2 – FIND CHECKPOINT
    output_dir = Path("outputs/training_with_metrics") / dataset
    checkpoint_dir = output_dir / "checkpoints"
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        print("\n✗ No checkpoints found. Stopping pipeline.")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"\n✓ Using checkpoint: {latest_checkpoint}")
    
    # STEP 3 – VISUALIZATION (IF YOU ADD SCRIPT LATER)
    print("\n\n>>> STEP 2: Generating Visualizations (optional)")
    
    vis_script = Path("src/iris_ml/scripts/visualize_results.py")
    
    if vis_script.exists():
        vis_cmd = [
            sys.executable,
            "-m",
            "iris_ml.scripts.visualize_results",
            "--dataset", dataset,
            "--checkpoint", str(latest_checkpoint),
            "--dataset-root", str(dataset_root),
            "--num-cases", str(num_vis_cases)
        ]
        
        run_command(vis_cmd, "Visualization Generation")
    else:
        print("\n⚠ Visualization script not found – skipping visualization step.")
    
    # STEP 4 – REPORT GENERATION (OPTIONAL)
    print("\n\n>>> STEP 3: Report Generation (optional)")
    
    metrics_file = output_dir / "training_metrics.json"
    
    report_script = Path("src/iris_ml/scripts/generate_report.py")
    
    if metrics_file.exists() and report_script.exists():
        report_cmd = [
            sys.executable,
            "-m",
            "iris_ml.scripts.generate_report",
            "--metrics", str(metrics_file),
            "--output-dir", str(output_dir / "report")
        ]
        
        run_command(report_cmd, "Report Generation")
    else:
        print("\n⚠ Report generation skipped (script or metrics not available)")
    
    # FINAL SUMMARY
    print("\n\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    
    print("\n📊 Results Summary:")
    print(f"  • Dataset: {dataset}")
    print(f"  • Training outputs: {output_dir}")
    print(f"  • Model checkpoint: {latest_checkpoint}")
    
    if metrics_file.exists():
        print(f"  • Metrics: {metrics_file}")
    
    print("\n✓ Pipeline finished successfully!")


if __name__ == "__main__":
    main()
