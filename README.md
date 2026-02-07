# IRIS Refactor – Few Shot Medical Image Segmentation

This repository contains a refactored implementation of the **IRIS (In-Context Learning for Medical Image Segmentation)** framework with support for episodic training, visualization, and metrics tracking.

The project currently supports training on the **MSD Pancreas dataset**, and can be extended to additional datasets such as **ACDC, AMOS, SegTHOR**, etc.

---

## 📌 Features

- Episodic few-shot training pipeline (as in original IRIS paper)
- Metrics tracking (loss, Dice score, validation performance)
- Automatic visualization of results
- Complete training pipeline script
- Support for multiple medical segmentation datasets
- Modular refactored codebase

---

# 🚀 Getting Started

Follow these steps to run the project on your local machine.

---

## 1. Clone the Repository

```bash
git clone (the link of this repo)
cd iris-refactor
```

---

## 2. Create a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

### IMPORTANT  
Datasets are NOT included in this repository.

You must download and place them manually.

---

## 🩺 Supported Datasets

Currently configured:

- MSD Pancreas (primary tested dataset)-https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 (downlaod-task-07 pancreas)
- ACDC (optional)
- AMOS (optional)
- SegTHOR (optional)

---

## 📥 Setting Up MSD Pancreas Dataset

Download the dataset from:

https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

After downloading, place it in the following structure:
Extract the .tar file and place it in the following structure:
```
iris-refactor/
└── datasets/
    └── msd_pancreas/
        ├── imagesTr/
        ├── labelsTr/
        ├── imagesTs/
        └── dataset.json
```
delete the excess files they are not of any use
---

### Verify Dataset Path

After setup, your folder should look exactly like:

```
iris-refactor/datasets/msd_pancreas
```

---

# ▶ Running the Project

---

## Train the Model

Run the complete training pipeline:

```bash
python run_complete_pipeline.py --dataset msd_pancreas --iterations 1000
```

### Available Arguments

| Argument | Description |
|--------|-------------|
| --dataset | Dataset name (e.g. msd_pancreas) |
| --iterations | Number of training iterations |
| --max-samples | Limit dataset size (optional) |

Example:

```bash
python run_complete_pipeline.py --dataset msd_pancreas --iterations 2000
```

---

## Outputs

After training, all results are saved in:

```
outputs/
└── training_with_metrics/
    └── msd_pancreas/
        ├── checkpoints/
        ├── training_metrics.json
        └── visualizations/
```

---

# 📊 Visualization

Visualizations are automatically generated after training.

To manually generate visualizations:

```bash
python -m iris_ml.scripts.visualize_results
```

Generated plots include:

- Training loss curve
- Dice score curve
- Sample prediction visualizations

---

# ➕ Adding Additional Datasets

The project supports multiple datasets.

---

## Example: Adding ACDC Dataset

### Step 1 – Download ACDC

Download from:

https://www.creatis.insa-lyon.fr/Challenge/acdc/

---

### Step 2 – Place in datasets folder

```
iris-refactor/
└── datasets/
    └── acdc/
        ├── training/
        ├── testing/
        └── dataset.json
```

---

### Step 3 – Run Training on ACDC

```bash
python run_complete_pipeline.py --dataset acdc --iterations 1000
```

---

## Running Multiple Datasets

You can train different datasets independently:

```bash
python run_complete_pipeline.py --dataset msd_pancreas --iterations 1000
python run_complete_pipeline.py --dataset acdc --iterations 1000
```

Each dataset will create its own output folder under:

```
outputs/training_with_metrics/<dataset_name>/
```

---

# 🧪 Running Tests

To verify installation:

```bash
python -m pytest -q
```

---

# 🧩 Project Structure

```
iris-refactor/
├── src/
│   └── iris_ml/
│       ├── data/
│       ├── model/
│       ├── training/
│       └── scripts/
│
├── datasets/            (created by user)
├── outputs/             (generated after training)
├── run_complete_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 💻 System Requirements

- Python 3.9+
- PyTorch 2.x
- CUDA (optional, recommended)
- At least 16GB RAM
- GPU recommended for faster training

---

# 🛠 Troubleshooting

### CUDA Not Available

If training runs on CPU instead of GPU:

- Ensure CUDA-compatible GPU drivers are installed
- Install correct PyTorch version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Dataset Errors

Make sure:

- Dataset path is correct
- Folder names exactly match expected structure
- No corrupted files
- dataset.json is present in dataset root

---

### Training Stops Early

Early stopping is enabled by default.  
To disable it, modify:

```
--early-stopping-patience 0
```

---

# 🤝 Contributing

Feel free to:

- Add new datasets
- Improve visualizations
- Optimize training pipeline
- Extend episodic training logic

---

# 📄 License

This project is provided for research and educational purposes.

---

# 👤 Author

Refactored Implementation by: **Your Name**

Based on the original IRIS research codebase.

---

### Notes

- This refactor was tested successfully on Windows 10/11 with CUDA.
- Designed for reproducibility and ease of use.

---

Happy Segmenting! 🚀
