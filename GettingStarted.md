## Getting Started

This section describes how to set up a local Python environment, install dependencies, run the interactive web demo, and (optionally) reproduce the training pipeline.

#### 1. Clone the repository

```bash
git clone https://github.com/ElliotBlackstone/automotive-ssd-object-detection.git
cd automotive-ssd-object-detection
```

#### 2. Create and activate a new Python virtual environment

Use Python 3.12 or later.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

After activation, your shell prompt should show `(.venv)`.

#### 3. Install PyTorch

This repository uses PyTorch for training and PyTorch-based inference. If you want GPU support, install a CUDA-enabled PyTorch build before installing the rest of the requirements.

Choose the correct command for your operating system and CUDA version from the official PyTorch install selector:

```text
https://pytorch.org/get-started/locally/
```

For CPU only PyTorch:
```bash
python -m pip install torch torchvision
```

For a pip install with CUDA 12.8, the command is typically:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

To verify the PyTorch install:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: True` is expected only if you installed a CUDA-enabled PyTorch build on a machine with a compatible NVIDIA GPU/driver.

#### 4. Install the remaining dependencies

PyTorch (CPU or GPU) must be installed first.

All Python dependencies are listed in `requirements.txt`.

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

The `--no-build-isolation` flag is used because this project depends on `gen-nms-package`, a custom NMS package installed from GitHub. That package may need access to the already-installed PyTorch build when it is built.

Check dependency consistency:

```bash
python -m pip check
```

#### 5. Smoke test the environment

Run the smoke test from the repository root:

```bash
python smoke_test_requirements.py
```

A successful smoke test should import the main third-party packages and local project modules, then print diagnostic information such as the Python version, PyTorch version, whether CUDA is available, and the ONNX Runtime providers.

If the smoke test fails, inspect the first failed import. Common causes are:

* PyTorch was installed without CUDA support.
* `gen-nms-package` failed to build.
* The command was run from the wrong directory.
* A local module path changed.

You can also test the command-line interfaces without loading a model:

```bash
python SSD_fp32_pytorch_realtime_video_v2_gpu.py --help
python SSD_int8_realtime_video_v2.py --help
```

#### 6. Run the web demo locally

The FastAPI/Uvicorn app lives in `app_files/` and exposes the same interface as the deployed Cloud Run demo.

From the repository root:

```bash
cd app_files
uvicorn ssd_demo_app:app --host 0.0.0.0 --port 8000 --reload
```

Open a browser at:

```bash
http://localhost:8000/
```

Upload a dashcam-style image to visualize SSD predictions. The original image is shown on the left and the predictions are overlaid on the right.

#### 7. (Optional) Prepare the dataset and run training

If you want to reproduce training rather than only use the pre-trained model:

1. Download the dataset [here](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset).
2. Preprocess the dataset into an SSD-ready format.
   * Open `preprocess_car.ipynb` and update any paths pointing to your local copy of the dataset export and `_annotations.csv`.
   * Run the notebook to collapse traffic light subclasses into a single `trafficLight` class, create stratified train/test splits, and write the corresponding images/CSV files into `train/` and `test/` folders.
   * Empty/background images should be included in the corresponding CSV files.
3. Train the SSD model.
   * Open `SSD_model_train.ipynb`.
   * Point the data-root paths to the preprocessed `train/` and `test/` folders.
   * Run the notebook to train the SSD model using the augmentation, optimizer, and scheduler settings described in the Model training section.
   * Results can be inspected at the end of that notebook. Pre-trained models can be loaded there as well.
