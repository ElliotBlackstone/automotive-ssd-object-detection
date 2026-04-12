# Automotive SSD Object Detection

A from-scratch PyTorch implementation of the Single Shot Multibox Detector (SSD) for automotive object detection on dashcam imagery.

This repository now contains **two generations of the project**:

- **v1**: the original baseline implementation
- **v2**: a reorganized and expanded codebase with modular model/training files, model comparisons, benchmarking, ONNX/PTQ experiments, and local real-time video inference

The project also includes:

- dataset exploration and preprocessing notebooks
- training and evaluation code
- benchmarking and profiling utilities
- ONNX export and post-training quantization (PTQ) workflows
- local webcam/video inference scripts
- the earlier FastAPI/Cloud Run web app code used for a hosted demo

## Repository Overview

```text
.
├── v1/                           # Original baseline SSD code
├── v2/                           # Newer codebase
│   ├── model_files/              # SSD model, priors, encode/decode, NMS helpers, visualization
│   └── training_files/           # Training loop, dataloaders, losses, profiling, checkpoints
├── PTQ_testing/                  # ONNX export, parity tests, PTQ, evaluation, benchmark scripts
├── benchmarking/                 # DataLoader and training/inference benchmark utilities
├── docs/                         # GitHub pages website files
├── app_files/                    # Earlier FastAPI + Docker web demo code
├── figures/                      # Figures and plots
├── papers/                       # Reference material
├── EDA_car.ipynb                 # Dataset exploration
├── SSD_explained.md              # High-level SSD explanation
├── v1_vs_v2.ipynb                # Version comparison notebook
├── SSDInt8_ONNX_Pred.py          # ONNX int8 predictor
├── SSDInt8_ONNX_Pred_v2.py       # ONNX int8 predictor for v2 workflow
├── SSD_int8_realtime_video.py    # Live webcam demo script using v1 model
├── SSD_int8_realtime_video_v2.py # Live webcam demo script using v2 model
├── SSD_video_predict.py          # Video-file inference utility
├── GettingStarted.md             # Setup and usage notes
└── training.md                   # Training details, results, and plots
```

## Dataset

The models are trained on the Udacity self-driving car dataset, an automotive-focused detection dataset containing **29,800 images** and **194,539 labeled bounding boxes**. The object classes used in this project are automotive road-scene categories such as cars, trucks, pedestrians, bikers, and traffic lights.

The repository includes exploratory analysis and preprocessing notebooks:

- [`EDA_car.ipynb`](./EDA_car.ipynb) for exploratory data analysis
- [`v1/preprocess_car.ipynb`](./v1/preprocess_car.ipynb) for the preprocessing workflow

The preprocessing pipeline collapses traffic-light subclasses into a single class and prepares SSD-ready train/test data.

## Project Versions

### v1

[`v1/`](./v1) contains the original baseline implementation of the project. It includes:

- the original dataset wrapper
- the original SSD model implementation
- the baseline trainer
- the initial preprocessing and training notebooks

Use v1 if you want to inspect the first complete end-to-end version of the project or compare later design changes against the original code.

### v2

[`v2/`](./v2) contains the newer codebase. The main difference is that the project has been split into smaller modules for cleaner experimentation and easier comparison. Moreover, the package [`gen-nms-package`](https://github.com/ElliotBlackstone/gen-nms-package), which was created for this project, provides GIoU, DIoU, and CIoU based non-maximum suppression with C++/CUDA backends.  The usage of this package significantly reduces the runtime of NMS.

#### `v2/model_files/`

[`v2/model_files/`](./v2/model_files) contains the model-side code, including:

- SSD architecture definition
- default/prior box creation
- box encoding and decoding
- NMS variant selection utilities
- prediction visualization helpers

#### `v2/training_files/`

[`v2/training_files/`](./v2/training_files) contains the training and evaluation workflow, including:

- hard-negative-mining classification loss
- conditional IoU crop augmentation
- cosine learning-rate scheduling
- target building and matching
- training and test steps
- checkpoint save/load utilities
- profiling notebooks and scripts

Use v2 if you want the current, more modular version of the project.

## Training and Evaluation

The SSD detector is trained end-to-end in PyTorch using:

- localization and classification losses
- hard negative mining
- torchvision v2-based preprocessing and augmentation
- mAP@0.50 evaluation

For more details on the training method and results, see [`training.md`](./training.md).

For a higher-level walkthrough of the model, see [`SSD_explained.md`](./SSD_explained.md).

For setup and run instructions, see [`GettingStarted.md`](./GettingStarted.md).

## ONNX, Quantization, and Benchmarking

The repository now includes a fuller optimization workflow beyond baseline PyTorch training.

### PTQ / ONNX

[`PTQ_testing/`](./PTQ_testing) contains scripts and assets for:

- exporting SSD models to ONNX
- parity testing between PyTorch and ONNXRuntime
- post-training quantization to INT8
- evaluation of quantized models
- pipeline benchmarking

This directory is the main entry point for the ONNX and INT8 inference work in the project.

### Benchmarking

[`benchmarking/`](./benchmarking) contains utilities for profiling and benchmarking:

- DataLoader performance
- training-loop performance
- GPU timing utilities
- configuration sweeps for loader settings

These tools are useful for understanding where the project spends time during training and inference.

## Local Inference and Real-Time Video Detection

A major addition to the repository is support for **local inference** workflows.

The root directory now includes scripts for:

- **ONNX int8 image prediction**
  - [`SSDInt8_ONNX_Pred.py`](./SSDInt8_ONNX_Pred.py)
  - [`SSDInt8_ONNX_Pred_v2.py`](./SSDInt8_ONNX_Pred_v2.py)
- **real-time webcam/video-stream detection**
  - [`SSD_int8_realtime_video.py`](./SSD_int8_realtime_video.py)
  - [`SSD_int8_realtime_video_v2.py`](./SSD_int8_realtime_video_v2.py)
- **video-file inference**
  - [`SSD_video_predict.py`](./SSD_video_predict.py)

These scripts shift the project beyond offline notebook evaluation and toward practical local deployment.

## Earlier Web App

The repository still contains the earlier FastAPI + Docker web app in [`app_files/`](./app_files), including the Dockerfile, model-serving app, and related assets. That code documents the original server-hosted deployment based on Google Cloud Run.

The repository now also emphasizes **local inference**, **ONNX/PTQ**, and **real-time video detection**. A similar demo is also available as a static GitHub Pages site [here](https://elliotblackstone.github.io/automotive-ssd-object-detection/). Unlike the earlier Cloud Run deployment, the GitHub Pages version performs inference directly in the browser using ONNX Runtime Web, so no backend model server is required.  This means uploaded images are processed client-side, and performance may vary depending on the user’s browser and hardware.

## Comparison Notebooks

The repository contains several notebooks that are useful for understanding how the project evolved:

- [`SSD_model_compare.ipynb`](./SSD_model_compare.ipynb)
- [`v1_vs_v2.ipynb`](./v1_vs_v2.ipynb)
- [`SSD_inference_profiling.ipynb`](./SSD_inference_profiling.ipynb)
- [`int8_model_testing.ipynb`](./int8_model_testing.ipynb)

These are the best places to document detailed experiment results, plots, and version-to-version comparisons without overloading the top-level README.

## What Changed in This Repository

Compared with the earlier form of the project, the repository now clearly includes:

- a legacy **v1** implementation and a newer **v2** implementation
- a more modular training and model structure in v2
- ONNX and PTQ tooling
- benchmarking/profiling utilities
- local real-time video detection scripts
- the earlier web app preserved as part of the project history

## Getting Started

Start with [`GettingStarted.md`](./GettingStarted.md).

Then choose the workflow that matches what you want to do:

- inspect the original baseline in [`v1/`](./v1)
- work with the newer modular code in [`v2/`](./v2)
- explore ONNX/PTQ in [`PTQ_testing/`](./PTQ_testing)
- test local inference with the real-time/video scripts in the repository root
