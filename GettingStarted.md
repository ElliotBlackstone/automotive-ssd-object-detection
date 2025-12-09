## Getting Started

This section describes how to set up a local environment, run the interactive web demo, and (optionally) reproduce the training pipeline.

#### 1. Clone the repository

```bash
git clone https://github.com/ElliotBlackstone/automotive-ssd-object-detection.git
cd automotive-ssd-object-detection
```


#### 2. Create and activate a Python enviornment
Use Python $\geq 3.10$.

```bash
conda create -n automotive-ssd python=3.10
conda activate automotive-ssd
```


#### 3. Install dependencies
All Python dependencies (training + web app) are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

#### 4. Run the web demo locally
The FastAPI/uvicorn app lives in `app_files/` and exposes the same interface as the deployed Cloud Run demo.

From the repository root:
```bash
cd app_files
uvicorn ssd_demo_app:app --host 0.0.0.0 --port 8000 --reload
```

By default the app listens on port `8080`. Open a browser at:
```bash
http://localhost:8080/
```
Upload a dashcam-style image to visualize SSD predictions (original image on the left, predictions overlaid on the right).

#### 5. (Optional) Prepare the dataset and run training
If you want to reproduce training rather than only use the pre-trained model:
1) Download the dataset [here](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset).
2) Preprocess into an SSD-ready dataset
   * Open `preprocess_car.ipynb` and update any paths pointing to your local copy of the dataset export and `_annotations.csv`.
   * Run the notebook to collapse traffic light subclasses into a single `trafficLight` class, create stratified train/test splits and write the corresponding images/CSV files into train/ and test/ folders with “empty” background images included in the corresponding .csv files.
3) Train the SSD model
   * Open `SSD_model_train.ipynb`.
   * Point the data-root paths to the preprocessed `train/` and `test/` folders.
   * Run the notebook to train the SSD model using the augmentation, optimizer, and scheduler settings described in the Model training section.
   * Results can be inspected at the end of that notebook. Pre-trained models can be loaded there as well.