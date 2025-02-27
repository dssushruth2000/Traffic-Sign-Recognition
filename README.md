# Traffic Sign Classification using CNNs

This project classifies German traffic signs using three deep learning models:

### Models Used:
- **Model 1:** MobileNetV2-based Model
- **Model 2:** Custom CNN with Dropout and Batch Normalization
- **Model 3:** Deeper Custom CNN with Batch Normalization and Dropout

The models are trained and evaluated using the **GTSRB dataset** (German Traffic Sign Recognition Benchmark).

## Setup Instructions

### 1. Install Dependencies
Ensure the following libraries are installed. You can install them using pip:

```sh
pip install tensorflow numpy pandas opencv-python matplotlib seaborn scikit-learn pillow
```

### 2. Run Jupyter Notebook
To open Jupyter Notebook, run this in your terminal or command prompt:

```sh
jupyter notebook
```

## How to Run the Code in Jupyter Notebook:

1. Make sure the repository contains the dataset (**gtsrb**).
2. Open the Jupyter Notebook and run **Final_Project_Traffic.ipynb**.
3. Run the cells sequentially:
   - Preprocess and normalize images.
   - Define, train, and evaluate three models (**Model 1, Model 2, Model 3**).
   - Generate performance metrics such as:
     - Training and Validation Metrics (accuracy/loss plots).
     - Confusion Matrices for all models.
     - ROC Curves for model comparison.
   - Visualize test predictions with actual and predicted labels.
   - Trained models will be saved in the **Model/** directory.

## Dependencies
Ensure the following Python libraries are installed:

- TensorFlow (2.x)
- NumPy
- Pandas
- OpenCV (opencv-python)
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow (PIL)
