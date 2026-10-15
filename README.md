
# Network Anomaly Detection System

A machine learning project that analyzes network traffic and classifies records as Normal or Attack using the NSL-KDD dataset.

## Project Overview

This project compares four machine learning approaches for network anomaly detection:

- **Random Forest:** supervised classification.
- **Isolation Forest:** unsupervised anomaly detection.
- **One-Class SVM:** learns patterns from normal network traffic.
- **Autoencoder:** identifies unusual records using reconstruction error.

A Streamlit dashboard provides model performance comparisons, confusion matrices, charts, dataset exploration, and prediction demonstrations.

## Features

- Comparison of four trained machine learning models.
- Accuracy, precision, recall, and F1-score evaluation.
- Interactive confusion matrix.
- Model performance comparison chart.
- NSL-KDD test dataset exploration.
- Manual demonstration predictions.
- Predictions on complete NSL-KDD test records.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Streamlit
- NSL-KDD dataset

## Project Structure

```text
Network_Anomaly_Detection/
├── app.py
├── main.py
├── autoencoder.py
├── random_forest_model.pkl
├── isolation_forest_model.pkl
├── one_class_svm_model.pkl
├── models/
│   └── autoencoder_final.pkl
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Network_Anomaly_Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn joblib tensorflow
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Dataset

The project uses the NSL-KDD dataset, containing labeled network connection records. The test dataset contains 22,544 records in this project.

Each record contains 41 input features, with a separate target label and difficulty field in the original files.

## Evaluation

The dashboard calculates accuracy, attack precision, attack recall, and attack F1-score on the supplied NSL-KDD test dataset.

The reported metrics describe performance on this benchmark dataset. They should not be interpreted as guaranteed performance on live network traffic.

## Important Limitations

- The manual demo mode fills unspecified features from a sample training record. Its prediction is illustrative only.
- The complete-record mode evaluates records from the labeled NSL-KDD test dataset; it does not monitor live network traffic.
- Results depend on the saved model artifacts and preprocessing used during training.
- Benchmark performance may not generalize to modern network environments.

## Future Improvements

- Add CSV upload for batch prediction.
- Add more detailed attack-category analysis.
- Improve interactive visualizations.
- Support complete network-record input and validation.
- Explore live traffic integration in a controlled environment.

## Author

Computer Science and Engineering student project focused on machine learning and network anomaly detection.
