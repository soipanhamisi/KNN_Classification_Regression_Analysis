# KNN Classification & Regression Analysis

A machine learning project analyzing the No-Show Appointments dataset from Kaggle using K-Nearest Neighbors algorithms to predict patient appointment attendance.

## 📋 Overview

This project uses the "No Show Appointments" dataset to build and evaluate KNN models for predicting whether patients will miss their scheduled medical appointments. The analysis includes comprehensive data preprocessing, feature engineering, label encoding, and model implementation with hyperparameter tuning.

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/soipanhamisi/KNN_Classification_Regression_Analysis.git
   cd KNN_Classification_Regression_Analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis**
   - Open `notebooks/assignment_2.ipynb` in Jupyter Notebook or your IDE
   - Run cells sequentially
   - The dataset will be automatically downloaded from Kaggle on first run

## 📊 Dataset

**Source:** [No Show Appointments Dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

The dataset contains medical appointment records with patient demographics, appointment details, and no-show outcomes.

### Dataset Features

The analysis works with the following features:

- **Gender**: Encoded as binary (0=Male, 1=Female)
- **Age**: Patient age
- **Neighbourhood**: 80 unique neighborhoods (label encoded 0-79)
- **Scholarship**: Whether patient has scholarship (0/1)
- **Hipertension**: Hypertension status (0/1)
- **Diabetes**: Diabetes status (0/1)
- **Alcoholism**: Alcoholism status (0/1)
- **Handcap**: Handicap level (0-4)
- **SMS_received**: Whether SMS reminder was sent (0/1)
- **WaitingDays**: Days between scheduling and appointment
- **appointment_month**: Month of appointment (4-6)
- **day_of_week_encoded**: Day of week (0=Monday to 6=Sunday)
- **No-show**: Target variable (0=Attended, 1=No-show)

### Data Preprocessing

1. **Removed Features**: PatientId, AppointmentID (non-predictive identifiers)
2. **Binary Encoding**: Gender and No-show columns
3. **Label Encoding**: 80 unique neighborhoods mapped to integer IDs
4. **Feature Engineering**:
   - Calculated WaitingDays from appointment scheduling
   - Extracted appointment day of week and month
   - Removed negative waiting times (assumed data collection errors)
5. **Temporal Features**: Day of week and month encoded

## 🤖 Model Performance

- **Algorithm**: K-Nearest Neighbors (KNN) Classifier
- **Initial Model** (k=5): ~66.6% accuracy
- **Hyperparameter Tuning**: Tested k values from 1 to 99
- **Best Performance**: ~71.5% accuracy (achieved with higher k values)

### Key Findings

- The model shows moderate predictive capability for appointment no-shows
- Performance plateaus around 71-72% accuracy
- Higher k values (around 20-30+) provide better generalization

### Neighbourhood Lookup Table

The model uses 80 neighborhoods from Vitória, Brazil, each assigned a unique ID:
- ID 0: AEROPORTO
- ID 1: ANDORINHAS
- ... (see notebook for complete mapping)
- ID 79: VILA RUBIM

---

**Note:** This is an educational project focused on KNN algorithms for classification tasks. The model demonstrates the importance of feature engineering and hyperparameter tuning in machine learning workflows.