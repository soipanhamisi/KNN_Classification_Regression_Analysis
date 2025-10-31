
# Machine Learning Assignment 2 - KNN Analysis

This repository contains a comprehensive analysis using K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks.

## 📋 Project Overview

This assignment demonstrates the application of KNN algorithm on two distinct datasets:
1. **Medical No-Show Appointments** (Classification Task)
2. **Concrete Compressive Strength** (Regression Task)

## 🗂️ Project Structure

.  
├── notebooks/  
│   └── `assignment_2.ipynb`  
├── `README.md`  
├── `requirements.txt`    
└── `.gitignore`

## 📊 Datasets

### 1. No-Show Appointments Dataset (Kaggle)
- **Source**: Kaggle - `joniarroba/noshowappointments`
- **Task**: Binary classification (predict patient no-show)
- **Features**: Patient demographics, appointment details, medical conditions
- **Target**: No-show status (Yes/No)

### 2. Concrete Compressive Strength Dataset (UCI)
- **Source**: UCI Machine Learning Repository (ID: 165)
- **Task**: Regression (predict compressive strength)
- **Features**: Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age
- **Target**: Compressive Strength (MPa)

## 🔧 Installation

### Prerequisites
- Python 3.12.6
- virtualenv

### Setup

1. Clone the repository:
```

bash git clone <repository-url> cd <repository-name>``` 

2. Create and activate virtual environment:
```

bash python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate``` 

3. Install dependencies:
```

bash pip install -r requirements.txt``` 

## 🚀 Usage

1. Launch Jupyter Notebook:
```

bash jupyter notebook``` 

2. Open `notebooks/assignment_2.ipynb`

3. Run all cells sequentially

⚠️ **Note**: The hyperparameter tuning cells may take several minutes to complete.

## 📈 Analysis Workflow

### Part 1: No-Show Appointments Classification

1. **Data Loading & Exploration**
   - Load dataset from Kaggle
   - Initial data exploration (head, info, describe)

2. **Data Preprocessing**
   - Remove irrelevant features (PatientId, AppointmentID)
   - Encode categorical variables (Gender, No-show)
   - Convert date columns to numeric format

3. **Feature Engineering**
   - Create `WaitingDays` (duration between schedule and appointment)
   - Extract temporal features (day of week, month)
   - Add interaction features (age × hypertension)
   - Create boolean features (is_weekend, is_friday)
   - Categorize age into groups (child, teen, young_adult, adult, middle_aged, senior)
   - One-hot encode age groups
   - Encode neighborhoods (80 unique locations)

4. **Model Training & Evaluation**
   - Train KNN classifier with various k values (1-100)
   - Evaluate model performance on test set
   - Visualize accuracy vs. k neighbors

### Part 2: Concrete Strength Regression

1. **Data Loading & Quality Check**
   - Load dataset from UCI repository
   - Check for missing values and duplicates

2. **Exploratory Data Analysis**
   - Distribution analysis of all features
   - Correlation matrix visualization
   - Identify key predictors

3. **Data Preprocessing**
   - Train-test split (80/20)
   - Feature scaling using StandardScaler

4. **Model Development**
   - Train KNN regressor (initial k=5)
   - Calculate evaluation metrics (MSE, RMSE, MAE, R²)

5. **Hyperparameter Tuning**
   - Test k values from 1 to 30
   - Compare training vs. testing R² scores
   - Select optimal k value

6. **Final Model Evaluation**
   - Train final model with optimal k
   - Generate prediction plots (Actual vs. Predicted)
   - Analyze residual plots

## 📊 Key Results

### KNN Classification (No-Show Appointments)
- **Task**: Binary classification
- **Best Accuracy**: ~80% (varies with k)
- **Key Findings**: 
  - 37% of records showed negative waiting times (data quality issue)
  - Temporal features (day of week, waiting time) are important predictors
  - Age-hypertension interaction provides useful signal

### KNN Regression (Concrete Strength)
- **Task**: Regression
- **Best R² Score**: ~0.87 (optimal k varies, typically 5-15)
- **Key Findings**:
  - Cement and Age are strongest predictors
  - Feature scaling significantly improves performance
  - Model shows good predictive capability with minimal residual bias

## 🔑 Key Insights

1. **Algorithm Versatility**: KNN performs well for both classification and regression tasks
2. **Feature Engineering Impact**: Derived features significantly improve model performance
3. **Scaling Importance**: Feature scaling is crucial for KNN, especially in regression
4. **Hyperparameter Sensitivity**: Optimal k varies between datasets and tasks
5. **Interpretability**: KNN's instance-based nature provides good model interpretability
6. **Computational Cost**: Performance scales with dataset size (trade-off to consider)

## 📦 Dependencies

Key libraries used:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- kagglehub
- ucimlrepo
- tqdm

See `requirements.txt` for complete list.

## 👤 Authors
Hamisi  
Isadore

## 📝 License

This project is for educational purposes.

