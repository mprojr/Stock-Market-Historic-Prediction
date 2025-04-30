# Stock Market Historical Prediction

A machine learning project for predicting stock market movements based on historical price data.

## Project Structure

```
Stock-Market-Historic-Prediction/
├── data/                     # Dataset files
│   ├── raw/                 # Original Kaggle dataset (not committed)
│   └── processed/           # Preprocessed data (e.g., CSV with features)
├── notebooks/                # Jupyter notebooks for analysis and modeling
│   ├── eda.ipynb            # Exploratory data analysis
│   └── baseline_model.ipynb # Baseline model implementation
├── src/                      # Python scripts for reusable code
│   ├── preprocess.py        # Data preprocessing and feature engineering
│   ├── models.py            # Machine learning model definitions
│   └── train.py             # Training and evaluation scripts
├── report/                   # Report drafts and templates
│   ├── literature_review.md # Draft for literature review
│   ├── report.tex           # LaTeX file for final report
│   └── figures/             # Figures for results (e.g., loss curves)
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore               # Ignore unnecessary files
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Stock-Market-Historic-Prediction.git
   cd Stock-Market-Historic-Prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place your raw stock market data in the `data/raw/` directory. The data should be in CSV format with at least the following columns:
   - Date
   - Open
   - High
   - Low
   - Close
   - Volume

2. Run the preprocessing script to generate features:
   ```
   python src/preprocess.py --input data/raw/your_data.csv
   ```

## Running the Models

### Training a Model

To train a model on the preprocessed data:

```
python src/train.py --input data/processed/your_processed_data.csv --model-type rf
```

Available model types:
- `linear`: Linear Regression
- `ridge`: Ridge Regression
- `lasso`: Lasso Regression
- `elasticnet`: ElasticNet Regression
- `rf`: Random Forest
- `gb`: Gradient Boosting
- `svm`: Support Vector Machine

### Exploratory Analysis

Explore the data and model performance using the Jupyter notebooks:

```
jupyter notebook notebooks/eda.ipynb
```

## Model Commands

> **Note**: In all commands below, `AAWW.csv` can be replaced with any stock data CSV file you've placed in the `data/raw/` directory. The CSV file should contain at least: Date, Open, High, Low, Close, and Volume columns.
> 
> Example stock files:
> - AAWW.csv (Atlas Air Worldwide Holdings)
> - AAPL.csv (Apple Inc.)
> - GOOGL.csv (Alphabet Inc.)
> - Any other stock data in proper CSV format

### Traditional Models
1. Random Forest:
```bash
# Basic usage
python src/train.py --input data/raw/AAWW.csv --model-type rf --output-dir report

# With parameters
python src/train.py --input data/raw/AAWW.csv --model-type rf --max-depth 5 --min-samples-split 5 --n-estimators 50
```

2. Gradient Boosting:
```bash
# Basic usage
python src/train.py --input data/raw/AAWW.csv --model-type gb --output-dir report

# With parameters
python src/train.py --input data/raw/AAWW.csv --model-type gb --max-depth 3 --n-estimators 100
```

3. Linear Models:
```bash
# Linear Regression
python src/train.py --input data/raw/AAWW.csv --model-type linear --output-dir report

# Ridge Regression
python src/train.py --input data/raw/AAWW.csv --model-type ridge --output-dir report

# Lasso Regression
python src/train.py --input data/raw/AAWW.csv --model-type lasso --output-dir report

# ElasticNet Regression
python src/train.py --input data/raw/AAWW.csv --model-type elasticnet --output-dir report
```

4. Support Vector Machine:
```bash
python src/train.py --input data/raw/AAWW.csv --model-type svm --output-dir report
```

### LSTM Model
```bash
# Basic usage
python src/models/lstm_model.py --csv data/raw/AAWW.csv

# With custom parameters
python src/models/lstm_model.py --csv data/raw/AAWW.csv --time_steps 10 --epochs 50 --batch_size 32

# For longer training
python src/models/lstm_model.py --csv data/raw/AAWW.csv --time_steps 20 --epochs 100 --batch_size 64
```

### Additional Options for All Models
```bash
# Disable visualizations
python src/train.py --input data/raw/AAWW.csv --model-type rf --no-visualize

# Disable model saving
python src/train.py --input data/raw/AAWW.csv --model-type rf --no-save-model

# Disable results saving
python src/train.py --input data/raw/AAWW.csv --model-type rf --no-save-results

# Disable volume profile analysis
python src/train.py --input data/raw/AAWW.csv --model-type rf --no-volume-profile

# Change forecast horizon
python src/train.py --input data/raw/AAWW.csv --model-type rf --horizon 5
```

### Output Analysis
All models will generate:
- Model performance metrics
- Overfitting/underfitting analysis
- Computational complexity analysis (training time, memory usage, etc.)
- Visualizations in the `report` directory:
  - Actual vs Predicted plots
  - Technical indicators
  - Training history (for LSTM)
  - Model-specific analysis

### Analysis Commands
1. Model Fitting Analysis (Overfitting/Underfitting):
```bash
# For traditional models
python src/train.py --input data/raw/AAWW.csv --model-type rf --output-dir report

# For LSTM
python src/models/lstm_model.py --csv data/raw/AAWW.csv
```

2. Ablation Studies:
```bash
# For traditional models
python src/train.py --input data/raw/AAWW.csv --model-type modelName --ablation

# For LSTM
python src/models/lstm_model.py --csv data/raw/AAWW.csv --ablation
```

Note: Replace `modelName` with any of: `rf`, `gb`, `linear`, `ridge`, `lasso`, `elasticnet`, or `svm`

## Project Status

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing datasets
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [pandas](https://pandas.pydata.org/) for data processing
