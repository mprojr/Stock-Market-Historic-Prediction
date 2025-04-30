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

## Project Status

This project is currently in development. Future improvements include:
- Adding more advanced deep learning models (LSTM, Transformer)
- Incorporating sentiment analysis from news and social media
- Developing a backtesting framework for trading strategies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing datasets
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
- [pandas](https://pandas.pydata.org/) for data processing

## Commands
1) python src/train.py --input data/raw/AAWW.csv --model-type modelName --ablation : Running ablation for traditional models
2) python src/models/lstm_model.py --csv data/raw/AAWW.csv --ablation : Running ablation for lstm model