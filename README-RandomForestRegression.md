# Random Forest Regression for Atom Site Data

## Overview

This script is designed for data preprocessing, model training, and evaluation using a Random Forest Regressor to predict atomic positions based on given features. The dataset comprises atomic coordinates and types, which are preprocessed and then used to train a machine learning model.

## Requirements

Ensure you have the following Python libraries installed:

- numpy
- pandas
- scikit-learn

You can install them using pip:

```sh
pip install numpy pandas scikit-learn
```

## Data Loading and Preprocessing

### 1. Load Data

The script reads data from two CSV files:

- `train-RFR-1000-ledoutput_process.csv`: Training dataset
- `test-final-100-ledoutput_process.csv`: Testing dataset

### 2. Clean and Convert Data

The script includes functions to clean and convert string representations of lists into actual Python lists of floats.

- **convert_to_float_list**: Converts a string of numbers into a list of floats.
- **clean_atom_site_type_symbol**: Cleans and converts atom site type symbols from string format.

These functions are applied to the relevant columns in both training and testing datasets.

### 3. Add `num_atoms` Column

A new column `num_atoms` is added to both datasets, representing the number of atoms based on the `Atom_site_type_symbol` column.

### 4. Flatten Data

The nested lists in the dataset are flattened to create individual rows for each atom. This process ensures that each atom's data is represented in a single row, making it suitable for model training.

### 5. Encode Categorical Data

The atom site type symbols are encoded using `LabelEncoder` to convert categorical data into numerical format.

## Model Training and Evaluation

### 1. Split Data

The data is split into training and testing sets using an 80-20 split.

### 2. Train Model

A `RandomForestRegressor` is trained with the following hyperparameters:

- `n_estimators=300`
- `random_state=42`

### 3. Make Predictions

Predictions are made on both the test set and an additional test dataset (`df_test`). The mean squared error (MSE) is calculated to evaluate the model's performance.

### 4. Hyperparameter Tuning

A grid search with cross-validation (`GridSearchCV`) is performed to find the best hyperparameters for the `RandomForestRegressor`. The parameters grid includes:

- `n_estimators`: [100, 200, 300]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

The best parameters are printed, and the model with the best parameters is evaluated again.

## Usage

1. Place the script in the same directory as your data files (`train-RFR-1000-ledoutput_process.csv` and `test-final-100-ledoutput_process.csv`).
2. Run the script:

```sh
python script_name.py
```

3. The script will output the mean squared error for the predictions and the best hyperparameters found by the grid search.

## Output

The script prints the following:

- The training and testing data after preprocessing.
- Mean squared error for initial model predictions.
- Best hyperparameters found by the grid search.
- Mean squared error for the model with the best hyperparameters.

## Notes

- Ensure your data files are correctly formatted and located in the right directory.
- Modify the parameter grid in the script if you want to experiment with different hyperparameters.

This README provides a comprehensive guide to understanding and running the script. Follow the steps carefully to ensure successful execution and accurate results.
