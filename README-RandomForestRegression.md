# Random Forest Regression for Atom Site Data


This repository contains code to process and analyze atom site data using a Random Forest Regressor. The script loads, cleans, and transforms data, trains a model, and evaluates its performance.

## Overview

The script performs the following tasks:

1. **Data Loading**: Reads a CSV file containing processed atom site data.
2. **Data Cleaning and Transformation**: Converts string representations of lists to numeric values and cleans symbolic data.
3. **Data Flattening**: Expands lists into individual rows for each atom.
4. **Feature Encoding**: Encodes categorical atom site type symbols.
5. **Model Training and Evaluation**: Trains a Random Forest Regressor on the data and evaluates its performance.

## Requirements

- `numpy`
- `pandas`
- `scikit-learn`
- `re` (Standard Python library)

You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn
```

## Usage

1. **Prepare the Data**:
   Ensure you have a CSV file named `led_processed_data.csv` in the same directory as the script. The CSV file should include the following columns:
   - `Updated_x`, `Updated_y`, `Updated_z`: Lists of predicted atom site coordinates.
   - `Real_x`, `Real_y`, `Real_z`: Lists of actual atom site coordinates.
   - `Atom_site_type_symbol`: List of atom site type symbols.

2. **Run the Script**:
   Execute the script using Python:

   ```bash
   python RandomForestRegression.py
   ```

   The script will:

   - Load and clean the data from `led_processed_data.csv`.
   - Flatten the data and encode categorical variables.
   - Split the data into training and testing sets.
   - Train a Random Forest Regressor model.
   - Print the Mean Squared Error (MSE) of the model's predictions.

## Code Details

### Functions

- **`convert_to_float_list(num_str)`**: Converts a string representation of a list of numbers to a list of floats.
- **`clean_atom_site_type_symbol(symbol_str)`**: Cleans and converts atom site type symbols into a list of strings.

### Data Processing

- **Data Cleaning**: Converts list strings to numeric values and cleans symbolic data.
- **Data Flattening**: Expands lists into individual rows for each atom, ensuring each atom's data is represented in separate rows.
- **Feature Encoding**: Encodes categorical atom site type symbols into numeric values for model training.

### Model Training

- **Features and Targets**: Defines features (`Updated_x`, `Updated_y`, `Updated_z`, `Atom_site_type_symbol`) and targets (`Real_x`, `Real_y`, `Real_z`).
- **Train-Test Split**: Splits the data into training and testing sets.
- **Model Training**: Trains a Random Forest Regressor with 100 estimators.
- **Model Evaluation**: Calculates and prints the Mean Squared Error (MSE) of the model's predictions.

## Example

Given an input CSV with:

| Updated_x  | Updated_y  | Updated_z  | Real_x    | Real_y    | Real_z    | Atom_site_type_symbol |
|------------|------------|------------|-----------|-----------|-----------|-----------------------|
| "[1.0, 0.5]" | "[0.8, 0.4]" | "[0.6, 0.3]" | "[1.0, 0.5]" | "[0.8, 0.4]" | "[0.6, 0.3]" | "[A, B]" |

The script will:

1. Clean and convert data.
2. Flatten lists and encode categorical variables.
3. Train a Random Forest model.
4. Print the Mean Squared Error of the predictions.
