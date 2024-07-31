# README

## Overview

This script is designed to preprocess atomic position data from a CSV file containing predicted outputs. It extracts and cleans the data, aligns it with real atomic positions, and saves the processed data to a new CSV file for further analysis or model training.

## Requirements

Ensure you have the following Python libraries installed:

- pandas
- numpy
- re (Regular Expression operations, part of the Python Standard Library)

You can install the required libraries using pip:

```sh
pip install pandas numpy
```

## Data Loading and Preprocessing

### 1. Load Data

The script reads data from a CSV file named `test-final-100-ledoutput2.csv`, which contains columns for predicted outputs and real atomic positions.

### 2. Extract Columns

The following columns are extracted from the dataset:
- `predicted_output-led`: The predicted atomic positions as strings.
- `atom_site_fract_x`, `atom_site_fract_y`, `atom_site_fract_z`: The real atomic positions.
- `atom_site_type_symbol`: The type of atoms.

### 3. Convert String to Float Lists

A function `convert_to_float_list` is defined to convert strings containing numbers into lists of floats by:
- Removing surrounding brackets.
- Splitting the string by commas and converting each element to a float.

### 4. Replace Specific Float Values

A function `replace_specific_floats` is defined to replace specific float values (0.5, 0.75, 0.25) in the lists if necessary. This can be customized if more replacements are needed.

### 5. Process Predicted Outputs

The predicted outputs are processed as follows:
- Extract individual lists from the predicted output strings using regex.
- Ensure each item contains exactly three parts (for x, y, and z coordinates).
- Convert the extracted lists to floats and replace specific values as necessary.

### 6. Align with Real Atomic Positions

The processed predicted coordinates are aligned with the real atomic positions and atom type symbols from the dataset.

### 7. Create Processed DataFrame

The processed data is stored in a new DataFrame with the following columns:
- `Updated_x`: Processed predicted x coordinates.
- `Updated_y`: Processed predicted y coordinates.
- `Updated_z`: Processed predicted z coordinates.
- `Real_x`: Real x coordinates.
- `Real_y`: Real y coordinates.
- `Real_z`: Real z coordinates.
- `Atom_site_type_symbol`: Atom type symbols.

### 8. Save Processed Data

The processed DataFrame is saved to a new CSV file named `test-final-100-ledoutput_process.csv`.

## Usage

1. Place the script in the same directory as your data file (`test-final-100-ledoutput2.csv`).
2. Run the script:

```sh
python script_name.py
```

3. The script will process the data and save it to `test-final-100-ledoutput_process.csv`.

## Output

The script prints the following:
- Any items skipped due to incorrect format or errors during processing.
- Confirmation that the processed data has been saved to `test-final-100-ledoutput_process.csv`.

## Notes

- Ensure your data file is correctly formatted and located in the right directory.
- Customize the `replace_specific_floats` function if additional specific replacements are needed.
