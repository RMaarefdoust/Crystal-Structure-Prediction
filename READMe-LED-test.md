# Custom LED Model for Text Generation

This repository contains code to use a pre-trained LED (Longformer Encoder-Decoder) model for text generation. The code loads a pre-trained model, processes input text from a CSV file, generates predictions, and saves the results back to a CSV file.

## Overview

The script performs the following tasks:

1. **Model Initialization**: Loads a pre-trained LED model and tokenizer.
2. **Prediction Generation**: Uses the model to generate predictions from input text.
3. **Data Processing**: Reads input data from a CSV file, generates predictions, and saves the results to a new CSV file.

## Requirements

- `torch`
- `transformers`
- `pandas`

You can install these dependencies using pip:

```bash
pip install torch transformers pandas
```

## Usage

1. **Prepare the Data**:
   Ensure you have a CSV file named `test-final-100.csv` in the same directory as the script. The CSV file should have a column named `atom_site_type_symbol` containing the input text for which you want to generate predictions.

2. **Model Path**:
   Make sure you have a pre-trained LED model saved locally at `custom_led_sequence_model-2`. If not, you can change the `model_path` variable to point to the correct location of your pre-trained model.

3. **Run the Script**:
   Execute the script using Python:

   ```bash
   python predict_led_model.py
   ```

   The script will:

   - Load the pre-trained LED model and tokenizer.
   - Process each row in the CSV file and generate predictions.
   - Save the predictions to a new CSV file named `test-final-100-ledoutput2.csv`.

## Code Details

### CustomLED Class

The `CustomLED` class extends `torch.nn.Module` and is used to:

- Initialize the LED model and tokenizer from the provided model path.
- Generate predictions for input text.

### Prediction Process

- The script reads input data from `test-final-100.csv`.
- For each row, it generates a prediction using the `CustomLED` model.
- The predictions are appended to the DataFrame and saved to `test-final-100-ledoutput2.csv`.

## Example

Given an input CSV with:

| atom_site_type_symbol |
|-----------------------|
| example input text    |
| another example text  |

The script will output a CSV with:

| atom_site_type_symbol | predicted_output-led   |
|-----------------------|-------------------------|
| example input text    | predicted output text  |
| another example text  | another predicted text |

