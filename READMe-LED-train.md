# Sequence-to-Sequence Model Training with LED

This repository contains code for training a sequence-to-sequence model using the [LED (Longformer Encoder-Decoder)](https://huggingface.co/allenai/led-base-16384) architecture from the Hugging Face Transformers library. The code is designed to handle a dataset of sequences and their associated numerical labels.

## Overview

The script performs the following tasks:

1. **Data Preparation**: Reads data from a CSV file, processes it into sequences and labels, and splits it into training and validation sets.
2. **Model Setup**: Initializes the LED model and tokenizer.
3. **Training**: Trains the model on the training dataset and evaluates its performance on the validation dataset.
4. **Evaluation**: Tracks and plots training and validation losses over epochs.
5. **Saving**: Saves the trained model and tokenizer, and saves a plot of the training and validation losses.

## Requirements

- `torch`
- `transformers`
- `pandas`
- `matplotlib`
- `tqdm`

You can install these dependencies using pip:

```bash
pip install torch transformers pandas matplotlib tqdm
```

## Usage

1. **Prepare the Data**:
   Ensure that you have a CSV file named `Train-5000.csv` in the same directory as the script. The CSV file should have four columns where:

   - Column 1: List of input sequences.
   - Column 2: Numerical labels (Y values).
   - Column 3: Numerical labels (Z values).
   - Column 4: Numerical labels (W values).

2. **Run the Script**:
   Execute the script using Python:

   ```bash
   python LED-train.py
   ```

   The script will:

   - Read and process the CSV data.
   - Train the LED model for 10 epochs.
   - Save the trained model and tokenizer.
   - Save a plot of the training and validation losses.

## Model & Tokenizer

The script uses the LED model and tokenizer from the Hugging Face Transformers library:

- **Model**: `allenai/led-base-16384`
- **Tokenizer**: `allenai/led-base-16384`

## Results

The script saves the following outputs:

- The trained model and tokenizer in the `custom_led_sequence_model-2` directory.
- A plot of training and validation losses as `loss_plot.png` and `loss_plot.pdf`.

## Example

Here is an example of how the input and output sequences might be structured:

**Input Sequence**: `["example", "input", "sequence"]`

**Output Labels**: `["1.0", "2.0", "3.0"]`

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
```
