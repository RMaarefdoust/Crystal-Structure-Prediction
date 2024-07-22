# Material Crystal Structure Generation

## Overview

This project provides a framework for generating and refining material crystal structures using Natural Language Processing (NLP) and machine learning techniques. The workflow includes training an LED model, testing it, processing the output for Random Forest regression, and finally applying Random Forest regression to predict crystal structures.

## Repository Structure

The repository contains the following scripts:

1. **`LED-train.py`**: Script to train the Longformer-Encoder-Decoder (LED) model.
2. **`LED-test.py`**: Script to test the trained LED model and generate initial crystal structures.
3. **`LEDoutput-process-RandomForestRegressioninput.py`**: Script to process the LED model outputs to prepare them for Random Forest regression.
4. **`RandomForestRegression.py`**: Script to train a Random Forest Regressor on the processed data and predict final crystal structures.

## Installation

To run the scripts, you need to install the required Python packages. Use the following command to install them:

```bash
pip install pandas numpy scikit-learn transformers
```

## Usage

### 1. Training the LED Model

**`LED-train.py`**

This script trains the LED model on your dataset to generate initial crystal structures from atom descriptions.

Usage:

```bash
python LED-train.py
```

**Functionality**:
- Loads training data.
- Trains the LED model.
- Saves the trained model for later use.

### 2. Testing the LED Model

**`LED-test.py`**

This script tests the trained LED model and generates initial crystal structures based on test data.

Usage:

```bash
python LED-test.py
```

**Functionality**:
- Loads the trained LED model.
- Processes test data to generate crystal structures.
- Saves the generated structures for further processing.

### 3. Processing LED Output for Random Forest Regression

**`LEDoutput-process-RandomForestRegressioninput.py`**

This script processes the LED model outputs to prepare them for use in Random Forest regression.

Usage:

```bash
python LEDoutput-process-RandomForestRegressioninput.py
```

**Functionality**:
- Loads the LED-generated crystal structures.
- Cleans and converts data into a format suitable for Random Forest regression.
- Saves the processed data.

### 4. Training and Evaluating Random Forest Regressor

**`RandomForestRegression.py`**

This script trains a Random Forest Regressor on the processed data and evaluates its performance.

Usage:

```bash
python RandomForestRegression.py
```

**Functionality**:
- Loads the processed data from the previous step.
- Trains the Random Forest Regressor.
- Evaluates the model using metrics like Mean Squared Error (MSE).
- Prints evaluation results.

## Data

- **Training Data**: Used in `LED-train.py` to train the LED model.
- **Test Data**: Used in `LED-test.py` to generate crystal structures.
- **Processed Data**: Output from `LEDoutput-process-RandomForestRegressioninput.py`, used for training the Random Forest model.

## Results

The methodology aims to improve material crystal structure predictions by combining LED-based NLP techniques with Random Forest regression. The results demonstrate how well the model performs in generating and refining crystal structures.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.


