import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re

# Function to convert string with numbers to list of floats
def convert_to_float_list(num_str):
    num_str = num_str.strip('[]')  # Remove the surrounding brackets
    num_str = re.sub(r"[^\d.,-]", "", num_str)  # Remove any characters that are not digits, commas, dots, or hyphens
    num_list = [round(float(num.strip()), 2) for num in num_str.split(',')]
    return num_list

# Function to clean and convert atom site type symbol
def clean_atom_site_type_symbol(symbol_str):
    symbol_str = symbol_str.strip('[]')  # Remove the surrounding brackets
    symbol_list = [s.strip().strip("'") for s in symbol_str.split(',')]
    return symbol_list

# Load data from CSV
df = pd.read_csv('led_processed_data.csv')

# Clean and convert list strings
df['Updated_x'] = df['Updated_x'].apply(lambda x: convert_to_float_list(str(x)))
df['Updated_y'] = df['Updated_y'].apply(lambda x: convert_to_float_list(str(x)))
df['Updated_z'] = df['Updated_z'].apply(lambda x: convert_to_float_list(str(x)))
df['Real_x'] = df['Real_x'].apply(lambda x: convert_to_float_list(str(x)))
df['Real_y'] = df['Real_y'].apply(lambda x: convert_to_float_list(str(x)))
df['Real_z'] = df['Real_z'].apply(lambda x: convert_to_float_list(str(x)))
df['Atom_site_type_symbol'] = df['Atom_site_type_symbol'].apply(lambda x: clean_atom_site_type_symbol(str(x)))

# Add num_atoms column
df['num_atoms'] = df['Atom_site_type_symbol'].apply(lambda x: len(x))

# Flatten the lists to create individual rows for each atom
flattened_data = []
for i in range(len(df)):
    for j in range(len(df['Updated_x'][i])):
        flattened_data.append([
            df['Updated_x'][i][j],
            df['Updated_y'][i][j],
            df['Updated_z'][i][j],
            df['Real_x'][i][j % len(df['Real_x'][i])],
            df['Real_y'][i][j % len(df['Real_y'][i])],
            df['Real_z'][i][j % len(df['Real_z'][i])],
            df['Atom_site_type_symbol'][i][j % len(df['Atom_site_type_symbol'][i])]
        ])

flattened_df = pd.DataFrame(flattened_data, columns=['Updated_x', 'Updated_y', 'Updated_z', 'Real_x', 'Real_y', 'Real_z', 'Atom_site_type_symbol'])

# Encode the atom types
le = LabelEncoder()
flattened_df['Atom_site_type_symbol'] = le.fit_transform(flattened_df['Atom_site_type_symbol'])

# Features and targets
X = flattened_df[['Updated_x', 'Updated_y', 'Updated_z', 'Atom_site_type_symbol']]
y = flattened_df[['Real_x', 'Real_y', 'Real_z']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
