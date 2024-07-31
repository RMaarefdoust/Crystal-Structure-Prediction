import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import re
from sklearn.model_selection import GridSearchCV

# Function to convert string with numbers to list of floats
def convert_to_float_list(num_str):
    num_str = num_str.strip('[]')  # Remove the surrounding brackets
    num_str = re.sub(r"[^\d.,-]", "", num_str)  # Remove any characters that are not digits, commas, dots, or hyphens
    num_list = [float(num.strip()) for num in num_str.split(',')]
    return num_list

# Function to clean and convert atom site type symbol
def clean_atom_site_type_symbol(symbol_str):
    symbol_str = symbol_str.strip('[]')  # Remove the surrounding brackets
    symbol_list = [s.strip().strip("'") for s in symbol_str.split(',')]
    return symbol_list

# Load data from CSV
df = pd.read_csv('train-RFR-1000-ledoutput_process.csv')
df_test = pd.read_csv('test-final-100-ledoutput_process.csv')#'test-low-led.csv')

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

# Clean and convert list strings
df_test['Updated_x'] = df_test['Updated_x'].apply(lambda x: convert_to_float_list(str(x)))
df_test['Updated_y'] = df_test['Updated_y'].apply(lambda x: convert_to_float_list(str(x)))
df_test['Updated_z'] = df_test['Updated_z'].apply(lambda x: convert_to_float_list(str(x)))
df_test['Real_x'] = df_test['Real_x'].apply(lambda x: convert_to_float_list(str(x)))
df_test['Real_y'] = df_test['Real_y'].apply(lambda x: convert_to_float_list(str(x)))
df_test['Real_z'] = df_test['Real_z'].apply(lambda x: convert_to_float_list(str(x)))
df_test['Atom_site_type_symbol'] = df_test['Atom_site_type_symbol'].apply(lambda x: clean_atom_site_type_symbol(str(x)))

# Add num_atoms column
df_test['num_atoms'] = df_test['Atom_site_type_symbol'].apply(lambda x: len(x))

# Flatten the lists to create individual rows for each atom
flattened_data = []

for i in range(len(df)):
    for j in range(len(df['Real_x'][i])):
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


flattened_data_test = []
for i in range(0,len(df_test)):
    for j in range(0,len(df_test['Real_x'][i])):
        flattened_data_test.append([
            df_test['Updated_x'][i][j],
            df_test['Updated_y'][i][j],
            df_test['Updated_z'][i][j],
            df_test['Real_x'][i][j % len(df_test['Real_x'][i])],
            df_test['Real_y'][i][j % len(df_test['Real_y'][i])],
            df_test['Real_z'][i][j % len(df_test['Real_z'][i])],
            df_test['Atom_site_type_symbol'][i][j % len(df_test['Atom_site_type_symbol'][i])]
        ])

flattened_data_test = pd.DataFrame(flattened_data_test, columns=['Updated_x', 'Updated_y', 'Updated_z', 'Real_x', 'Real_y', 'Real_z', 'Atom_site_type_symbol'])

# Encode the atom types
le = LabelEncoder()
flattened_df['Atom_site_type_symbol'] = le.fit_transform(flattened_df['Atom_site_type_symbol'])
flattened_data_test['Atom_site_type_symbol'] = le.fit_transform(flattened_data_test['Atom_site_type_symbol'])

# Features and targets
X = flattened_df[['Updated_x', 'Updated_y', 'Updated_z', 'Atom_site_type_symbol']]
y = flattened_df[['Real_x', 'Real_y', 'Real_z']]

X_test_test = flattened_data_test[['Updated_x', 'Updated_y', 'Updated_z', 'Atom_site_type_symbol']]
y_test_test = flattened_data_test[['Real_x', 'Real_y', 'Real_z']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=300, random_state=42) 

model.fit(X_train, y_train)



# Predictions
y_pred = model.predict(X_test)
np.set_printoptions(precision=15)
print(X_test)
print(y_pred)
# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

y_pred_test=model.predict(X_test_test)
print(y_test_test)
print(y_pred_test)
# Model evaluation
mse_test = mean_squared_error(y_test_test, y_pred_test)
print(f"Mean Squared Error test: {mse}")


# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='neg_mean_squared_error',
                           verbose=1,
                           n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)
print(y_pred)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
