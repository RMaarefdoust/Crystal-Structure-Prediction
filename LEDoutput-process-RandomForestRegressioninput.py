import re
import pandas as pd
import numpy as np

# Load data from CSV
data = pd.read_csv("test-lstm-led_8000.csv")
datalist = data["predicted_output-led"].tolist()

# Ensure y_test matches the shape of x_test
Real_x = np.array(data['atom_site_fract_x'].tolist())
Real_y = np.array(data['atom_site_fract_y'].tolist())
Real_z = np.array(data['atom_site_fract_z'].tolist())
Atom_site_type_symbol = np.array(data['atom_site_type_symbol'].tolist())

# Function to convert string with numbers to list of floats
def convert_to_float_list(num_str):
    num_str = num_str.strip('[]')  # Remove the surrounding brackets
    num_list = [float(num.strip()) for num in num_str.split(',')]
    return num_list

# Function to replace specific float values
def replace_specific_floats(num_list):
    replaced_list = []
    for num in num_list:
        if num == 0.5:
            replaced_list.append(0.5)
        elif num == 0.75:
            replaced_list.append(0.75)
        elif num == 0.25:
            replaced_list.append(0.25)
        else:
            replaced_list.append(num)
    return replaced_list

# Lists to store processed data
processed_data = []

for item_index, item in enumerate(datalist):
    try:
        # Extract individual lists using regex
        matches = re.findall(r'\[.*?\]', item)
        
        # Check if matches contain exactly 3 parts, otherwise skip this item
        if len(matches) != 3:
            print("Skipping item due to incorrect format:", item)
            continue

        # Convert each extracted list to floats
        x = convert_to_float_list(matches[0])
        y = convert_to_float_list(matches[1])
        z = convert_to_float_list(matches[2])

        # Replace specific floats in each list if necessary
        x = replace_specific_floats(x)
        y = replace_specific_floats(y)
        z = replace_specific_floats(z)

        # Append to processed data list
        processed_data.append({
            'Updated_x': x,
            'Updated_y': y,
            'Updated_z': z,
            'Real_x': Real_x[item_index],
            'Real_y': Real_y[item_index],
            'Real_z': Real_z[item_index],
            'Atom_site_type_symbol': Atom_site_type_symbol[item_index]
        })

    except Exception as e:
        print("Skipping item due to an error:", e)
        continue

# Convert processed data to DataFrame
processed_df = pd.DataFrame(processed_data)

# Save processed data to a new CSV file
processed_df.to_csv('test-lstm-led_8000_processed.csv', index=False)

print("Processed data saved to 'test-lstm-led_8000_processed.csv'")

