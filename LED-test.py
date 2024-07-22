import torch
import pandas as pd
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLED(nn.Module):
    def __init__(self, model_path):
        super(CustomLED, self).__init__()
        self.led = LEDForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = LEDTokenizer.from_pretrained(model_path)

    def forward(self, input_text):
        input_encoding = self.tokenizer(input_text, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = input_encoding["input_ids"].to(device)
        attention_mask = input_encoding["attention_mask"].to(device)

        # Generate predictions
        with torch.no_grad():
            outputs = self.led.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024, num_beams=1)
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output

# Read the CSV file for test data
df = pd.read_csv('test-final-100.csv')

# Instantiate the custom LED model using the saved path
model_path = "custom_led_sequence_model-2"  
model = CustomLED(model_path).to(device)

# Process each row in the CSV
predictions = []
for index, row in df.iterrows():
    input_text = row['atom_site_type_symbol']
    predicted_output = model(input_text)
    predictions.append(predicted_output)

# Add the predictions to the DataFrame
df['predicted_output-led'] = predictions

# Save the DataFrame to a new CSV file
df.to_csv('test-final-100-ledoutput2.csv', index=False)

print("Predictions have been saved to 'test-final-100-ledoutput-2.csv'")
