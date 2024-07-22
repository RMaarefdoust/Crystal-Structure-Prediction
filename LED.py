import torch
import torch.nn as nn
from transformers import LEDTokenizer, LEDForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
import gc

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
torch.cuda.empty_cache()

# Define the dataset class
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=1024):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_text = self.sequences[idx]
        output_text = self.labels[idx]

        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        label_encoding = self.tokenizer(output_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": label_encoding["input_ids"].flatten(),
        }

# Read the CSV file
df = pd.read_csv('Train-5000.csv')
column1_lists = df.iloc[:, 0].str.strip("[]").str.replace("'", "").str.split(',').apply(lambda x: [item.strip() for item in x if pd.notna(item)] if isinstance(x, list) else []).tolist()

column2 = df.iloc[:, 1].fillna('').str.strip("[]").str.replace("'", "").str.split(',').apply(lambda x: [float(item.strip()) if item.strip() != '' else None for item in x]).tolist()
column3 = df.iloc[:, 2].fillna('').str.strip("[]").str.replace("'", "").str.split(',').apply(lambda x: [float(item.strip()) if item.strip() != '' else None for item in x]).tolist()
column4 = df.iloc[:, 3].fillna('').str.strip("[]").str.replace("'", "").str.split(',').apply(lambda x: [float(item.strip()) if item.strip() != '' else None for item in x]).tolist()

data = [{'X': x, 'Y': y, 'Z': z, 'W': w} for x, y, z, w in zip(column1_lists, column2, column3, column4)]

# Shuffle the data
random.shuffle(data)
# Split data into train and test sets (e.g., 80% train, 20% test)
train_size = int(0.8 * len(data))
train_data, valid_data = data[:train_size], data[train_size:]

# Extract sequences and labels from train data
train_sequences = [' '.join(d['X']) for d in train_data]
train_labels = [' '.join(map(str, [d['Y'], d['Z'], d['W']])) for d in train_data]

# Extract sequences and labels from validation data
valid_sequences = [' '.join(d['X']) for d in valid_data]
valid_labels = [' '.join(map(str, [d['Y'], d['Z'], d['W']])) for d in valid_data]

# Initialize the LED model and tokenizer
tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')

# Prepare the train dataset and dataloader
train_dataset = SequenceDataset(train_sequences, train_labels, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduce batch size to 4

# Prepare the validation dataset and dataloader
valid_dataset = SequenceDataset(valid_sequences, valid_labels, tokenizer)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)  # Reduce batch size to 4

# Training loop
model.to(device)  # Move model to appropriate device

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

train_losses = []
valid_losses = []

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    train_losses.append(average_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}")

    # Validation
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_valid_loss += loss.item()

    average_valid_loss = total_valid_loss / len(valid_dataloader)
    valid_losses.append(average_valid_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_valid_loss:.4f}")

    # Step the scheduler
    scheduler.step()

# Save the trained model
model.save_pretrained("custom_led_sequence_model-2")
tokenizer.save_pretrained("custom_led_sequence_model-2")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()

# Set x-ticks to be integers
epochs = range(1, len(train_losses) + 1)
plt.xticks(epochs)

# Save the plot
plt.savefig('loss_plot.png')
plt.savefig('loss_plot.pdf', format='pdf')
plt.show()
