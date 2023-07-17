import torch
from torch.utils.data import random_split,DataLoader
from transformers import BertModel, BertTokenizer
from bertmodel import AspectDataset
from preprocessing import df

##TRAINING
# Load the pre-trained BERT tokenizer
tokenizer_aspect_names = BertTokenizer.from_pretrained('bert-base-german-cased')
tokenizer_aspect_values = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load the pre-trained BERT model
model = BertModel.from_pretrained('bert-base-german-cased')

#possible aspect names
aspect_names = ['Abteilung', 'Aktivität', 'Akzente','Anlass','Besonderheiten','Charakter', 'Charakter Familie',
                'Dämpfungsgrad', 'Erscheinungsjahr', 'EU-Schuhgröße' ,'Farbe','Futtermaterial','Gewebeart',
                'Herstellernummer', 'Herstellungsland und', 'Innensohlenmaterial' , 'Jahreszeit',
                'Laufsohlenmaterial', 'Marke','Maßeinheit','Modell','Muster','Obermaterial',
                'Produktart','Produktlinie','Schuhschaft-Typ','Schuhweite','Stil','Stollentyp',
                'Thema','UK-Schuhgröße','US-Schuhgröße','Verschluss', 'Zwischensohlen-Typ']

# Create the dataset
dataset = AspectDataset(df, tokenizer_aspect_names,tokenizer_aspect_values,aspect_names)

# Split your dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # Remaining 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


batch_size = 2  # Adjust the batch size as per your requirements

# Define the collate function for handling variable-length sequences
def collate_fn(batch):
    aspect_name_ids = [item['aspect_name_ids'] for item in batch]
    aspect_value_ids = [item['aspect_value_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    aspect_name_ids = torch.nn.utils.rnn.pad_sequence(aspect_name_ids, batch_first=True, padding_value=0)
    aspect_value_ids = torch.nn.utils.rnn.pad_sequence(aspect_value_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    aspect_name_labels = []  # Create an empty list to store aspect name labels

    # Assuming you have a list of aspect name labels called aspect_name_label_list
    for item in batch:
        # Print aspect_names and aspect_name_ids
        print('Aspect Names:', aspect_names)
        print('Aspect Name IDs:', aspect_name_ids)

        aspect_name_labels.append([aspect_names[aspect_id.item()] for aspect_id in item['aspect_name_ids']])

    return {
        'aspect_name_ids': aspect_name_ids,
        'aspect_value_ids': aspect_value_ids,
        'attention_masks': attention_masks,
        'aspect_name_labels': aspect_name_labels
    }

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_function = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10  # Adjust the number of epochs as per your requirements

# Define the loss function and optimizer
aspect_name_criterion = torch.nn.CrossEntropyLoss()
aspect_value_criterion = torch.nn.CrossEntropyLoss()


num_classes = len(aspect_names)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_predictions = 0

    for batch in train_loader:
        print(batch)
        aspect_name_ids = batch['aspect_name_ids'].to(device)
        #aspect_value_ids = batch['aspect_value_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        aspect_name_logits, _ = model(aspect_name_ids, attention_mask)

        # Compute loss for aspect name prediction
        aspect_name_labels = aspect_name_ids[:, 1:]  # Remove the start token from labels

        aspect_name_labels = [aspect_names[label] for label in aspect_name_labels.flatten().tolist()]  # Convert aspect name labels to their corresponding names

        aspect_name_loss = aspect_name_criterion(aspect_name_logits[:, :-1], aspect_name_labels.reshape(-1))

        aspect_name_loss.backward()
        optimizer.step()

        total_loss += aspect_name_loss.item()
        total_correct += (aspect_name_logits.argmax(2) == aspect_name_labels).sum().item()
        total_predictions += aspect_name_labels.numel()


# Compute average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_predictions

    # Print the metrics for the epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')


# Save the trained model
torch.save(model.state_dict(), 'aspect_model.pth')
