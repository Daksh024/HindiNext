import torch
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader
import os

from transformers import AutoTokenizer
# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.01
batch_size = 64
num_epochs = 50
# input_size = 28
# sequence_length = 28
# num_layers = 2
# hidden_size = 256
# num_classes = 10

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

from tqdm import tqdm

# List all files in a directory using os.listdir
basepath = "/home/dai/CapDuck/Project/programs/Sharvari/project_cdac/dataset" # Directory to save dataset
best_loss = float('inf')  # To track the best validation loss
checkpoint_dir =  "/home/dai/CapDuck/Project/programs/Sharvari/project_cdac/models" # Directory to save checkpoints

data = []

for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        file = open("{}/{}".format(basepath,entry))
        sentences = file.readlines()

        for sentence in tqdm(sentences):
            tokens = tokenizer.tokenize(sentence, max_length=len(sentence), truncation=True)
            for i in range(1, 100):
                n_gram_sequence = tokens[:i+1]
                data.append(n_gram_sequence)

input = []
for i in tqdm(data):
    input.append(torch.tensor(tokenizer.convert_tokens_to_ids(i)))

from torch.nn.utils.rnn import pad_sequence
pad_input = pad_sequence(input,batch_first=True)

training_data = (pad_input[:,:-1],pad_input[:,1:])

train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)

import sys

print(sys.getsizeof(training_data)," Bytes")

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, pretrained_embeddings=None):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Freeze embeddings if using pretrained
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 because of bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.fc(out[:, -1, :])  # Using the final hidden state for classification
        return out
    
model = BiLSTM(tokenizer.vocab_size, 100, 256, 2, tokenizer.vocab_size).to(device)
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:  # Adjust this to your dataloader
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # ...
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {loss:.4f}')
        
        # Save model checkpoint if loss improves
        if loss < best_loss:
            best_loss = loss
            checkpoint_path = os.path.join(checkpoint_dir, f'best_loss_checkpoint.bin')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"Best loss checkpoint saved at {checkpoint_path}")
