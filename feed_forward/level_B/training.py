import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gensim.models import FastText
import numpy as np

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class OffensiveLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=0)
    data = data.dropna()
    data['label'] = (data['average'] >= 0.5).astype(int)
    return data

def train_fast_text_model(data):
    return FastText(sentences=data['text'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, epochs=10)

def create_embedding(text, fasttext_model):
    embedding = [fasttext_model.wv[word] for word in text.split() if word in fasttext_model.wv]
    if not embedding:
        return np.zeros(fasttext_model.vector_size)
    return np.mean(embedding, axis=0) 

def main():
    data = load_data('./formated_file_b.tsv')
    fasttext_model = train_fast_text_model(data)
    X = data['text'].apply(lambda x : create_embedding(x, fasttext_model)).to_numpy()
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = OffensiveLanguageDataset(X_train, y_train)
    test_dataset = OffensiveLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = fasttext_model.vector_size
    hidden_dim = 128

    model = FeedForwardNetwork(input_dim, hidden_dim).to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss/len(train_loader):.4f}')

    correct = 0
    total = 0
    all_labels = []
    all_predections = []
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predections.extend(predicted.cpu().numpy())

    f1 = f1_score(all_labels, all_predections)
    with open('./feed_forward/level_B/scores.txt', 'a') as f:
        f.write('XXXXXXXXXXXXXXXXXXTestingScoreXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
        f.write(f"Accuracy: {100 * correct / total:.2f}%\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    fasttext_model.save("./feed_forward/level_B/fasttext_model.bin")
    torch.save(model.state_dict(), "./feed_forward/level_B/logistic_regression_model.pt")
    
if __name__ == "__main__":
    main()
