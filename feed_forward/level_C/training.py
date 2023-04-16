import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


def load_data(file_path='./formated_file_c.tsv'):
    data = pd.read_csv(file_path, sep='\t', header=0)
    data = data.dropna()
    return data

def train_fast_text_model(data):
    corpus = [text.split() for text in data['text']]
    ft_model = FastText(sentences=data['text'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, epochs=10)
    return ft_model

class OffensiveLanguageDataset(Dataset):
    def __init__(self, data, ft_model):
        self.data = data
        self.ft_model = ft_model
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        embedding = torch.tensor(self.ft_model.wv[text])
        target_values = [self.data.iloc[idx]['average_ind'], self.data.iloc[idx]['average_grp'], self.data.iloc[idx]['average_oth']]
        target = torch.tensor(target_values.index(max(target_values)))  # Get the index of the max value as the target label
        return embedding, target

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy

def test(model, test_dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    loss = running_loss / len(test_dataloader.dataset)
    accuracy = running_corrects.double() / len(test_dataloader.dataset)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    with open('./feed_forward/level_C/scores.txt', 'a') as f:
        f.write('XXXXXXXXXXXXXXXXXXTestingScoreXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
        f.write(f"Accuracy: {100 * loss:.2f}%\n")
        f.write(f"F1 Score: {f1:.4f}\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data()
    ft_model = train_fast_text_model(data)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = OffensiveLanguageDataset(train_data, ft_model)
    test_dataset = OffensiveLanguageDataset(test_data, ft_model)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    input_size = ft_model.vector_size
    hidden_size = 128
    output_size = 3
    model = FeedForward(input_size, hidden_size, output_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_dataloader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

    test(model, test_dataloader, criterion, device)

    ft_model.save("./feed_forward/level_C/fasttext_model.bin")
    torch.save(model.state_dict(), "./feed_forward/level_C/logistic_regression_model.pt")

    
if __name__ == '__main__':
    main()
