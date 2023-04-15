import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
from logistic_regression.level_A.logistic_regression_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

class OffensiveLanguageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx].toarray().squeeze(), dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32) 

# Load and preprocess the data
def load_process_file(file, seperator, header):
    data = pd.read_csv(file, sep=seperator, header=header)
    data['text'].fillna('', inplace=True) 
    data["LABEL"] = (data["average"] >= 0.5).astype(int)
    return data

def vectorise_sparse_bag_of_words(data):
    vectorizer = CountVectorizer()
    batch_size = 1000

    n_batches = int(np.ceil(len(data['text']) / batch_size))
    first_batch = True

    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = data['text'][start:end]

        if first_batch:
            X = vectorizer.fit_transform(batch)
            first_batch = False
        else:
            X_batch = vectorizer.transform(batch)
            X = vstack([X, X_batch])

    X_sparse = csr_matrix(X)
    Y = data["LABEL"].values
    return X_sparse, Y, vectorizer


# Train the logistic regression model
def train_model(train_loader, device, model, optimizer, criterion):
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (tweets, labels) in enumerate(train_loader):
            tweets, labels = tweets.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tweets).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss/len(train_loader):.4f}')

# Test the logistic regression model
def test_model(test_loader, device, model):
    correct = 0
    total = 0
    all_labels = []
    all_predections = []
    with torch.no_grad():
        for tweets, labels in test_loader:
            tweets, labels = tweets.to(device), labels.to(device)
            outputs = model(tweets).squeeze()
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predections.extend(predicted.cpu().numpy())
            
    f1 = f1_score(all_labels, all_predections)

    with open('./logistic_regression/level_A/scores.txt', 'a') as f:
        f.write('XXXXXXXXXXXXXXXXXXTestingScoreXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
        f.write(f"Accuracy: {100 * correct / total:.2f}%\n")
        f.write(f"F1 Score: {f1:.4f}\n")


# Save the CountVectorizer instance
def save_bag_of_words_mode(vectorizer):
    with open("./logistic_regression/level_A/count_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

# Test on a sample text
def sampling_test(vectorizer, model, device):
    sample_text = input('please enter a text for test or q to quit :    ')
    while(sample_text != 'q'):
        sample_vec = vectorizer.transform(np.array([sample_text])).toarray()
        sample_tensor = torch.tensor(sample_vec, dtype=torch.float32).unsqueeze(0).to(device)
        sample_output = model(sample_tensor).squeeze().item()

        if sample_output >= 0.5:
            print("The sample text is predicted to be offensive.")
        else:
            print("The sample text is predicted to be non-offensive.")

        sample_text = input('please enter a text for test or q to quit :    ')

def main():
    data = load_process_file('./formated_file_a.tsv', seperator="\t", header=0)
    X, Y, vectorizer = vectorise_sparse_bag_of_words(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    train_dataset = OffensiveLanguageDataset(X_train, y_train)
    test_dataset = OffensiveLanguageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegression(input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, device, model, optimizer, criterion)
    test_model(test_loader, device, model)

    save_bag_of_words_mode(vectorizer)
    
    # Save the Logistic Regression model
    torch.save(model.state_dict(), "./logistic_regression/level_A/logistic_regression_model.pt")
    
    sampling_test(vectorizer, model, device)


if __name__ == "__main__":
    main()
