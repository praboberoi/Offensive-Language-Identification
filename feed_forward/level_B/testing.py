import torch
import numpy as np
from gensim.models import FastText
import re
import emoji

from feed_forward.level_A.feed_forward_model import FeedForwardNetwork

def load_fasxttxt_model(file_path='./feed_forward/level_A/fasttext_model.bin'):
    return FastText.load(file_path)

def load_feed_forward_model(input_dim, device, hidden_size=128, file_path="./feed_forward/level_A/logistic_regression_model.pt"):
    model = FeedForwardNetwork(input_dim, hidden_size).to(device)
    model.load_state_dict(torch.load(file_path))
    return model

def create_embedding(text, fasttext_model):
    embedding = [fasttext_model.wv[word] for word in text.split() if word in fasttext_model.wv]
    if not embedding:
        return np.zeros(fasttext_model.vector_size)
    return np.mean(embedding, axis=0)

def preprocess(text):
    # Remove @user
    text = re.sub(r'@[\w]+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove dates
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b', '', text)

    # Convert emojis to text
    text = emoji.demojize(text)

    return text

def predict(feedforward_model, embedding_tensor, device):
    embedding_tensor = embedding_tensor.to(device)
    with torch.no_grad():
        output = feedforward_model(embedding_tensor)
        _, predicted = torch.max(output.data, 1)

    if predicted.item() == 1:
        print("The message is offensive.")
    else:
        print("The message is not offensive.")


def main():
    fasttext_model = load_fasxttxt_model()
    input_dim = fasttext_model.vector_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feedforward_model = load_feed_forward_model(input_dim, device)
    user_input = input("Please enter a message or (q) to quit: ")
    while(user_input != 'q'):
        processed_input = preprocess(user_input)
        embedding = create_embedding(processed_input, fasttext_model)
        embedding_tensor = torch.tensor(np.array([embedding]), dtype=torch.float32)
        predict(feedforward_model, embedding_tensor, device)
        user_input = input("Please enter a message or (q) to quit: ")

if __name__ == "__main__":
    main()