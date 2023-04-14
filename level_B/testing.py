import pandas as pd
import torch
import numpy as np
from logistic_regression.level_A.logistic_regression_model import LogisticRegression
from logistic_regression.level_A.formating_tweets import format_tweets
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score

def main():
    
    # Load the CountVectorizer instance
    with open("./logistic_regression/level_B/count_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load the Logistic Regression model
    input_dim = len(vectorizer.vocabulary_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegression(input_dim).to(device)
    model.load_state_dict(torch.load("logistic_regression_model.pt"))

    # Load the test TSV file
    test_data = pd.read_csv("./dev.tsv", sep="\t", header=None, names=["text", "Previous_predict"])
    test_data = format_tweets(test_data)
    test_data = test_data.dropna()
    totalEntries = len(test_data)
    correct = 0

    # Process and predict for each tweet in the test data
    predictions = []
    true_labels = []
    for i, row  in test_data.iterrows():
        sample_vec = vectorizer.transform(np.array([row['text']])).toarray()
        sample_tensor = torch.tensor(sample_vec, dtype=torch.float32).unsqueeze(0).to(device)
        sample_output = model(sample_tensor).squeeze().item()
        if sample_output >= 0.5 and row['Previous_predict'] == 1:
            correct += 1
        elif sample_output < 0.5 and row['Previous_predict'] == 0:
            correct += 1
        if sample_output >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        true_labels.append(row['Previous_predict'])
    
    print(f'Accuracy: {100 * correct/totalEntries:.2f}%')
    print(f'F1 Score: {f1_score(true_labels, predictions):.4f}')

    with open('./logistic_regression/level_B/scores.txt', 'a') as f:
        f.write('XXXXXXXXXXXXXXXXXXTestingScoreXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
        f.write(f'Accuracy: {100 * correct/totalEntries:.2f}%\n')
        f.write(f'F1 Score: {f1_score(true_labels, predictions):.4f}\n')
    
    # Add the predictions to the test_data DataFrame
    test_data['predection'] = predictions

    # Save the results to a new TSV file
    test_data.to_csv("./logistic_regression/level_B/output_file.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
