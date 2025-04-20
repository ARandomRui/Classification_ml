import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train_model(train_dataset):
    global model, X, y, tfidf
    # Example: Clean and vectorize movie descriptions
    tfidf = TfidfVectorizer(stop_words='english', max_features=20000) #5000 -> 0.62 #10000 -> 0.7, #20000 -> 0.71
    X = tfidf.fit_transform(train_dataset['DESCRIPTION'])  # Input: Text descriptions
    y = train_dataset['GENRE']  # Target: Genre labels
    #Note: X and y are the standard for input and answer/label/target whatever you want to say

    # 3. Train a lightweight model (Logistic Regression)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    print("Model trained, Accuracy:", model.score(X, y))
    return model

def create_save_prompt():
    global new_model_prompt
    new_model_prompt = input("Do you want to create a new model? (y for yes and anything else to use current saved model)")

def check_data(train_dataset):
    #Check for any NaN values
    list = train_dataset.isnull().sum()
    print(list)

    for each in Title_names_list:
        if (list[each]) != 0:
            print("Missing Value Found!") #Haven't implemented what happens if find NaN
            sys.exit(0)

    print("All datas are fully loaded!\n\n")
            

    print("Sample Data:")
    print(train_dataset.iloc[:5])  #print the first five as sample

    temp = train_dataset["GENRE"].value_counts()
    print("Data description:")
    print(temp)

def show_results():
    global new_model_prompt
    # Make predictions
    y_pred = model.predict(X)

    # Find where the predictions are wrong (compares two numpy arrays of the same length , returns a boolean value)
    incorrect_indices = y_pred != y

    # Print out the incorrect cases
    print("Examples of incorrect predictions:")
    range_to_print = 0
    for i in range(X.shape[0]):
        if incorrect_indices[i]:
            range_to_print += 1
            if range_to_print > 10:
                break
            description = train_dataset.iloc[i]["DESCRIPTION"]
            print(f"Index: {i}, Description: {description} True: {y[i]}, Predicted: {y_pred[i]}") 

def save_model():
    if new_model_prompt == "y":
        with open('logistic_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
         
def main():
    global Title_names_list
    global train_dataset
    train_dataset_path = "./data/train_data.txt"

    Title_names_list = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

    train_dataset = pd.read_csv(train_dataset_path,sep=' ::: ', names=Title_names_list ,engine='python')
    print(train_dataset.columns)
    create_save_prompt()
    if new_model_prompt == 'y':
        check_data(train_dataset)
        train_model(train_dataset)
        show_results()
        save_model()
    else:
        show_results()

        
if __name__ == "__main__":
    main()