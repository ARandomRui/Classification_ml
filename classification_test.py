import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#First edition (unchanged): 0.58 Accuaracy 

def show_results():
    global new_model_prompt
    # Make predictions
    predicted_genres = model.predict(X)
    
    print("Accuaracy: ", model.score(X, y))

    # Find where the predictions are wrong
    incorrect_indices = predicted_genres  != y
    
    test_dataset['PREDICTED_GENRE'] = predicted_genres
    test_dataset['TRUE_GENRE'] = y
    
    # Print out the incorrect cases
    print("Examples of incorrect predictions:")
    range_to_print = 0
    for i in range(X.shape[0]):
        if incorrect_indices[i]:
            range_to_print += 1
            if range_to_print > 10:
                break
            description = test_dataset.iloc[i]["DESCRIPTION"]
            print(f"Index: {i}, Description: {description} True: {test_dataset.iloc[i]['TRUE_GENRE']}, Predicted: {test_dataset.iloc[i]['PREDICTED_GENRE']}") 
           
def load_model():
    global model

    with open('logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)    
        
def main():
    global Title_names_list
    global test_dataset
    global test_dataset_ans
    global X,y
    
    #Setup paths and titles
    test_dataset_path = "./data/test_data.txt"
    test_dataset_ans_path = "./data/test_data_solution.txt"
    Title_names_list = ['ID', 'TITLE', 'DESCRIPTION']
    Ans_Title_names_list = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
    
    #Loading datasets
    test_dataset = pd.read_csv(test_dataset_path,sep=' ::: ', names=Title_names_list ,engine='python')
    test_dataset_ans = pd.read_csv(test_dataset_ans_path,sep=' ::: ', names=Ans_Title_names_list ,engine='python')
    
    #Vectorize dataset main input
    #Why not tfidf = TfidfVectorizer(stop_words='english', max_features=20000) like classification? 
    #This is because tfidf have been tweaked using the .fit() (think of it like adjusting settings for tfidf)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    X_test_vec = tfidf.transform(test_dataset["DESCRIPTION"])
    
    #Setup variables
    X = X_test_vec
    y = test_dataset_ans['GENRE']  # Target: Genre labels
    
    
    load_model()
    show_results()


        
if __name__ == "__main__":
    main()