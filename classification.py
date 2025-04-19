import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train_dataset_path = "./data/train_data.txt"

Title_names_list = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

train_dataset = pd.read_csv(train_dataset_path,sep=' ::: ', names=Title_names_list ,engine='python')
print(train_dataset.columns)

#Check for any NaN values
list = train_dataset.isnull().sum()

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

# Example: Clean and vectorize movie descriptions
tfidf = TfidfVectorizer(stop_words='english', max_features=20000) #5000 -> 0.62 #10000 -> 0.7, #20000 -> 0.71
X = tfidf.fit_transform(train_dataset['DESCRIPTION'])  # Input: Text descriptions
y = train_dataset['GENRE']  # Target: Genre labels

# 3. Train a lightweight model (Logistic Regression)
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

print("Accuracy:", model.score(X, y))

# Make predictions
y_pred = model.predict(X)

# Find where the predictions are wrong
incorrect_indices = y_pred != y