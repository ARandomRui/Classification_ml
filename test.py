import pandas as pd
import spacy


# Load spaCy's English model (used for lemmatization)
nlp = spacy.load("en_core_web_sm")

custom_stopwords = [
    'ravi', 'arts', 'get', 'guy', 'decides', 'second', 
    'ladies', 'africa', 'earth', 'job', 'look', 
    'today', 'issues', 'll', 'meets', 'fi'
]

lemmatized_stopwords = set()

for word in custom_stopwords:
    doc = nlp(word)         # Run small nlp processing
    for token in doc:
        lemmatized_stopwords.add(token.lemma_)

for word in lemmatized_stopwords:
    lex = nlp.vocab[word]
    lex.is_stop = True


def main():
    global Title_names_list
    global train_dataset
    train_dataset_path = "./data/train_data.txt"

    Title_names_list = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

    train_dataset = pd.read_csv(train_dataset_path,sep=' ::: ', names=Title_names_list ,engine='python')
    print(train_dataset.columns)

    train_dataset.to_csv("cleaned_train_data.txt", sep=str(','), index=False)


        
if __name__ == "__main__":
    main()