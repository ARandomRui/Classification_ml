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

    # Lemmatize entire DESCRIPTION column
    descriptions = train_dataset['DESCRIPTION'].astype(str).tolist()
    print("converted to list done..")
    
    # Process them with nlp.pipe() cuz better than apply()
    lemmatized_descriptions = []
    count = 0
    for doc in nlp.pipe(descriptions, batch_size=50):  # batch_size can be tuned
        lemmas = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
        print(count)
        count+=1
        lemmatized_descriptions.append(lemmas)

    train_dataset['DESCRIPTION'] = lemmatized_descriptions
    train_dataset.to_csv("./data/cleaned_train_data.txt", sep=str(','), index=False)


        
if __name__ == "__main__":
    main()