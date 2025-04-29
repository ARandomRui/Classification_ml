import pickle

def load_vectorizer(path='tfidf_vectorizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model(path='logistic_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def show_top_terms_per_class(tfidf, model, top_n=40):
    feature_names = tfidf.get_feature_names_out()
    class_labels = model.classes_
    coefficients = model.coef_ #This is a 2D array/list
    
    print("----")
    print(coefficients)
    print("----")
    

    for i, class_label in enumerate(class_labels):
        print(f"\n🔹 Top {top_n} terms for genre: {class_label}")
        top_indices = coefficients[i].argsort()[::-1][:top_n] #basically get the list of the genre, argsort() basically "sorts the array" returns a list being ["The index of the lowest number, the index of the 2nd lowest,... The index of the highest], so [::-1] reverses it being high to low. The top_n is the first X elements given by params
        for idx in top_indices:
            term = feature_names[idx]
            weight = coefficients[i][idx]
            print(f"  {term:<20} {weight:.4f}") #Formating magick-

def main():
    tfidf = load_vectorizer()
    model = load_model()
    show_top_terms_per_class(tfidf, model)

if __name__ == "__main__":
    main()