import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
import re
import multiprocessing


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def preprocess_text(text):
    if isinstance(text, str): 
        text = re.sub(r'[^a-zA-Z\s]', '', text)  
        text = text.lower()                       
        return text  
    return ''


def parallel_preprocess(data):
    with multiprocessing.Pool() as pool:
        return pool.map(preprocess_text, data)


if __name__ == '__main__':
   
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    
    train_data['crimeaditionalinfo'] = train_data['crimeaditionalinfo'].fillna('')
    test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].fillna('')

    
    train_data['cleaned_text'] = parallel_preprocess(train_data['crimeaditionalinfo'])
    test_data['cleaned_text'] = parallel_preprocess(test_data['crimeaditionalinfo'])

    vectorizer = TfidfVectorizer(max_features=10000)  
    X_train_vectorized = vectorizer.fit_transform(train_data['cleaned_text'])
    X_test_vectorized = vectorizer.transform(test_data['cleaned_text'])

    
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0]  
    }

    grid_search = GridSearchCV(MultinomialNB(), param_grid, scoring='f1_weighted', cv=5)


    grid_search.fit(X_train_vectorized, train_data['category'])

    print("Best parameters:", grid_search.best_params_)
    print("Best F1 Score from cross-validation:", grid_search.best_score_)

  
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_vectorized)

    # Accuracy Measurement with zero_division parameter
    accuracy = accuracy_score(test_data['category'], y_pred)
    precision = precision_score(test_data['category'], y_pred, average='weighted', zero_division=0)
    recall = recall_score(test_data['category'], y_pred, average='weighted', zero_division=0)
    f1 = f1_score(test_data['category'], y_pred, average='weighted', zero_division=0)

    # Output the results
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')