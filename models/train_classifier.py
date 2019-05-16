# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import sqlalchemy as sqla
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import pickle


def load_data(database_filepath):
    '''
    INPUT:  
        database_filepath (str): database with table name "Messages" having processed messages
    OUTPUT: 
        X (pandas dataframe): messages column
        Y (pandas dataframe): category columns marked as 1 if the message belongs to that category 
        category_names (list of strings): list of category names
    DESCRIPTION:
            read table named "Messages" from the given database
            and select 'message' as X and all ccategories columns as Y
            and get list of catefories as category_names
    '''

    engine = sqla.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]

    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    '''
    INPUT:  
        text (str): text message
    OUTPUT: 
        clean_tokens (list of strings): clean tokens
    DESCRIPTION:
            read the given text message and convert into clean tokens
            using word_tokenizer and WordNetLemmatizer
    '''

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT:  None
    OUTPUT: multiclass classifier model
    DESCRIPTION:
            define ML pipeline to build a multiclass classifier model
            in this case using AdaboostClassifier
            set param to perform a GridSerach
            build model with best performing params using GridSearchCV
    '''
    
    # create ML pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])

    # set parameters for to perform GridSerach
    param_grid = {"clf__estimator__n_estimators": [50, 100],
                "clf__estimator__learning_rate": [0.5,1]
                }

    # perform grid search
    model = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=3, verbose=2)

    return model
    
    

def display_results(y_test, y_pred, category_names):
    '''
    INPUT:  
        y_test (pandas dataframe): multiclass targeted labels
        y_pred (pandas dataframe): multiclass predicted labels
        category_names (list of strings): list of categories
    OUTPUT: for each category print multiclass precision, recall, f1-score, support

    DESCRIPTION:
            loop through the category_names 
            and print multiclass precision, recall, f1-score, support
            for each category 
    '''
    
    for i, col in enumerate(category_names):
        print("Category: ", col,"\n",
            classification_report(y_test.loc[:,col], y_pred[:,i]), "\n",
            "Accuracy: %.2f" %(accuracy_score(y_test.loc[:, col], y_pred[:,i])),"\n")

    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:  
        model (str): trained model
        X_test (pandas dataframe): messages in the test set
        Y_test (pandas dataframe): multiclass targeted labels
        category_names (list of strings): list of categories
    OUTPUT: display multiclass precision, recall, f1-score, support

    DESCRIPTION:
            test the trained model against the messages in test data set 
            and display multiclass precision, recall, f1-score, support
            for each category in the category_names
    '''

    Y_pred = model.predict(X_test)
    
    display_results(Y_test, Y_pred, category_names)

    

def save_model(model, model_filepath):
    '''
    INPUT:  
        model (str): trained model
        model_filepath (str): pickle file path to save the model 
    OUTPUT: 

    DESCRIPTION:
            save the model passed as the path given as input 
    '''

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:

        # get database_filepath and model_filepath as input
        database_filepath, model_filepath = sys.argv[1:]
        
        # load databse
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # build model
        print('Building model...')
        model = build_model()
        
        # tain model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Evaluate model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # save model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
