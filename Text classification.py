import numpy as np
import re #regex expressions for data preprocessing
import nltk #natural language tool kit
from sklearn.datasets import load_files #imports dataset. divides dataset into data and target set
nltk.download('stopwords')

import pickle #serializes objects. converts object(list,dict etc.) into character stream

from nltk.corpus import stopwords #'the' 'is' 'are'
nltk.download('wordnet')
movie_data = load_files(r"C:\Users\Mallika\Desktop\NLP miniproject\txt_sentoken")
X, y = movie_data.data, movie_data.target
documents = []
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer #term frequency
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split#split training and testing set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
stemmer = WordNetLemmatizer() #grouping different forms of the word as a single unit

def checkReview(review):
    if review == 0:
        return "Negative"
    else:
        return "Positive"

for sen in range(0, len(X)): #data preprocessing
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=10, stop_words=stopwords.words('english'))#using bag of words approach.text to numbers
X = vectorizer.fit_transform(documents).toarray()#min_df:min num of docs it should occur in.
    

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()


tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=10, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)#train algorthim
y_pred = classifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("\n1st Review : ", documents[8])
print("\nClass of review: ", checkReview(y_pred[8]))
