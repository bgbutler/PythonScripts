# Natural Language Processing

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# for loading the dataset
import os

# libraries for dealing with text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# %reset -f


dataFile = os.path.normpath("~/Documents/Programming/TrainingDataResources/Natural_Language_Processing/Restaurant_Reviews.tsv")

# set tab delimiter and ignore double quotes
dataset = pd.read_csv(dataFile, delimiter = '\t', quoting = 3)


number_of_reviews = len(dataset)


########## key cleaning steps
# loop through them all
# create a corpus
corpus = []


for i in range(0,number_of_reviews):
    
    # keep the letters, remove anything not a letter
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    # make it all lowercase

    review = review.lower()

    review = review.split()

    # include stemming into it
    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    # put the list back together into a string separated by spaces
    review = ' '.join(review)
    
    corpus.append(review)

# create the bag of words model
# creates a document term matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# split the data to training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.15, random_state = 0)


# create the classifier
# needs to be changed for the classifier to be used
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# predicting the test set results
y_pred = classifier.predict(X_test)

# check the results with a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




