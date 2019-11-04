path = "C:\\Users\\Nishant\\Desktop\\Data Science notes\\Data Science Projects\\"
train_values = "train_2kmZucJ.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm_notebook as tqdm
from ipywidgets import FloatProgress
from IPython.display import display


xdf = pd.read_csv(path + train_values).fillna(' ')
xdf.head(25)
xdf.describe()                                       #Descriptive Statistics

xdf['tweet'] = xdf['tweet'].str.lower()              #Converting the text to lower case
xdf.head()


from nltk.stem import PorterStemmer                  #Using the porterstemmer library to convert each word into its root form

ps = PorterStemmer()                        

def root_word(x):                                    #A user defined function to convert each word of the text feature into root-word 
    root = []
    for phrase in x.split():
        root.append(ps.stem(phrase))                 #Converting into root form and appending to the list
    return " ".join(root)

def preprocess(ReviewText):                          #A user defined function to preprocess the text feature 
    ReviewText = ReviewText.str.replace(",", " ")
    ReviewText = ReviewText.str.replace("!", " ")
    ReviewText = ReviewText.str.replace("-", " ")
    ReviewText = ReviewText.str.replace(";", " ")
    ReviewText = ReviewText.str.replace(":", " ")
    ReviewText = ReviewText.str.replace(".", " ")
    ReviewText = ReviewText.str.replace("=", " ")
    ReviewText = ReviewText.str.replace("@", " ")
    ReviewText = ReviewText.str.replace("&", " ")
    ReviewText = ReviewText.str.replace("$", " ")
    ReviewText = ReviewText.str.replace(".", " ")
    ReviewText = ReviewText.str.replace("|", " ")
    ReviewText = ReviewText.str.replace("#", " ")
    ReviewText = ReviewText.str.replace("?", " ")
    ReviewText = ReviewText.str.replace("^", " ")
    ReviewText = ReviewText.str.replace("*", " ")    #Replacing the punctuation marks
    
    return ReviewText

xdf['tweet'] = preprocess(xdf['tweet'])              #Applying the user defined function preprocesss on text feature column
xdf["tweet"] = xdf["tweet"].str.lower()

xdf["tweet"] = xdf["tweet"].map(lambda x: root_word(x)) #Applying the user defined function root_word on text feature column
xdf.head(30)

xdf.drop('id', axis = 1, inplace = True)


from textblob import TextBlob           #TextBlob for feature engg(Calculating sentiment polarity and total words in each text row)

#Feature Engineering
xdf['polarity'] = xdf['tweet'].map(lambda text: TextBlob(text).sentiment.polarity) #Creating a new row for sentiment_polarity
xdf['review_len'] = xdf['tweet'].astype(str).apply(len)                            #Calculating the number of characters in review column   
xdf['word_count'] = xdf['tweet'].apply(lambda x: len(str(x).split())) 
xdf.head()
max(xdf['review_len'])

train_y = xdf['label']                                                 #Creating the labels array
sentences = xdf['tweet']

from tensorflow.keras.preprocessing.text import Tokenizer              #To tokenize the text
from tensorflow.keras.preprocessing.sequence import pad_sequences      #To pad uneven length sequences

#Hyper parameters
num_words = 10000
pad_type = 'post'
oov_token = "<OOV>"                                                    #Out of vocabulary token
embedding_dim = 16
max_length = 250


tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)    #Creating a tokenizer object
tokenizer.fit_on_texts(sentences)                                      #Using the method fit_on_texts() to tokenize the text feature
word_index = tokenizer.word_index                                      #Displaying the word-index

 con
sequences = tokenizer.texts_to_sequences(sentences)                    #Using the texts_to_sequences() method to convert the tokens into sequences
padded_sequences = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating = 'post')
                                                                       #Padding the sequences. Padding type is "post".


pdf = pd.DataFrame(padded_sequences)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(pdf, train_y, test_size = 0.1, random_state = 10, shuffle = True) 
                                                                       #Splitting the dataset into train_data and test_data
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([                                                   #Creating a sequential model that uses embeddings of dimension 16
    Embedding(num_words, embedding_dim, input_length = max_length),
    GlobalAveragePooling1D(),                                          #GlobalAveragePooling1D() instead of Flatten() as the former results in less trainable parameters
    Dense(16, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, epochs = 50, validation_data = (test_x, test_y), verbose = 1)

test_file = "test_oJQbWVk.csv"
sub_file = "sample_submission_LnhVWA4.csv"

tdf = pd.read_csv(path + test_file).fillna(' ')
sdf = pd.read_csv(path + sub_file).fillna(' ')
tdf.head()

tdf['tweet'] = preprocess(tdf['tweet']) 
tdf["tweet"] = tdf["tweet"].str.lower()
tdf["tweet"] = tdf["tweet"].map(lambda x: root_word(x)) 
ID = tdf['id']
tdf.drop('id', axis = 1, inplace = True)
test_sequences = tokenizer.texts_to_sequences(tdf['tweet'])           #Tokenizing the test data 
padded_test = pad_sequences(test_sequences, padding = 'post', truncating = 'post', maxlen = max_length)  #Padding the test data


preds = model.predict(padded_test)       #Predicitions

for i, values in enumerate(preds):       #Coverting the prediction probabilities to binary classes
    if values > 0.5:
        preds[i] = 1
    else:
        preds[i] = 0

preds = pd.DataFrame(preds, columns = ['label'])     #Creating a dataframe of the predictions for submission
preds['id'] = tdf['id']
preds = preds[['id', 'label']]
preds.head()
preds.to_csv(path + "submit.csv", index = False)
