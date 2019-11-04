#!/usr/bin/env python
# coding: utf-8

# In[1]:


path = "C:\\Users\\Nishant\\Desktop\\Data Science notes\\Data Science Projects\\"
train_values = "train_2kmZucJ.csv"


# In[2]:


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


# In[3]:


xdf = pd.read_csv(path + train_values).fillna(' ')


# In[4]:


xdf.head(25)


# In[5]:


xdf.describe()


# In[6]:


xdf['tweet'] = xdf['tweet'].str.lower()


# In[7]:


xdf.head()


# In[8]:


from nltk.stem import PorterStemmer

ps = PorterStemmer()

def root_word(x):
    root = []
    for phrase in x.split():
        root.append(ps.stem(phrase))
    return " ".join(root)

def preprocess(ReviewText):     #Function 
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
    ReviewText = ReviewText.str.replace("*", " ")
    
    return ReviewText

xdf['tweet'] = preprocess(xdf['tweet']) 
xdf["tweet"] = xdf["tweet"].str.lower()

xdf["tweet"] = xdf["tweet"].map(lambda x: root_word(x)) 


# In[9]:


xdf.head(30)


# In[10]:


xdf.drop('id', axis = 1, inplace = True)


# In[11]:


xdf.head()


# In[12]:


from textblob import TextBlob

#Feature Engineering
xdf['polarity'] = xdf['tweet'].map(lambda text: TextBlob(text).sentiment.polarity) #Creating a new row for sentiment_polarity
xdf['review_len'] = xdf['tweet'].astype(str).apply(len)                            #Calculating the number of characters in review column   
xdf['word_count'] = xdf['tweet'].apply(lambda x: len(str(x).split())) 


# In[13]:


xdf.head()
max(xdf['review_len'])


# In[14]:


train_y = xdf['label']
sentences = xdf['tweet']


# In[15]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Hyper parameters
num_words = 10000
pad_type = 'post'
oov_token = "<OOV>"
embedding_dim = 16
max_length = 250


# In[16]:


tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


# In[17]:


sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating = 'post')


# In[18]:


pdf = pd.DataFrame(padded_sequences)


# In[19]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(pdf, train_y, test_size = 0.1, random_state = 10, shuffle = True)


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([
    Embedding(num_words, embedding_dim, input_length = max_length),
    GlobalAveragePooling1D(),
    Dense(16, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(train_x, train_y, epochs = 50, validation_data = (test_x, test_y), verbose = 1)


# In[21]:


test_file = "test_oJQbWVk.csv"
sub_file = "sample_submission_LnhVWA4.csv"

tdf = pd.read_csv(path + test_file).fillna(' ')
sdf = pd.read_csv(path + sub_file).fillna(' ')

tdf.head()


# In[22]:


tdf['tweet'] = preprocess(tdf['tweet']) 
tdf["tweet"] = tdf["tweet"].str.lower()

tdf["tweet"] = tdf["tweet"].map(lambda x: root_word(x)) 

ID = tdf['id']


# In[23]:


tdf.drop('id', axis = 1, inplace = True)


# In[24]:


test_sequences = tokenizer.texts_to_sequences(tdf['tweet'])


# In[25]:


padded_test = pad_sequences(test_sequences, padding = 'post', truncating = 'post', maxlen = max_length)


# In[26]:


preds = model.predict(padded_test)


# In[27]:


for i, values in enumerate(preds):
    if values > 0.5:
        preds[i] = 1
    else:
        preds[i] = 0


# In[28]:


preds = pd.DataFrame(preds, columns = ['label'])
preds['id'] = tdf['id']
preds = preds[['id', 'label']]
preds.head()


# In[ ]:


preds.to_csv(path + "submit.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:


8296045159

