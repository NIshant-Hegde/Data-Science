#!/usr/bin/env python
# coding: utf-8

# In[145]:


import os                                                        #To navigate between directories and files
import io                                                        #For file manipulation and management
from pandas import DataFrame                                     #To create dataframes to visualize data
import pandas as pd                                              #For data visualisations
from sklearn .feature_extraction.text import CountVectorizer     #To count words in a given file
from sklearn.naive_bayes import MultinomialNB                    #To implement Naive_Bayes_Classifier 
from sklearn.metrics import accuracy_score, precision_score, f1_score


# In[146]:


data = pd.read_csv('C:\\Users\\Nishant\\Desktop\\Data Science notes\\Machine Learning, Data Science and Deep Learning with Python\\ML_Course_materials\\emails.csv')
data = DataFrame(data)                                           #Creating a dataframe
data.head()                                                      #Printing the first five rows of the dataframe


# In[147]:


data.shape                                                       #Printing the shape of the dataframe


# In[148]:


data.drop_duplicates(inplace = True)                             #Deleting duplicates from the dataframe
data.shape                                                       #Checking if the duplicates have been removed by printing out the shape


# In[149]:


data['text'] = data['text'].map(lambda text: text[8:])           #Since every text message starts with 'SUBJECT', deleting that word 
                                                                 #in every text row
print(data['text'])                                              # Checking the new dataframe/text column


# In[150]:


from sklearn.model_selection import train_test_split     #Splitting the dataset into train and test 
X_train, X_test, Y_train, Y_test = train_test_split(data['text'], data['spam'], test_size = 0.2, random_state = 10)


# In[151]:


vectorizer = CountVectorizer()                          #Using the CountVectorizer() to count the frequency of every word
counts = vectorizer.fit_transform(X_train)              #Counting the frequency and storing them in a list called counts
targets = Y_train                                       #Creating targets for the classifier
classifier = MultinomialNB()                            #Creating the classifier model
classifier.fit(counts, targets)                         #Training the classifier


# In[152]:


def test_func(predictions):                             #A function to predict the emails
    pred_class = []
    for preds in predictions:
        if preds == 1:
            pred_class.append("spam")
        else:
            pred_class.append("ham")
    return pred_class                                  #returns a email consisting of predictions


# In[157]:


examples = ["congralutions! you've won a $100000", "Dear Sir, Will you be free tomorrow? This is to ask you regarding a meeting."]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)            #Testing
print(test_func(predictions))


# In[143]:


print(X_test.shape)


# In[144]:


X_test_counts = vectorizer.transform(X_test)
X_test_counts.shape
preds = classifier.predict(X_test_counts)
acc = accuracy_score(Y_test, preds)
precision = precision_score(Y_test, preds)
f1 = f1_score(Y_test, preds)
acc = acc * 100
precision = precision * 100
print("Mean accuracy: {} \nPrecision: {} \nF1_score: {}".format(acc, precision, f1))


# In[101]:





# In[ ]:




