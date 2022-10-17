#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[87]:


movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')


# In[88]:


movies.head(1)


# In[89]:


movies.shape


# In[90]:


movies = movies.merge(credits,on='title')


# In[91]:


movies.shape


# In[92]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[93]:


movies.head()


# In[94]:


import ast


# In[95]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[96]:


movies.dropna(inplace=True)


# In[97]:


movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[98]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[99]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[100]:


movies['cast'] = movies['cast'].apply(convert)
movies.head()


# In[101]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[102]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[ ]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[ ]:


movies.sample(5)


# In[ ]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[ ]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[ ]:


movies.head()


# In[ ]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[ ]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[ ]:


new = movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[ ]:


new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[ ]:


vector = cv.fit_transform(new['tags']).toarray()


# In[ ]:


vector.shape


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity = cosine_similarity(vector)


# In[ ]:


similarity


# In[ ]:


new[new['title'] == 'The Lego Movie'].index[0]


# In[ ]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[82]:


recommend('Gandhi')


# In[83]:


recommend('Avatar')


# In[84]:


recommend('Spectre')


# In[85]:


recommend('John Carter')

