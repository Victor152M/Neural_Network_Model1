#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np
from random import randint as randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(50):
    random_younger = randint(13,65)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)
    
for i in range(1000):
    random_younger = randint(13,65)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


train_labels=np.array(train_labels)
train_samples=np.array(train_samples)
train_labels,train_samples = shuffle(train_labels,train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))


# In[121]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy


# In[122]:


model = Sequential([
    Dense(units=16, input_shape=(1,), activation="sigmoid"),
    Dense(units=32, activation="sigmoid"),
    Dense(units=2, activation="softmax")
])


# In[123]:


model.compile(optimizer=SGD(learning_rate=0.005),loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[125]:


model.fit(x=scaled_train_samples,y=train_labels,validation_split=0.1,batch_size=10,epochs=100,shuffle=True,verbose=2)


# In[ ]:




