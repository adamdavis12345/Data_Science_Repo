
# coding: utf-8

# In[1]:


import keras as K
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D


# In[2]:


from keras.datasets import mnist


# In[15]:


# load dataset

(x_train , y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000,28,28,1))

x_test = x_test.reshape((10000,28,28,1))

y_train = K.utils.to_categorical(y_train)

y_test = K.utils.to_categorical(y_test)


# In[4]:


# get shape of dataset and show an example
print(x_train.shape)
rnd = random.randint(0,x_train.shape[0])
plt.imshow(x_train[rnd,:,:,0])


# In[10]:


#define sequential model
sec = Sequential()

#layer 1
sec.add(Conv2D(filters = 32,kernel_size = (3,3), strides = 1, padding = 'same',activation = 'relu', input_shape=(28,28,1),data_format = 'channels_last'))
sec.add(MaxPooling2D(pool_size =(3,3)))
sec.add(Dropout(0.2))

#layer 2
sec.add(Conv2D(filters = 64,kernel_size = (3,3), strides = 1, padding = 'same',activation = 'relu'))
sec.add(MaxPooling2D(pool_size =(3,3)))
sec.add(Dropout(0.2))

#layer 3
sec.add(Conv2D(filters = 128,kernel_size = (3,3), strides = 1, padding = 'same',activation = 'relu'))
sec.add(MaxPooling2D(pool_size =(3,3)))
sec.add(Dropout(0.2))

#fully connected layers
sec.add(Flatten())
sec.add(Dense(128, activation = 'relu'))
sec.add(Dropout(0.2))

sec.add(Dense(64, activation = 'relu'))
sec.add(Dropout(0.2))

sec.add(Dense(32, activation = 'relu'))
sec.add(Dropout(0.2))

#output layer
sec.add(Dense(10, activation = 'softmax'))


# In[11]:


#compile the model
sec.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[12]:


#train the model
sec.fit(x = x_train,y = y_train,batch_size = 64, epochs = 15, verbose =1)


# In[33]:


#evaluate model accuracy
scores = sec.evaluate(x_test,y_test)
print('the model is ', scores[1]*100, '% accurate') 


# In[35]:


#write model weights to disk
sec_json = sec.to_json()
with open("handwiting_recognition.json", "w") as json_file:
    json_file.write(sec_json)

