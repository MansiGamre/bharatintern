#!/usr/bin/env python
# coding: utf-8

# # Name : Mansi Gamre
# 

# # Handwritten Digit Recognition Using MNIST Dataset With The Help Of Neural Network.

# # Dataset Used : MNIST Dataset

# # About Dataset

# MNIST is a commonly used dataset in machine learning and computer vision research, which consists of a set of 70,000 images of handwritten digits (0-9), each of size 28x28 pixels.
# The dataset is split into two sets: a training set of 60,000 images and a test set of 10,000 images. The training set is used to train a machine learning model, while the test set is used to evaluate the model's performance.

# In[26]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import unique , argmax

# TensorFlow already contain MNIST data set which can be loaded using Keras
import tensorflow as tf # installing tenserflow
from tensorflow import keras


# In[27]:


# To Load the MNIST dataset from the Keras API provided by TensorFlow.
mnist = tf.keras.datasets.mnist


# The Above Code Reflects that the Dataset Contains :
# 
# 1. An array of 60,000 images, each represented as a 28x28 NumPy array, with pixel values ranging from 0 to 255.
# 2. An array of 60,000 labels, each representing the correct digit (0-9) for the 1.
# 3. An array of 10,000 images, each represented as a 28x28 NumPy array, with pixel values ranging from 0 to 255.
# 4.  An array of 10,000 labels, each representing the correct digit (0-9) for the 3.

# # Dividing the data into train and test data.

# In[28]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[29]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[30]:


print(x_train)


# In[31]:


print(x_test)


# In[32]:


# Reshaping the input Data which is used as a input in CNN in Tenserflow
# CNN takes the input Data in 4D Format with the shape (num_samples, image_height, image_width, num_channels)
# Here (num_channels) is set to 1 which means input image is Grayscale.

x_train = x_train.reshape((x_train.shape[0] , x_train.shape[1] , x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0] , x_test.shape[1] , x_test.shape[2],1))
print(x_train.shape)
print(x_test.shape)
print(x_train.dtype)
print(x_test.dtype)


# In[33]:


# Normalizing Pixel Values

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0
print(x_train.dtype)
print(x_test.dtype)


# In[34]:


# Visulaizing Subsets of images in MNIST Dataset along with coressponding labels.

fig=plt.figure(figsize=(5,3))
for i in range(20):
    ax =fig.add_subplot(2,10,i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]), cmap='Blues')
    ax.set_title(y_train[i])


# In[35]:


# showing shape of single image
img_shape= x_train.shape[1:]
img_shape


# # BUILDING NEURAL NETWORK THAT CAN READ HANDWRITTEN DIGITS.
# 

# In[36]:


# Creating aSequential Model in Keras
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


# In[37]:


model.summary()


# This summary shows that the model has four layers:
# 
# 1. A Flatten layer that flattens the input images to a 1D array of length 784.
# 2. A Dense layer with 128 units and ReLU activation.
# 3. A Dropout layer that randomly sets 20% of the input units to 0 during training.
# 4. A second Dense layer with 10 units and no activation function.
# 
# The summary also shows the number of trainable parameters in each layer, as well as the total number of trainable parameters in the model. In this case, the model has a total of 101,770 trainable parameters.

# In[38]:


# Displaying Neural Network Model
from tensorflow.keras.utils import plot_model
plot_model(model, 'model.jpg', show_shapes = True)


# In[39]:


# Making Prediction on Model
prediction = model(x_train[:1]).numpy()
prediction


# In[40]:


# Applying Softmax() Function to prediction array
# This convert an output vector of real numbers into a probability distribution over predicted classes
tf.nn.softmax(prediction).numpy()


# In[41]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], prediction).numpy()
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])


# # Model fitting

# In[42]:


# Training the Model
model.fit(x_train, y_train, epochs=5)


# In[43]:


# Evaluating the Model
model.evaluate(x_test, y_test, verbose=2)


# In[44]:


# Creating a new sequential model which includes both previously trained model and softmax layer.
probability_model = tf.keras.Sequential([ model,tf.keras.layers.Softmax() ])
probability_model(x_test[:5])


# In[45]:


# Displaying a Grayscale Image
img = x_train[12]
plt.imshow(np.squeeze(img) ,cmap='gray')
plt.show()


# In[46]:


# Predicting the Result
img= img.reshape(1, img.shape[0],img.shape[1],img.shape[2])
p= model.predict([img])
print("predicted : {}".format(argmax(p)))

