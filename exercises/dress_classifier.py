#Import TThis guide trains a neural network model to classify images of clothing, like sneakers and shirt

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


#Reads dataset from fashion_mnist that includes images of clothing, like sneakers, dresses and shirts
fashion_mnist = keras.datasets.fashion_mnist

#Classifiying the data into ones to be tested to check for algorithm performance and trained
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#This is optional for plotting training images before proceeding with the rest of the code
'''plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)'''


#Classifying parts of the data
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Cast the datatype of the image components from an integer to a float and divide by 255
train_images = train_images / 255.0
test_images = test_images / 255.0


#Displays the first 25 images from the training set and displays the class name below each image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    

#The basic building block of a neural network is the layer
#Layers extract representations from the data fed into them
#The Sequential model is a linear stack of layers
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), #transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixel
        keras.layers.Dense(128, activation=tf.nn.relu), #128 nodes (or neurons) densely-connected, or fully-connected, neural layers
        keras.layers.Dense(10, activation= tf.nn.softmax) #this returns an array of 10 probability scores that sum to 1
        ])
 
#Configures the model for training  
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Trains the model for a given number of epochs (iterations on a dataset)
model.fit(train_images, train_labels, epochs=5)

#Checks for the accuracy of the model against the test images 
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ",test_acc)

#Generates output predictions for the test images
predictions = model.predict(test_images)

# Plot the first 25 test images, their predicted label, and the true label
# Correct color predictions are set to green and incorrect predictions are set to red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
    
    
