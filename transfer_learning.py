# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 07:32:53 2024

@author:  PRINCELY OSEJI
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from IPython.display import display
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Initialize lists
train = []
valid = []
test = []

# Define the base directory
base_dir = "C:/Users/HYACINTH OSEJI/Downloads/food11"

# Iterate through the training directory
for i in os.listdir(os.path.join(base_dir, "training")): 
    train.extend(os.listdir(os.path.join(base_dir, 'training', i)))

# Iterate through the validation directory
for i in os.listdir(os.path.join(base_dir, "validation")): 
    valid.extend(os.listdir(os.path.join(base_dir, 'validation', i)))

# Iterate through the evaluation directory
for i in os.listdir(os.path.join(base_dir, "evaluation")): 
    test.extend(os.listdir(os.path.join(base_dir, 'evaluation', i)))

print('Number of train images: {} \nNumber of validation images: {} \nNumber of test images: {}'.format(len(train), len(valid), len(test)))

# Data exploration and visualization for training set
fig, axs = plt.subplots(11, 5, figsize=(32, 32))
count = 0

# Iterate through each class in the training directory
for i in os.listdir(os.path.join(base_dir, 'training')):
    # Get the list of all images that belong to a particular class
    train_class = os.listdir(os.path.join(base_dir, 'training', i))
  
    # Plot 5 images per class
    for j in range(5):
        img = os.path.join(base_dir, 'training', i, train_class[j])
        axs[count][j].title.set_text(i)
        axs[count][j].imshow(PIL.Image.open(img))  
    count += 1

fig.tight_layout()

# Counting the number of images per class
No_images_per_class = []
Class_name = []
for i in os.listdir(os.path.join(base_dir, 'training')):
    Class_name.append(i)
    train_class = os.listdir(os.path.join(base_dir, 'training', i))
    print('Number of images in {}={}\n'.format(i, len(train_class)))
    No_images_per_class.append(len(train_class))
    
# Bar plot of number of images per class in the training set
fig = plt.figure(figsize=(10,5))
plt.bar(Class_name, No_images_per_class, color = sns.color_palette("cubehelix", len(Class_name)))
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class in Training Set')
plt.xticks(rotation=90)
fig.tight_layout()

# Create run-time augmentation on training and test dataset
# For training datagenerator, we add normalization, shear angle, zooming range, and horizontal flip
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generator for training, validation, and test dataset.
train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir, 'training'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'validation'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        os.path.join(base_dir, 'evaluation'),
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

# Build and load the inception resnet model
base_model = InceptionResNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape = (256, 256, 3)))

# Print model summary
base_model.summary

# Freeze the basemodel weigths and add the clasification head to the model
base_model.trainable = False
headmodel = base_model.output
headmodel = GlobalAveragePooling2D(name = 'global_average_pool')(headmodel)
headmodel = Flatten(name = 'flatten')(headmodel)
headmodel = Dense(128, activation = 'relu', name = 'dense_1')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(512, activation = 'relu', name = 'dense_2')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(11, activation = 'softmax', name = 'dense_3')(headmodel)

model = Model(inputs = base_model.input, outputs = headmodel)

# Compile and train the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(learning_rate=0.01, momentum=0.9), 
    metrics=['accuracy']
)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="C:/Users/HYACINTH OSEJI/Downloads/food11/weights.hdf5.keras", verbose=1, save_best_only=True)

# fine tune the model with very low learning rate
history = model.fit(train_generator, steps_per_epoch= train_generator.n // 32, epochs = 10, validation_data= validation_generator, validation_steps= validation_generator.n // 32, callbacks=[checkpointer, earlystopping])

# Fine tune the model for transfer learning
base_model.trainable = True

earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="C:/Users/HYACINTH OSEJI/Downloads/food11/weights_fine.hdf5.keras", verbose=1, save_best_only=True)

# fine tune the model with very low learning rate
model.compile(
    loss='categorical_crossentropy', 
    optimizer=SGD(learning_rate=0.0001, momentum=0.9), 
    metrics=['accuracy']
)

history = model.fit(train_generator, steps_per_epoch= train_generator.n // 32, epochs = 10, validation_data= validation_generator, validation_steps= validation_generator.n // 32, callbacks=[checkpointer, earlystopping])


# Assess performance of trained model
# Loading pretrained weights
model.load_weights("C:/Users/HYACINTH OSEJI/Downloads/food11/weights_fine.hdf5")

# Evaluate the model performance
evaluate = model.evaluate_generator(test_generator, steps = test_generator.n // 32, verbose =1)

print('Accuracy Test : {}'.format(evaluate[1]))

# assigning label names to the corresponding indexes
labels = {0: 'Bread', 1: 'Dairy product', 2: 'Dessert', 3:'Egg', 4: 'Fried food', 5:'Meat',6:'Noodles-Pasta',7:'Rice', 8:'Seafood',9:'Soup',10: 'Vegetable-Fruit'}

# Initialize lists
prediction = []
original = []
image = []
count = 0

# Define the base directory for evaluation
evaluation_dir = os.path.join(base_dir, 'evaluation')

# Iterate through each class in the evaluation directory
for i in os.listdir(evaluation_dir):
    for item in os.listdir(os.path.join(evaluation_dir, i)):
        # Code to open the image
        img = PIL.Image.open(os.path.join(evaluation_dir, i, item))
        # Resizing the image to (256, 256)
        img = img.resize((256, 256))
        # Appending image to the image list
        image.append(img)
        # Converting image to array
        img = np.asarray(img, dtype=np.float32)
        # Normalizing the image
        img = img / 255.0
        # Reshaping the image into a 4D array
        img = img.reshape(-1, 256, 256, 3)
        # Making prediction with the model
        predict = model.predict(img)
        # Getting the index corresponding to the highest value in the prediction
        predict = np.argmax(predict)
        # Appending the predicted class to the list
        prediction.append(labels[predict])
        # Appending original class to the list
        original.append(i)
        
# visualizing the results
import random
fig=plt.figure(figsize = (100,100))
for i in range(20):
    j = random.randint(0,len(image))
    fig.add_subplot(20,1,i+1)
    plt.xlabel("Prediction -" + prediction[j] +"   Original -" + original[j])
    plt.imshow(image[j])
fig.tight_layout()
plt.show()

# Print the classification report
print(classification_report(np.assarray(original), np.array(prediction)))

# plot confusion matrix
plt.figure(figsize=(20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')