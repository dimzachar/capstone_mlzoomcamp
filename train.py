
import warnings

# suppress warning messages from TensorFlow
warnings.filterwarnings("ignore", module="tensorflow")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import load_img
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorboard import notebook

from tensorflow.keras.optimizers import SGD
from keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from keras.layers import Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

import tensorflow.keras.applications.xception as xc
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
import sklearn
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from time import time

from PIL import Image
from typing import List
import random
import plotly.express as px
import collections
import os



tf.random.set_seed(0)
print(f"pandas version  : {pd.__version__}")
print(f"numpy version   : {np.__version__}")
print(f"seaborn version : {sns.__version__}")
print(f"scikit-learn version  : {sklearn.__version__}")
print(f"tensorflow version  : {tf.__version__}")



# Set paths
train_dir = './Images/train/'
val_dir = './Images/val/'
test_dir = './Images/test/'

num_epochs = 100
img_input = 299

input_shape = (img_input, img_input, 3)

learning_rate = 0.0001
size_inner = 64
droprate = 0.2

base_model = Xception(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)



def build_model(base_model, input_shape, droprate, learning_rate, size_inner,
                include_dropout):
    """
    Creates a model for image classification using a specified pre-trained model as a base model,
    with some additional inner layers and a final output layer.

    Parameters:
    - base_model: keras.Model
        The pre-trained model to use as the base model.
    - input_shape: tuple
        The shape of the input data (e.g. (150, 150, 3) for images with 150x150 resolution and 3 color channels).
    - learning_rate: float, optional
        The learning rate for the Adam optimizer.
    - size_inner: int, optional
        The number of units in the inner dense layer.
    - droprate: float, optional
        The dropout rate for the dropout layer.
    
    Returns:
    - model: keras.Model
        The compiled model.
    """

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=input_shape)

    base = base_model(inputs, training=False)

    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)

    if include_dropout:
        drop = keras.layers.Dropout(droprate)(inner)
        outputs = keras.layers.Dense(1, activation='sigmoid')(drop)
    else:
        outputs = keras.layers.Dense(1, activation='sigmoid')(inner)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(learning_rate)
    model.summary()
    print("  Training starting..." )
    return model


def checkpoint_weights(
    model_name: str = 'model',
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'log_dir',
    delete_files: bool = True,
    restore_from_checkpoint: bool = False,
    callbacks: List[tf.keras.callbacks.Callback] = None
) -> List[tf.keras.callbacks.Callback]:
    """

        Creates a ModelCheckpoint and EarlyStopping callback for use during model training.

        The ModelCheckpoint callback saves the best model weights to a file with a name that includes the epoch number
        and the validation accuracy. The EarlyStopping callback stops the training if the validation accuracy does not
        improve after two epochs.

        If the delete_files flag is set to True, this function will delete all files in the checkpoint_dir directory
        that contain model_name in their names. If delete_files is not set or is set to False, this function will
        append an underscore and a number to the end of model_name if there are any files in the checkpoint_dir
        directory that contain model_name in their names, where the number is equal to the number of files in the
        directory.
        
        If the restore_from_checkpoint flag is set to True, this function will find the latest checkpoint file in the
    checkpoint_dir directory and load the model weights from it.

        Args:
            model_name: str, the name of the model.
            checkpoint_dir: str, the directory where the checkpoints will be saved.
            log_dir: str, the directory where the TensorBoard logs will be saved.
            delete_files: bool, flag to indicate whether to delete existing checkpoints.
            restore_from_checkpoint: bool, flag to indicate whether to restore the model weights from a checkpoint file.
            callbacks: List[tf.keras.callbacks.Callback], a list of callbacks to use during training. If this parameter
                   is not provided, the function will create the ModelCheckpoint, EarlyStopping, and TensorBoard
                   callbacks.
    
    Returns:
        List[tf.keras.callbacks.Callback], a list containing the ModelCheckpoint, EarlyStopping, and TensorBoard
        callbacks.
    """

    if callbacks is None:
        # Check if the checkpoint_dir directory exists and create it if it does not
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Check if the log_dir directory exists and create it if it does not
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # If restore_from_checkpoint is True, find the latest checkpoint file in the checkpoint_dir directory
        # and load the model weights from it
        if restore_from_checkpoint:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_dir) if model_name in f
            ]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                model.load_weights(
                    os.path.join(checkpoint_dir, latest_checkpoint))
        if delete_files:
            files = [f for f in os.listdir(checkpoint_dir) if model_name in f]
            for file in files:
                os.remove(os.path.join(checkpoint_dir, file))
        elif not delete_files:
            files = [f for f in os.listdir(checkpoint_dir) if model_name in f]
            if files:
                model_name = model_name + '_' + str(len(files))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                checkpoint_dir,
                f'{model_name}-{{val_accuracy:.3f}}-{{epoch:02d}}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max')

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, restore_best_weights=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        callbacks = [cp_callback, early_stopping, tensorboard_callback]

    return callbacks

def train(rates, callbacks, epochs, input_shape, include_dropout):
    #########################################
    """
    Train and evaluate a model for multiple learning rates.
    
    Parameters:
    - learning_rates: list of float values
        List of learning rates to use for training the model.
    - checkpoint_weights: function
        Function to create a ModelCheckpoint callback for saving the model weights during training.
    - epochs: int
        Number of epochs to train the model.
    - input_shape: tuple
        Shape of the input data.
    - include_dropout: bool
        Indicates whether or not to include dropout layers in the model.
        
    Returns:
    - scores: dictionary
        Dictionary containing the training history for each learning rate. The keys of the dictionary are
        the learning rates and the values are the training history objects returned by the fit method.
    - model: tf.keras.Model
        Trained model with the best validation accuracy.
"""
    #########################################
    scores = {}

    # Compile model's training function into a static graph @tf.function
    def train_step(inputs, labels):
        """
        Perform a single training step.
        
        Parameters:
        - inputs: numpy array
            Input data for the training step.
        - labels: numpy array
            Labels for the input data.
        
        Returns:
        - loss_value: float
            The loss value resulting from the training step.
        """
        # with tf.GradientTape() as tape:
        #     logits = model(inputs, training=True)
        #     loss_value = loss_fn(labels, logits)
        # grads = tape.gradient(loss_value, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # return loss_value

    for learning_rate in rates:
        for droprate in droprates:
            model = build_model(base_model, input_shape, droprate,
                                learning_rate, size_inner, include_dropout)

            # Use fit's steps_per_epoch argument to control the number of batches processed per epoch
            history = model.fit(train_generator,
                                epochs=num_epochs,
                                validation_data=validation_generator,
                                callbacks=callbacks,
                                steps_per_epoch=len(train_generator))
            hyperparameters = f'learning_rate={learning_rate}, droprate={droprate}'

            scores[hyperparameters] = history
    return scores, model



# # Final Model


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_input,
                                                                 img_input),
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    shuffle=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = val_datagen.flow_from_directory(val_dir,
                                                       target_size=(img_input,
                                                                    img_input),
                                                       batch_size=32,
                                                       class_mode='binary',
                                                       shuffle=True)





rates = [0.0001]

droprates = [0.2]

# Get the default callbacks from the checkpoint_weights function
callbacks = checkpoint_weights()


scores, model = train(rates, callbacks, num_epochs, input_shape, True)


print("  Training done..." )


print("  Testing..." )

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                       target_size=(img_input,
                                                                    img_input),
                                                       batch_size=32,
                                                       class_mode='binary',
                                                       shuffle=False)





# evaluate_model(model, test_generator)



def evaluate_model(model, test_generator):
  # Make predictions on the test set
  y_pred = model.predict(test_generator)
  y_true = test_generator.labels

    # Convert the continuous labels to binary labels
  y_pred = np.where(y_pred > 0.5, 1, 0)
  y_true = np.where(y_true > 0.5, 1, 0)

  # Compute the test accuracy
  test_acc = accuracy_score(y_true, y_pred)

  # Count the number of test examples and the number of correctly classified test examples
  num_test_examples = test_generator.n
  #  num_correct = sum(y_true == y_pred)

  num_correctly_classified = test_acc * num_test_examples
  # Start the timer
  start_time = time()

  # Evaluate the model
  model.evaluate(test_generator)

  # Stop the timer
  end_time = time()

  # Print the evaluation results
  print("  Evaluation results:")
  print("  Number of test examples: {}".format(num_test_examples))
  print("  Number of correctly classified test examples: {}".format(num_correctly_classified))
  print("  Number of incorrectly classified test examples: {}".format(num_test_examples - num_correctly_classified))
  print("  Test accuracy: {:.2f}%".format(test_acc * 100))
    
  # Compute the performance metrics
  precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')  
  print("  Precision: {:.2f}%".format(precision * 100))
  print("  Recall: {:.2f}%".format(recall * 100))
  print("  F1: {:.2f}%".format(f1 * 100))
  print("  Time taken to evaluate the model: {:.2f} seconds".format(end_time - start_time))
    
    
  # Print the classification report
  print(classification_report(y_true, y_pred))

  # Plot the confusion matrix
  confusion = confusion_matrix(y_true, y_pred)
  sns.heatmap(confusion, annot=True, fmt="d")
  plt.show()



# Find the best model
checkpoint_dir = './checkpoints'
model_name = 'model' 

# List all files in the checkpoint_dir directory
files = [f for f in os.listdir(checkpoint_dir) if model_name in f]

# Extract the accuracy values from the file names
accuracies = [float(f.split('-')[1]) for f in files]

# Find the maximum accuracy value
best_accuracy = max(accuracies)

# Find the index of the file with the best accuracy value
best_index = accuracies.index(best_accuracy)

# Get the name of the best model file
best_model = files[best_index]

# Load the best model using the load_model function
best_model = keras.models.load_model(os.path.join(checkpoint_dir, best_model))


print("  Evaluating best model..." )
evaluate_model(best_model, test_generator)

print("  Saving tflite model..." )
# Convert the best model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
model_lite = converter.convert()
with open('model.tflite', 'wb') as f_out:
    f_out.write(model_lite)


print("  Finished" )




