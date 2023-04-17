import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, Dropout, Activation, GlobalAveragePooling1D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.xception import Xception
from sklearn.utils.class_weight import compute_class_weight

# Define the number of GPUs you want to use
num_gpus = 2

# Create a MirroredStrategy
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:%d" % i for i in range(num_gpus)])

# Load feature
train_features = np.load('train_fets_ccii.npy', mmap_mode='r')
train_labels = np.load('train_labels_ccii.npy', mmap_mode='r')

test_features = np.load('test_fet_t2tvit_ccii.npy', mmap_mode='r')
test_labels = np.load('test_labels_ccii.npy', mmap_mode='r')

# Convert label into one-hot-encoder
# train_labels = to_categorical(train_labels, num_classes=3)
# test_labels = to_categorical(test_labels, num_classes=3)

# Define the shape of your feature vector
input_shape = (train_features.shape[1],)

# Load the pre-trained model
with strategy.scope():
    # Create the DenseNet201 model
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    # Add a fully connected layer on top of the base model
    inputs = Input(shape=input_shape)
    # Add a fully connected layer with residual connection
    x = layers.Dense(1024, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    residual = layers.Dense(1024)(inputs)
    x = layers.add([x, residual])
    x = layers.Activation('relu')(x)
    # Add a skip connection to the output layer
    x = layers.concatenate([x, inputs])
    outputs = layers.Dense(3, activation='softmax')(x)
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './checkpoint/denset2t_cnn/{epoch:02d}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)
# Train the model
with strategy.scope():
    model.fit(train_features, train_labels, 
              batch_size=32, epochs=300, 
              validation_data=(test_features, test_labels), 
              callbacks=[model_checkpoint_callback], verbose=1)
