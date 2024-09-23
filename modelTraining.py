import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (Input, Bidirectional, Conv2D,
                          BatchNormalization, Dropout, MaxPooling2D, LSTM, Reshape, Dense, Lambda, Activation)


# Define the preprocessing function
def preprocess_image(image):
    height, width = image.shape
    processed_image = np.full((64, 256), 255, dtype=np.uint8)  # Create a white image

    if width > 256:
        image = image[:, :256]

    if height > 64:
        image = image[:64, :]

    processed_image[:height, :width] = image
    return cv2.rotate(processed_image, cv2.ROTATE_90_CLOCKWISE)


# Load data
train_data = pd.read_csv('/Users/begumtamer/Desktop/archive/written_name_train_v2.csv')
valid_data = pd.read_csv('/Users/begumtamer/Desktop/archive/written_name_validation_v2.csv')


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx in range(6):
    img_path = os.path.join('/Users/begumtamer/Desktop/archive/train_v2/train', train_data.loc[idx, 'FILENAME'])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(train_data.loc[idx, 'IDENTITY'], fontsize=12)
        ax.axis('off')
    else:
        print(f"Warning: Unable to read image at {img_path}")

plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()

print(f"Missing values in training labels: {train_data['IDENTITY'].isnull().sum()}")
print(f"Missing values in validation labels: {valid_data['IDENTITY'].isnull().sum()}")

train_data.dropna(inplace=True)
valid_data.dropna(inplace=True)

unreadable_images = train_data[train_data['IDENTITY'] == 'UNREADABLE'].reset_index(drop=True)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx in range(6):
    img_filename = unreadable_images.loc[idx, 'FILENAME']
    img_path = os.path.join('/Users/begumtamer/Desktop/archive/train_v2/train', img_filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(unreadable_images.loc[idx, 'IDENTITY'], fontsize=12)
        ax.axis('off')
    else:
        print(f"Warning: Unable to read image at {img_path}")

plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()

train_data = train_data[train_data['IDENTITY'] != 'UNREADABLE']
valid_data = valid_data[valid_data['IDENTITY'] != 'UNREADABLE']

train_data['IDENTITY'] = train_data['IDENTITY'].str.upper()
valid_data['IDENTITY'] = valid_data['IDENTITY'].str.upper()

train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)

# I will be using 10% of the full dataset:
# train_size = 33,029
# valid_size = 4,128


# Number of samples
num_train_samples = 33029
num_valid_samples = 4128


# Paths for images
train_image_dir = '/Users/begumtamer/Desktop/archive/train_v2/train/'
valid_image_dir = '/Users/begumtamer/Desktop/archive/validation_v2/validation/'


# Process and preprocess images
train_images = []
for i in range(num_train_samples):
    img_filename = train_data.loc[i, 'FILENAME']
    img_path = os.path.join(train_image_dir, img_filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = preprocess_image(img) / 255.0
        train_images.append(img)
    else:
        print(f"Warning: Unable to read image at {img_path}")


valid_images = []
for i in range(num_valid_samples):
    img_filename = valid_data.loc[i, 'FILENAME']
    img_path = os.path.join(valid_image_dir, img_filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = preprocess_image(img) / 255.0
        valid_images.append(img)
    else:
        print(f"Warning: Unable to read image at {img_path}")


# Convert lists to numpy arrays and reshape
train_images = np.array(train_images).reshape(-1, 256, 64, 1)
valid_images = np.array(valid_images).reshape(-1, 256, 64, 1)


# Define charset and labels
charset = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_label_length = 24
num_chars = len(charset) + 1  # Including CTC blank
num_pred_timestamps = 64


def encode_labels(label):
    return np.array([charset.find(char) for char in label])


def decode_labels(encoded):
    return ''.join([charset[ch] for ch in encoded if ch != -1])

train_labels = np.full((num_train_samples, max_label_length), -1)
train_label_lengths = np.zeros((num_train_samples, 1))
train_input_lengths = np.full((num_train_samples, 1), num_pred_timestamps - 2)
train_targets = np.zeros((num_train_samples,))

for idx in range(num_train_samples):
    label = train_data.loc[idx, 'IDENTITY']
    train_label_lengths[idx] = len(label)
    train_labels[idx, :len(label)] = encode_labels(label)

valid_labels = np.full((num_valid_samples, max_label_length), -1)
valid_label_lengths = np.zeros((num_valid_samples, 1))
valid_input_lengths = np.full((num_valid_samples, 1), num_pred_timestamps - 2)
valid_targets = np.zeros((num_valid_samples,))

for idx in range(num_valid_samples):
    label = valid_data.loc[idx, 'IDENTITY']
    valid_label_lengths[idx] = len(label)
    valid_labels[idx, :len(label)] = encode_labels(label)

print('True label : ', train_data.loc[100, 'IDENTITY'], '\ntrain_labels : ', train_labels[100], '\ntrain_label_lengths : ', train_label_lengths[100], '\ntrain_input_lengths : ', train_input_lengths[100])


# Model definition
input_layer = Input(shape=(256, 64, 1), name='input')

conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_layer)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv2 = Dropout(0.3)(conv2)

reshaped = Reshape(target_shape=(64, 1024))(conv2)
dense1 = Dense(64, activation='relu', kernel_initializer='he_normal')(reshaped)

bi_lstm = Bidirectional(LSTM(256, return_sequences=True))(dense1)

output_layer = Dense(num_chars, kernel_initializer='he_normal')(bi_lstm)
softmax_output = Activation('softmax')(output_layer)


# Model for CTC loss
def compute_ctc_loss(args):
    predictions, true_labels, pred_lengths, label_lengths = args
    predictions = predictions[:, 2:, :]  # Discard initial outputs
    return tf.keras.backend.ctc_batch_cost(true_labels, predictions, pred_lengths, label_lengths)

true_labels_input = Input(name='true_labels', shape=[max_label_length], dtype='float32')
input_lengths = Input(name='input_lengths', shape=[1], dtype='int64')
label_lengths = Input(name='label_lengths', shape=[1], dtype='int64')

ctc_loss = Lambda(compute_ctc_loss, output_shape=(1,), name='ctc_loss')([softmax_output, true_labels_input, input_lengths, label_lengths])
final_model = Model(inputs=[input_layer, true_labels_input, input_lengths, label_lengths], outputs=ctc_loss)


# Define a named function for the loss instead of using a lambda
def ctc_loss_function(y_true, y_pred):
    return y_pred

final_model.compile(optimizer=Adam(learning_rate=0.0001), loss={'ctc_loss': ctc_loss_function})


# Create a ModelCheckpoint callback
checkpoint = ModelCheckpoint("model_checkpoint.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')


# Train the model with the checkpoint and early stopping callbacks
final_model.fit(
    x=[train_images, train_labels, train_input_lengths, train_label_lengths],
    y=train_targets,
    validation_data=([valid_images, valid_labels, valid_input_lengths, valid_label_lengths], valid_targets),
    epochs=50,
    batch_size=128,
    callbacks=[checkpoint]
)


# Save the model architecture to JSON
model_json = final_model.to_json()
with open("../../Desktop/handwriting_recognition_model.json", "w") as json_file:
    json_file.write(model_json)


# Save the model weights to HDF5
final_model.save_weights("handwriting_recognition_model.weights.h5")


# Create a prediction model
prediction_model = Model(inputs=input_layer, outputs=softmax_output)


# Predict using the prediction model
predictions = prediction_model.predict(valid_images)


# Save the prediction model architecture to JSON
prediction_model_json = prediction_model.to_json()
with open("handwriting_recognition_prediction_model.json", "w") as json_file:
    json_file.write(prediction_model_json)


# Save the prediction model weights
prediction_model.save_weights("handwriting_recognition_prediction_model.weights.h5")
print(f"Shape of predictions: {predictions.shape}")


# Assuming predictions shape is (num_samples, time_steps, num_classes)
input_length = np.ones(predictions.shape[0]) * predictions.shape[1]


# Decode the predictions using CTC
decoded_predictions = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=input_length, greedy=True)[0][0])


# Decode labels
decoded_labels = [decode_labels(decoded_predictions[i]) for i in range(num_valid_samples)]


# Evaluate accuracy
true_labels = valid_data['IDENTITY'].values[:num_valid_samples]

correct_chars = 0
total_chars = 0
correct_words = 0

for idx in range(num_valid_samples):
    pred = decoded_labels[idx]
    true = true_labels[idx]
    total_chars += len(true)
    correct_chars += sum(1 for p, t in zip(pred, true) if p == t)
    if pred == true:
        correct_words += 1

accuracy = correct_chars / total_chars
word_accuracy = correct_words / num_valid_samples

print(f'Character Accuracy: {accuracy:.4f}')
print(f'Word Accuracy: {word_accuracy:.4f}')
