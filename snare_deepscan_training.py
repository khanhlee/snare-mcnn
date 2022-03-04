#!/usr/bin/env python
# coding: utf-8

MAXSEQ      = 4753
NUM_FEATURE = 20
BATCH_SIZE  = 128

NUM_CLASSES = 2
CLASS_NAMES = ['SNARE','NON-SNARE']
EPOCHS      = 10


import csv
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import math

from sklearn import metrics
from tensorflow.keras import Model, layers

def load_ds(file_path):
  NUM_SAMPLES = 0
  with open(file_path) as file:
    NUM_SAMPLES = sum(1 for row in file)

  data = np.zeros((NUM_SAMPLES, MAXSEQ * NUM_FEATURE), dtype=np.float32 )
  labels = np.zeros((NUM_SAMPLES, 1), dtype=np.uint8 )

  with open(file_path) as file:
    file = csv.reader(file, delimiter = ',')
    m = 0
    for row in file:
      labels[m] = int(row[-1])
      data[m] = np.array(row[:-1]).astype('float32')
      m += 1
      print(f"\rReading {file_path}...\t{m}/{NUM_SAMPLES}", end='')
  print('\tDone')
  return data, labels


data_dir = ''

x_train, y_train = load_ds(os.path.join(data_dir, 'dataset', 'pssm.cv.csv'))
x_test, y_test = load_ds(os.path.join(data_dir, 'dataset', 'pssm.cv.csv'))

# Add a channels dimension
x_train = np.reshape( x_train, [-1,1, MAXSEQ, NUM_FEATURE] )
x_test = np.reshape( x_test, [-1,1, MAXSEQ, NUM_FEATURE] )

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")

print(f"Train label shape: {y_train.shape}")
print(f"Test label shape: {y_test.shape}")

# Convert to categorical labels
y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test,NUM_CLASSES)

class DeepScan(Model):
  def __init__(self,
               input_shape=(1,MAXSEQ, NUM_FEATURE),
               window_sizes=[8,12,16,20,24,28,32,36],
               num_filters=256,
               num_hidden=512):
    super(DeepScan, self).__init__()
    # Add input layer
    self.input_layer = tf.keras.Input(input_shape)
    self.window_sizes = window_sizes
    self.conv2d = []
    self.maxpool = []
    self.flatten = []
    for window_size in self.window_sizes:
      self.conv2d.append(layers.Conv2D(
        filters=num_filters,
        kernel_size=(1,window_size),
        activation=tf.nn.relu,
        padding='valid',
        bias_initializer=tf.constant_initializer(0.1),
        kernel_initializer=tf.keras.initializers.GlorotUniform()
      ))
      self.maxpool.append(layers.MaxPooling2D(
          pool_size=(1,MAXSEQ - window_size + 1),
          strides=(1,MAXSEQ),
          padding='valid'))
      self.flatten.append(layers.Flatten())
    self.dropout = layers.Dropout(rate=0.7)
    self.fc1 = layers.Dense(
      num_hidden,
      activation=tf.nn.relu,
      bias_initializer=tf.constant_initializer(0.1),
      kernel_initializer=tf.keras.initializers.GlorotUniform()
    )
    self.fc2 = layers.Dense(NUM_CLASSES,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    # Get output layer with `call` method
    self.out = self.call(self.input_layer)


  def call(self, x, training=False):
    _x = []
    for i in range(len(self.window_sizes)):
      x_conv = self.conv2d[i](x)
      x_maxp = self.maxpool[i](x_conv)
      x_flat = self.flatten[i](x_maxp)
      _x.append(x_flat)

    x = tf.concat(_x,1)
    x = self.dropout(x,training=training)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

def val_binary_init():
  fout = open(f'{LOG_DIR}/training.csv','w')
  fout.write(f'Epoch,TP,FP,TN,FN,Sens,Spec,Acc,MCC\n')
  fout.close()

def val_binary(epoch,logs):
  fout = open(f'{LOG_DIR}/training.csv','a')

  y_pred = model.predict(x_test)
  TN, FP, FN, TP =  metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)).ravel()

  Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
  Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
  Acc = (TP+TN)/(TP+FP+TN+FN)
  MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
  F1 = 2*TP/(2*TP+FP+FN)
  
  fout.write(f'{epoch + 1},{TP},{FP},{TN},{FN},{Sens:.4f},{Spec:.4f},{Acc:.4f},{MCC:.4f}\n')
  fout.close()


NUM_FILTER = 256
NUM_HIDDEN = 128
LOG_DIR    = f'/snare/logs'

model = DeepScan(
    num_filters=NUM_FILTER,
    num_hidden=NUM_HIDDEN,
    window_sizes=[8,16,24,32,40,48])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=x_train.shape)
model.summary()

# Train the model.
val_binary_init()

model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[
      tf.keras.callbacks.LambdaCallback(on_epoch_end=val_binary),
      tf.keras.callbacks.ModelCheckpoint(LOG_DIR + '/weights.{epoch:02d}', save_weights_only=True, monitor='val_loss', mode='max')
    ]
)

y_pred = model.predict(x_test)

out_file = open(os.path.join(data_dir, 'results', 'fold5.pssm.pred.txt'), 'w')
for row in y_pred:
    np.savetxt(out_file, row)
out_file.close()

out_file = open(os.path.join(data_dir, 'results', 'fold5.y_test.txt'), 'w')
for row in y_test:
    np.savetxt(out_file, row)
out_file.close()

