
"""Tensorflow predictor script."""

import pickle
import subprocess
import sys
import fire
import pandas as pd
import tensorflow as tf
import datetime
import os

def load_dataset(pattern, window_size=30, batch_size=16, shuffle_buffer=100):
    """
    Description:  
    Input: 
      - series:
      - window_size:
      - batch_size: the batches to use when training
      -shuffle_buffer: size buffer, how data will be shuffled

    Output:
    """
    
    # read data
    data = pd.read_csv(pattern)
    time = np.array(data.times)
    series = np.array(data.values)[:,1].astype('float32')
    
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1])) # x and y (last one)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def train_evaluate(training_dataset_path, 
                   # validation_dataset_path,
                   window_size,
                   batch_size,
                   epochs, lr,
                   # num_train_examples, num_evals, 
                   output_dir):
    """
    Description: train script
    """
    
    EPOCHS = epochs
    LR = lr
    
    l0 = tf.keras.layers.Dense(2*window_size+1, input_shape=[window_size], activation='relu')
    l2 = tf.keras.layers.Dense(1)
    model = tf.keras.models.Sequential([l0, l2])
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3)
    optimizer = tf.keras.optimizers.SGD(lr=LR, momentum=0.9)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae'])
    
    # load data
    trainds = load_dataset(pattern=training_dataset_path, window_size=window_size, batch_size=batch_size)
    # evalds = load_dataset(pattern=validation_dataset_path, mode='eval')
    
    history = model.fit(trainds, epochs=EPOCHS, verbose=0)
    
    EXPORT_PATH = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    tf.saved_model.save(obj=model, export_dir=EXPORT_PATH)  # with default serving function
    
    print("Exported trained model to {}".format(EXPORT_PATH))
    
if __name__ == '__main__':
    fire.Fire(train_evaluate)
