import pandas
import numpy as np
import tensorflow as tf
# A training window object designed for use with keras time series prediction
'''
input_width = size of input for 1 prediction
label_width  = size of output for 1 prediction
shift = distance between input and prediction
training_df = Dataframe containing testing data
validation_df = Dataframe containing validation data
testing_df = Dataframe containing trainging data
label_col_names = List of column names that will be output as predictions
'''

class TrainingWindow():
    def __init__(self, input_width, label_width, shift, training_df, validation_df, testing_df, label_col_names = None):
        # Input variables
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.training_df = training_df
        self.validation_df = validation_df
        self.testing_df = testing_df
        self.label_col_names = label_col_names

        # Calculated Variables
        self.label_col_indices = {}
        self.col_indices = {}
        n = 0
        if label_col_names is not None:
            for name in label_col_names:
                self.label_col_indices[name] = n
                n += n
        n = 0
        for name in training_df.columns:
            self.col_indices[name] = n
            n += n

        self.size = input_width+shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.size)[self.input_slice]

        self.label_start = self.size - label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indicies = np.arange(self.size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:,self.input_slice, :]
        labels = features[:,self.labels_slice, :]
        if self.label_col_names is not None:
            labels = tf.stack([labels[:, :, self.col_indices[name]] for name in self.label_col_names], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, stride, shuff, batch_size):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length= self.size,
            sequence_stride=stride,
            shuffle=shuff,
            batch_size = batch_size
        )
        ds = ds.map(self.split_window)
        return ds

    # @property
    # def train(self):
    #     return self.make_dataset(self.train_df)
    #
    # @property
    # def val(self):
    #     return self.make_dataset(self.val_df)
    #
    # @property
    # def test(self):
    #     return self.make_dataset(self.test_df)



