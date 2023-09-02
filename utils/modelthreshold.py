import tensorflow as tf


class ThresholdCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, threshold, logs=None):
        self.model.threshold = threshold
