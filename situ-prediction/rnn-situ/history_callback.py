import time
from keras.callbacks import Callback

class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        return super().on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        return super().on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))
        self.times.append(time.time() - self.epoch_time_start)