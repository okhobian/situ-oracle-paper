from logging import raiseExceptions
from matplotlib import pyplot as plt

class PLOT():
    def __init__(self):
        self.plt = plt
        
    def add_figure(self, data, title, xlabel, ylabel, legend):
        self.plt.plot(data)
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.legend(legend, loc='upper left')
        self.plt.figure()

    def add_multi_data_figure(self, multi_data, title, xlabel, ylabel, legend):
        
        if len(multi_data) != len(legend):
            print("ERROR: len(legend) != len(data)")
        
        for data in multi_data:
            self.plt.plot(data)
            
        self.plt.title(title)
        self.plt.xlabel(xlabel)
        self.plt.ylabel(ylabel)
        self.plt.legend(legend, loc='upper left')
        self.plt.figure()

    def show_all(self):
        self.plt.show()