import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plot():
    def __init__(self, history_1, history_2, history_3, history_4, history_5, history_6, epochs):
        plt.plot(np.arange(1, epochs + 1), history_1['loss'], label = 'Basis Encoding (0-1)')
        plt.plot(np.arange(1, epochs + 1), history_2['loss'], label = 'Basis Encoding (3-6)')
        plt.plot(np.arange(1, epochs + 1), history_3['loss'], label = 'Amplitude Encoding (0-1)')
        plt.plot(np.arange(1, epochs + 1), history_4['loss'], label = 'Amplitude Encoding (3-6)')
        plt.plot(np.arange(1, epochs + 1), history_5['loss'], label = 'FRQI (0-1)')
        plt.plot(np.arange(1, epochs + 1), history_6['loss'], label = 'FRQI (3-6)')
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, 2))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig('D:/OneDrive/Desktop/{}.png'.format('Loss'), bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        plt.plot(np.arange(1, epochs + 1), history_1['accuracy'], label = 'Basis Encoding (0-1)')
        plt.plot(np.arange(1, epochs + 1), history_2['accuracy'], label = 'Basis Encoding (3-6)')
        plt.plot(np.arange(1, epochs + 1), history_3['accuracy'], label = 'Amplitude Encoding (0-1)')
        plt.plot(np.arange(1, epochs + 1), history_4['accuracy'], label = 'Amplitude Encoding (3-6)')
        plt.plot(np.arange(1, epochs + 1), history_5['accuracy'], label = 'FRQI (0-1)')
        plt.plot(np.arange(1, epochs + 1), history_6['accuracy'], label = 'FRQI (3-6)')
            
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, 2))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig('D:/OneDrive/Desktop/{}.png'.format('Accuracy'), bbox_inches = 'tight', pad_inches = 0)
        plt.show()
    
def main():
    history_1 = pd.read_csv('D:/OneDrive/Desktop/Tasks/1. Image Encoding/QML Basis Encoding (0-1).csv')
    history_2 = pd.read_csv('D:/OneDrive/Desktop/Tasks/1. Image Encoding/QML Basis Encoding (3-6).csv')
    history_3 = pd.read_csv('D:/OneDrive/Desktop/Tasks/1. Image Encoding/QML Amplitude Encoding (0-1).csv')
    history_4 = pd.read_csv('D:/OneDrive/Desktop/Tasks/1. Image Encoding/QML Amplitude Encoding (3-6).csv')
    history_5 = pd.read_csv('D:/OneDrive/Desktop/Tasks/1. Image Encoding/QML FRQI (0-1).csv')
    history_6 = pd.read_csv('D:/OneDrive/Desktop/Tasks/1. Image Encoding/QML FRQI (3-6).csv')
    Plot(history_1, history_2, history_3, history_4, history_5, history_6, epochs = 20)

if __name__ == '__main__':
    main()
