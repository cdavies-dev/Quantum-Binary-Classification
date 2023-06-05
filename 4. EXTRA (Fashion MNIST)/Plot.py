import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Plot():
    def __init__(self, history_1, history_2, epochs):
        plt.plot(np.arange(1, epochs + 1), history_1['loss'], label = 'MNIST')
        plt.plot(np.arange(1, epochs + 1), history_2['loss'], label = 'Fashion MNIST')
            
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, 2))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig('D:/OneDrive/Desktop/{}.png'.format('Loss'), bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        plt.plot(np.arange(1, epochs + 1), history_1['accuracy'], label = 'MNIST')
        plt.plot(np.arange(1, epochs + 1), history_2['accuracy'], label = 'Fashion MNIST')
            
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xticks(np.arange(0, epochs + 1, 2))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig('D:/OneDrive/Desktop/{}.png'.format('Accuracy'), bbox_inches = 'tight', pad_inches = 0)
        plt.show()
    
def main():
    history_1 = pd.read_csv('D:/OneDrive/Desktop/Tasks/QML Amplitude Encoding (0-1) + Ansatz Variation 1, 2 layers + MNIST.csv')
    history_2 = pd.read_csv('D:/OneDrive/Desktop/Tasks/QML Amplitude Encoding (0-1) + Ansatz Variation 1, 2 layers + Fashion MNIST.csv')
    Plot(history_1, history_2, epochs = 20)

if __name__ == '__main__':
    main()
