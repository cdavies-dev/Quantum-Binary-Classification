import numpy as np
import pandas as pd
import pennylane as qml
import tensorflow as tf
from matplotlib import pyplot as plt

tf.keras.backend.set_floatx('float64')

class Preprocessing():

    def __init__(self, dataset):
        (x_train, y_train), (x_test, y_test) = dataset

        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

        x_train, self.y_train = self.filter_classes(x_train, y_train)
        x_test, self.y_test = self.filter_classes(x_test, y_test)
        
        x_train = tf.image.resize(x_train, (28,28)).numpy()
        x_test = tf.image.resize(x_test, (28,28)).numpy()

        self.plot(x_train[0, :, :, 0], '28x28 Training Example')

        self.x_train = x_train[:,:,:,:].reshape((-1,784))
        self.x_test = x_test[:,:,:,:].reshape((-1,784))
  
    def filter_classes(self, x, y):
        keep = (y == 0) | (y == 1)
        x, y = x[keep], y[keep]
        y = y == 0
        return x, y
    
    def plot(self, image, title, vmin = None, vmax = None):
        plt.imshow(image, cmap = 'Greys')
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(title, fontsize = 20)
        plt.show()

class QuantumNeuralNetwork():
    def __init__(self, x_train, y_train, x_test, y_test, epochs, layers):
        self.epochs = epochs
        model = self.generate_model(x_train, layers = layers)
        model.compile(
            loss = tf.keras.losses.Hinge(),
            optimizer = tf.keras.optimizers.Adam(),
            metrics = [self.accuracy])
        history, results, model = self.train_model(model, x_train, y_train, x_test, y_test)
        model.save('QML Amplitude Encoding (0-1) + Ansatz Variation 1, 4 layers.h5')
        history_df = pd.DataFrame(history.history)
        with open('QML Amplitude Encoding (0-1) + Ansatz Variation 1, 4 layers.csv', mode = 'w') as f:
            history_df.to_csv(f)
        print('\nModel complete!\n')

    def generate_model(self, x_train, layers):
        n_qubits = 10
        n_layers = layers
        dev = qml.device('default.qubit', wires = n_qubits)

        @qml.qnode(dev, diff_method = 'adjoint')
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires = range(10), pad_with = 0.0, normalize = True)

            for ii in range(n_qubits):
                qml.RY(np.pi * inputs[ii], wires = ii)

            for jj in range(n_layers):
                for ii in range(n_qubits - 1):
                    qml.RZ(weights[jj, 2 * ii, 0], wires = 0)
                    qml.RY(weights[jj, 2 * ii, 1], wires = 0)
                    qml.RZ(weights[jj, 2 * ii, 2], wires = 0)

                    qml.RZ(weights[jj, 2 * ii + 1, 0], wires = ii + 1)
                    qml.RY(weights[jj, 2 * ii + 1, 1], wires = ii + 1)
                    qml.RZ(weights[jj, 2 * ii + 1, 2], wires = ii + 1)

                    qml.CNOT(wires = [ii + 1, 0])
                    
                qml.RZ(weights[jj, 2 * (n_qubits - 1), 0], wires = 0)
                qml.RY(weights[jj, 2 * (n_qubits - 1), 1], wires = 0)
                qml.RZ(weights[jj, 2 * (n_qubits - 1), 2], wires = 0)

            return qml.expval(qml.PauliZ(0))

        weight_shapes = {'weights': (n_layers, 2 * (n_qubits - 1) + 1, 3)}
        qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim = 1, name = 'quantumLayer')
        inputs = tf.keras.Input(shape = (784,), name = 'inputs')
        outputs = qlayer(inputs)
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'QNN')
        model.predict(x_train[0,:].reshape((1,-1)))
        print(model.summary())
        
        return model

    def train_model(self, model, x_train, y_train, x_test, y_test):
        EPOCHS = self.epochs
        BATCH_SIZE = 32
        NUM_EXAMPLES = 512

        y_train_hinge = 2.0 * y_train - 1.0

        x_train_sub = x_train[:NUM_EXAMPLES,:]
        y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

        x_test_sub = x_test[:,:]
        y_test_sub = y_test[:]

        qnn_history = model.fit(
            x_train_sub,
            y_train_hinge_sub,
            batch_size = BATCH_SIZE,
            epochs = EPOCHS,
            verbose = 1)#,
            #validation_data = (x_test_sub, y_test_sub))
        
        qnn_results = model.evaluate(x_test_sub, y_test_sub)
        
        return qnn_history, qnn_results, model

    def accuracy(self, y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)

        return tf.reduce_mean(result)
    
def main():
    layers = 4
    epochs = 20
    dataset = Preprocessing(tf.keras.datasets.mnist.load_data())
    x_train, y_train, x_test, y_test = dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test
    QuantumNeuralNetwork(x_train, y_train, x_test, y_test, epochs, layers)

if __name__ == '__main__':
    main()