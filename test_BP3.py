import numpy as np
import prettytable as pt


# take sigmoid as activate function
def sigmoid(x: float) -> float:
    return .5 * (1 + np.tanh(.5 * x))


# define derivative of sigmoid function
def sigmoid_derivative(x: float) -> float:
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int):
        self.table_x = []
        self.table_y = []

        # init 4 hidden layers randomly
        self.hidden_layers = np.random.randint(1, 100, size=4)
        self.layers_size = [input_size] + list(self.hidden_layers) + [output_size]

        # init weight matrix and b
        self.weights = []
        self.biases = []
        for i in range(len(self.layers_size) - 1):
            w = np.random.randn(self.layers_size[i], self.layers_size[i + 1])
            b = np.zeros((1, self.layers_size[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def feedforward(self, X: float):
        # 前向传播
        a = X
        for i in range(len(self.layers_size) - 2):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = sigmoid(z)
        z_last = np.dot(a, self.weights[-1]) + self.biases[-1]
        y_hat = sigmoid(z_last)
        return y_hat

    def train(self, X, y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                # forward
                a = [X[i]]
                for j in range(len(self.layers_size) - 2):
                    z = np.dot(a[j], self.weights[j]) + self.biases[j]
                    a.append(sigmoid(z))
                z_last = np.dot(a[-1], self.weights[-1]) + self.biases[-1]
                y_hat = sigmoid(z_last)

                # backward
                error = y_hat - y[i]
                delta = error * sigmoid_derivative(y_hat)

                deltas = [delta]
                for k in range(len(self.layers_size) - 2, 0, -1):
                    delta = np.dot(delta, self.weights[k].T) * sigmoid_derivative(a[k])
                    deltas.insert(0, delta)

                # renew weights and b
                for l in range(len(self.layers_size) - 1):
                    self.weights[l] -= learning_rate * np.dot(a[l].reshape(-1, 1), deltas[l].reshape(1, -1))
                    self.biases[l] -= learning_rate * deltas[l]

            # print loss every 10 epochs
            if epoch % 10 == 0:
                loss = np.mean((self.feedforward(X) - y) ** 2)
                self.table_x.append(epoch)
                self.table_y.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        # prediction
        y_pred = self.feedforward(X)
        return np.argmax(y_pred, axis=1)

    def show_train_cfg(self):
        import matplotlib.pyplot as plt
        plt.axis([0, 100, 0, 1.0])  # （0, 100）range of x， （0, 1.0）range of y
        plt.xticks([i * 10 for i in range(0, 11)], fontsize=26)  # show x-axis
        plt.yticks([i * 0.1 for i in range(0, 11)], fontsize=26)  # show y-axis
        plt.plot(self.table_x, self.table_y, color="r", linestyle="-", linewidth=4, label="label")
        plt.title("Learning error convergence curves", color='k', fontsize=26)
        plt.ylabel("Learning error rate", fontsize=26)  # name of y-axis
        plt.xlabel("Learning epoch", fontsize=26)  # name of x-axis
        plt.show()

    def test_dataset_load(self, testFile):
        # read test data
        data = np.loadtxt(testFile)
        # split data and labels
        X_ = data[:, :-1]
        y_ = data[:, -1]
        # turn labels into one-hot
        y_onehot = np.zeros((y_.shape[0], 6))
        for i in range(y_.shape[0]):
            y_onehot[i, int(y_[i])] = 1
        print("Testing dataset read in successfully!")
        return X_, y_onehot


if __name__ == "__main__":
    # read in training data
    training_data_path = "dataset/fire_train.txt"
    test_data_path = "dataset/fire_test.txt"
    data = np.loadtxt(training_data_path)

    # split the data and labels
    X = data[:, :-1]
    y = data[:, -1]

    # turn labels into one-hot
    y_onehot = np.zeros((y.shape[0], 3))  # 3 is the output num****
    for i in range(y.shape[0]):
        y_onehot[i, int(y[i])] = 1

    # create bp nn
    model = NeuralNetwork(input_size=3, output_size=3)  # 3 is the input num and output num****

    # training process
    model.train(X, y_onehot, epochs=1000, learning_rate=0.01)
    model.show_train_cfg()

    # use model to predict
    test_data1, test_label = model.test_dataset_load(test_data_path)
    y_pred = model.predict(test_data1)
    pred = ""
    label_1 = 0
    corr_num = 0
    wrong_num = 0
    tb = pt.PrettyTable()
    tb.field_names = ["Test data idx", "Predict type", "True type", "Prediction status"]
    for idx, i in enumerate(y_pred):
        label = test_label[idx]
        label_num = np.argmax(label)
        if y_pred[idx] == label_num:
            pred = "[Prediction correct]"
            corr_num = corr_num + 1
        else:
            wrong_num = wrong_num + 1
            pred = "[**Prediction error**]"

        tb.add_row([idx + 1, (y_pred[idx] + 1), (label_num + 1), pred])
    print('\n')
    print(tb)

    print('=' * 14, 'BP Test result', '=' * 14)
    print('Number of correct predictions: {}'.format(corr_num))
    print('Number of test samples: {}'.format(200))
    print('Correct prediction rate: {}%'.format(round(((corr_num) / 200), 2) * 100))