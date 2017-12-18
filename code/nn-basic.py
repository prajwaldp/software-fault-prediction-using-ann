'''
A neural network built using numpy. It trains on the camel 1.6 data-set
and the accuracy is derived from it's predictions on the test split (30 %)
of the data-set.
'''

import numpy as np


def nonlin(x, deriv=False):
    '''Sigmoid non-linear function

    Applies the sigmoid non-linear function on `x`

    Args:
        x (np.ndarray): A numpy array
        derive (boolean): if true, the derivative of the non-linear function is applied on `x`
    '''

    if deriv == True:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


# Random seed
np.random.seed(42)

# Read data
data = np.loadtxt('../data/camel-1.6.csv', skiprows=1,
                  delimiter=',',usecols=(3, 4, 5, 6, 7, 8, -1),
                  converters={ -1: lambda x: 0 if int(x) == 0 else 1 })

# Split data into train and test
np.random.shuffle(data)
split_index = round(data.shape[0] * 0.7)
X_train = data[:split_index, :-1]
y_train = data[:split_index, -1:]
X_test = data[split_index:, :-1]
y_test = data[split_index:, -1:]

# Random Weight Initialization
w0 = 2 * np.random.random((6, 12)) - 1
w1 = 2 * np.random.random((12, 1)) - 1

# Hyper-parameters
learning_rate = 0.01
epochs = 1000000

for j in range(epochs):

    # Feed Forward
    layer0 = X_train
    layer1 = nonlin(np.dot(layer0, w0))
    layer2 = nonlin(np.dot(layer1, w1))

    # Back Propagation
    layer2_error = y_train - layer2
    layer2_delta = learning_rate * layer2_error * nonlin(layer2, deriv=True)
    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = learning_rate * layer1_error * nonlin(layer1, deriv=True)

    # Weight Updating
    w1 += layer1.T.dot(layer2_delta)
    w0 += layer0.T.dot(layer1_delta)

    if j % 100000 == 0:
        print('Error: {}'.format(np.mean(np.abs(layer2_error))))


# Prediction
layer0 = X_test
layer1 = nonlin(np.dot(layer0, w0))
y_pred = nonlin(np.dot(layer1, w1))


# Compute accuracy of the model
correct = 0
incorrect = 0

for i in range(len(y_test)):
    if np.round(y_pred[i]) == y_test[i]:
        correct += 1
    else:
        incorrect += 1

print('Accuracy: {:.2f}%'.format(correct * 100 / (correct + incorrect)))


# Output

# Error: 0.6418343623700322
# Error: 0.241984802997631
# Error: 0.20175685973828136
# Error: 0.19343417517527117
# Error: 0.18906891790605854
# Error: 0.18734524975062278
# Error: 0.18307317711739549
# Error: 0.18075306721062184
# Error: 0.17888788051817153
# Error: 0.17792151273610735
# Accuracy: 78.20%
