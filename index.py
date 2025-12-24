import numpy as np

n = [2, 3, 3, 1] # architecture of the network (number of neurons in each layer)

W1 = np.random.randn(n[1], n[0])
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)

X = np.array([
    [150, 70],
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])

A0 = X.T

y = np.array([
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    0,
    1,
    0
])
m = 10

Y = y.reshape(n[3], m)

def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

# feed-forward process
m = 10

Z1 = W1 @ A0 + b1
assert Z1.shape == (n[1], m)
A1 = sigmoid(Z1)

Z2 = W2 @ A1 + b2
assert Z2.shape == (n[2], m)
A2 = sigmoid(Z2)

Z3 = W3 @ A2 + b3
assert Z3.shape == (n[3], m)
A3 = sigmoid(Z3)

print(A3.shape)
y_hat = A3
print(y_hat)