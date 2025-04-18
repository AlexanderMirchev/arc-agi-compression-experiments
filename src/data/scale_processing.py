import numpy as np
from scipy.stats import mode

def padding(X, height=30, width=30, direction='norm'):
    h = X.shape[0]
    w = X.shape[1]

    a = (height - h) // 2
    aa = height - a - h

    b = (width - w) // 2
    bb = width - b - w

    if direction == 'norm':
        X_pad = np.pad(X, pad_width=((a, aa), (b, bb)), mode='constant')

    # Reverse padding for rescaling
    else:
        if height == 30 and width == 30:
            X_pad = X[:, :]
        elif height == 30:
            X_pad = X[:, abs(bb):b]
        elif width == 30:
            X_pad = X[abs(aa):a, :]
        else:
            X_pad = X[abs(aa):a, abs(bb):b]

    return X_pad

# Scaling of the ARC matrices using the Kronecker Product, retaining all the information
def scaling(X, height=30, width=30, direction='norm'):
    h = height/X.shape[0]
    w = width/X.shape[1]
    d = np.floor(min(h, w)).astype(int)

    X_scaled = np.kron(X, np.ones((d, d)))

    if direction == 'norm':
        return padding(X_scaled, height, width).astype(int)

    # Retain information for reverse scaling
    else:
        return d, X_scaled.shape

# Reverse scaling of the ARC matrices for final computations
def reverse_scaling(X_orig, X_pred):
    d, X_shape = scaling(X_orig, 30, 30, direction='rev') # get scaling information
    X_pad_rev = padding(X_pred, X_shape[0], X_shape[1], direction='rev') # reverse padding

    mm = X_shape[0] // d
    nn = X_shape[1] // d
    X_sca_rev = X_pad_rev[:mm*d, :nn*d].reshape(mm, d, nn, d)

    X_rev = np.zeros((mm, nn)).astype(int)
    for i in range(mm):
        for j in range(nn):
            X_rev[i,j] = mode(X_sca_rev[i,:,j,:], axis=None, keepdims=False)[0]

    return X_rev