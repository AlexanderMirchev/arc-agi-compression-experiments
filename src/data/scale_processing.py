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

def scaling(X, height=30, width=30, direction='norm'):
    h = height/X.shape[0]
    w = width/X.shape[1]
    d = np.floor(min(h, w)).astype(int)

    X_scaled = np.kron(X, np.ones((d, d)))

    if direction == 'norm':
        return padding(X_scaled, height, width).astype(int)

    else:
        return d, X_scaled.shape

def reverse_scaling(X_orig, X_pred):
    d, X_shape = scaling(X_orig, 30, 30, direction='rev')  # d and padded shape

    # Resize or pad prediction to match padded scaled shape
    X_pad_rev = padding(X_pred, X_shape[0], X_shape[1], direction='rev')

    mm = X_shape[0] // d
    nn = X_shape[1] // d

    X_rev = np.zeros((mm, nn), dtype=int)

    for i in range(mm):
        for j in range(nn):
            block = X_pad_rev[i*d:(i+1)*d, j*d:(j+1)*d]

            # Safeguard: if block size isn't d x d (e.g., on edge), skip or handle
            if block.size == 0:
                continue

            # Use rounded mode (or fallback to rounded mean)
            block_rounded = np.rint(block).astype(int).flatten()
            m = mode(block_rounded, keepdims=False)
            X_rev[i, j] = m.mode if m.count > 0 else int(np.round(np.mean(block)))

    return X_rev