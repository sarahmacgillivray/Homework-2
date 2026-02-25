'''
 Author: Dr. Souradeep Dutta
 
 This code is provided for demonstration and instructional purposes
 for CPEN 355 at the University of British Columbia (UBC).
 
 Course website:
 https://souradeep-dutta-01.github.io/ubc-cpen-355-website
 
'''

import numpy as np
import matplotlib.pyplot as plt


def generate_linearly_separable_data(n=80, seed=0):
    if n % 2 != 0:
        raise ValueError("n must be even to create a balanced dataset.")

    rng = np.random.default_rng(seed)

    w_true = rng.normal(size=2)
    b_true = rng.normal()



    X_pos = []
    X_neg = []
    margin = 0.2
    while len(X_pos) < n // 2 or len(X_neg) < n // 2:
        x = rng.uniform(-10, 10, size=2)
        score = float(np.dot(w_true, x) + b_true)
        if abs(score) < margin:
            continue
        if score > 0.0 and len(X_pos) < n // 2:
            X_pos.append(x)
        elif score < 0.0 and len(X_neg) < n // 2:
            X_neg.append(x)

    X = np.array(X_pos + X_neg)
    y = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))

    return X, y, w_true, b_true


def generate_linearly_separable_no_bias(n=80, seed=0):
    if n % 2 != 0:
        raise ValueError("n must be even to create a balanced dataset.")

    rng = np.random.default_rng(seed)

    w_true = rng.normal(size=2)

    X_pos = []
    X_neg = []
    margin = 0.2
    while len(X_pos) < n // 2 or len(X_neg) < n // 2:
        x = rng.uniform(-10, 10, size=2)
        score = float(np.dot(w_true, x))
        if abs(score) < margin:
            continue
        if score > 0.0 and len(X_pos) < n // 2:
            X_pos.append(x)
        elif score < 0.0 and len(X_neg) < n // 2:
            X_neg.append(x)

    X = np.array(X_pos + X_neg)
    y = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))

    return X, y, w_true



def generate_circles_dataset(n=200, center=(0.0, 0.0), radius=1.0, noise=0.05, seed=0):
    if n % 2 != 0:
        raise ValueError("n must be even to create a balanced dataset.")

    rng = np.random.default_rng(seed)
    center = np.array(center, dtype=float)

    X_inside = []
    X_outside = []
    while len(X_inside) < n // 2 or len(X_outside) < n // 2:
        x = rng.uniform(-2.0, 2.0, size=2)
        r = np.linalg.norm(x - center)
        if r < radius and len(X_inside) < n // 2:
            X_inside.append(x)
        elif r >= radius and len(X_outside) < n // 2:
            X_outside.append(x)

    X_inside = np.array(X_inside) + rng.normal(scale=noise, size=(n // 2, 2))
    X_outside = np.array(X_outside) + rng.normal(scale=noise, size=(n // 2, 2))

    X = np.vstack([X_inside, X_outside])
    y = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))

    return X, y


def generate_circles_with_band_gap(
    n=200,
    center=(0.0, 0.0),
    radius=1.0,
    noise=0.05,
    band_c=1.0,
    seed=0,
):
    if n % 2 != 0:
        raise ValueError("n must be even to create a balanced dataset.")

    X, y = generate_circles_dataset(
        n=n, center=center, radius=radius, noise=noise, seed=seed
    )

    x_vals = X[:, 0]
    y_vals = X[:, 1]
    in_band = (y_vals >= -x_vals - band_c) & (y_vals <= -x_vals + band_c)
    remove_mask = in_band & (y < 0)
    keep_mask = ~remove_mask

    X_filtered = X[keep_mask]
    y_filtered = y[keep_mask]

    return X_filtered, y_filtered


def main():

    X_band, y_band = generate_circles_with_band_gap()
    pos_b = y_band > 0
    neg_b = y_band < 0

    plt.figure(figsize=(6, 5))
    plt.scatter(X_band[pos_b, 0], X_band[pos_b, 1], c="tab:blue", label="inside (+1)")
    plt.scatter(X_band[neg_b, 0], X_band[neg_b, 1], c="tab:orange", label="outside (-1)")
    plt.title("Circles Dataset with Band Gap")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
