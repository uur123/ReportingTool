
import numpy as np


def boxcount(Z, k):

    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1
    )

    return len(np.where((S > 0) & (S < k * k))[0])


def fractal_dimension(image):

    Z = image < 128

    p = min(Z.shape)

    n = 2 ** np.floor(np.log(p) / np.log(2))

    n = int(n)

    Z = Z[:n, :n]

    sizes = 2 ** np.arange(
        int(np.log(n) / np.log(2)),
        1,
        -1
    )

    counts = []

    for size in sizes:

        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(
        np.log(sizes),
        np.log(counts),
        1
    )

    return -coeffs[0]
