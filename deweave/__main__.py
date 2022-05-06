import sys
import numpy as np
import cv2 as cv
import scipy.fftpack as fft
from pathlib import Path
from matplotlib import pyplot as plt

plt.rcParams.update(
    {
        "figure.figsize": (7, 7),
        "figure.dpi": 150,
        "font.family": "serif",
    }
)


def main(image_file: str | Path):
    img = cv.imread(str(Path(image_file).resolve()))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = (img / img.max()).astype("float32")
    fig = plt.figure(figsize=(7 * 2, 7))

    ax = fig.add_subplot(221)
    ax.imshow(img, "gray")
    ax.set_title("image")
    ax.axis("off")

    ax = fig.add_subplot(222)
    img_fft = fft.fft2(img)
    power = np.log(fft.fftshift(abs(img_fft)))
    ax.imshow(power, "gray")
    ax.set_title("image FT")
    ax.axis("off")

    ax = fig.add_subplot(223)
    mask = np.zeros_like(img, dtype="uint8")
    mask[371:434, 1256:1326] = 1
    mask[456:474, 1327:1385] = 1
    mask[486:523, 1434:1491] = 1
    mask[605:644, 1333:1400] = 1
    ax.imshow(mask, "gray")
    ax.set_title("mask")
    ax.axis("off")

    ax = fig.add_subplot(224)
    mask_inverse = np.logical_not(mask).astype("uint8")
    clean = fft.fftshift(img_fft) * mask_inverse
    clean = fft.ifft2(clean)
    clean = abs(clean)
    clean = (clean / clean.max()).astype("float32")
    ax.imshow(clean, "gray")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
