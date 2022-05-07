import sys
import numpy as np
import cv2 as cv
from numpy import ndarray
from scipy import fft
from pathlib import Path
from matplotlib import pyplot as plt

plt.rcParams.update(
    {
        "figure.figsize": (7, 7),
        "figure.dpi": 150,
        "font.family": "serif",
    }
)

Pathlike = str | Path


def normalize_image(image: ndarray) -> ndarray:
    img_type = str(image.dtype)
    if img_type.startswith("uint"):
        type_max: int = np.iinfo(img_type).max
    else:
        type_max: float = max(1, image.max())
    return (image / type_max).astype("float32")


class DeWeave:
    def __init__(self, image_file: Pathlike):
        filename = Path(image_file).resolve()
        img = cv.imread(str(filename))
        self.NUM_CHANNELS = 3
        self.img: ndarray = normalize_image(img)
        self.img_fft: ndarray = np.zeros_like(img)
        self.filename = filename.name

    def to_fourier(self, save: bool = True):
        img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        img_fft = fft.fft2(img, axes=(0, 1))
        if save:
            power = fft.fftshift(img_fft, axes=(0, 1))
            power = np.log10(abs(power))
            power = np.round(power / power.max() * (2**8 - 1))
            power = abs(power).astype("uint8")
            cv.imwrite("fftpower.png", power)
        self.img_fft = img_fft

    def apply_mask(self, mask: Pathlike):
        mask = cv.imread(str(Path(mask).resolve()))
        mask = np.round(normalize_image(mask)).astype("uint8")
        mask_inverse = np.logical_not(mask).astype("uint8")
        clean = fft.fftshift(self.img_fft, axes=(0, 1)) * mask_inverse
        clean = fft.ifft2(clean, axes=(0, 1))
        clean = abs(clean)
        clean = ((clean / clean.max()) * (2**16 - 1)).astype("uint16")
        filename = "".join(self.filename.split(".")[:-1])
        clean = cv.cvtColor(clean, cv.COLOR_RGB2BGR)
        cv.imwrite(f"{filename}-clean.png", clean)


if __name__ == "__main__":
    dw = DeWeave(sys.argv[2])
    match sys.argv[1]:
        case "fft":
            dw.to_fourier()
        case "apply":
            dw.to_fourier(False)
            dw.apply_mask(sys.argv[3])
        case _:
            raise ValueError(f"unrecognized command {sys.argv[1]}")
