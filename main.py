import numpy as np
import torch
import cv2 as cv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from model import Sharp

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device}")

    training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            # transform=ToTensor(),
            )

    test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            # transform=ToTensor(),
            )

    x_train = []
    y_train = []

    fx_train = []

    print("blurring image")
    for x in training_data:
        blur = cv.blur(np.asarray(x[0]), (7, 7))
        fx_train.append(blur)

        x_train.append(np.asarray(x[0]))
        y_train.append(x[1])
        # cv.imshow("frame1", np.asarray(x[0][0]))
        # cv.imshow("frame", blur)
        # key = cv.waitKey()

        # if key == 113:
            # quit()
        # plt.imshow(out)
        # plt.show()

    # for idx, x in enumerate(x_train):
    #     cv.imshow("frame1", x)
    #     cv.imshow("frame", fx_train[idx])

    #     key = cv.waitKey()

    #     if key == 113:
    #         quit()

