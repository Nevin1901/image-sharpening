import numpy as np
import torch
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import pickle
from model import Sharp

class BlurredImageDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda"), torch.flatten(y.to("cuda"), start_dim=1)
        pred = model(X)

        plt.imshow(np.reshape(pred[0].cpu().detach().numpy(), (28, 28)))
        plt.show()
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cuda"), torch.flatten(y.to("cuda"), start_dim=1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()


    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f}")


def get_data(save):
    files = ["x_train.pkl", "y_train.pkl", "x_test.pkl", "y_test.pkl"]
    transform = transforms.Compose([
        transforms.PILToTensor()
        ])
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    if save == True:
        training_data = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                )

        test_data = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                )

        for x in training_data:
            blur = cv.blur(np.asarray(x[0]), (1, 1))
            blur = torch.from_numpy(blur).float()
            y = transform(x[0]).float()
            x_train.append(blur)
            y_train.append(y[0])

        for x in test_data:
            blur = cv.blur(np.asarray(x[0]), (7, 7))
            blur = torch.from_numpy(blur).float()
            y = transform(x[0]).float()
            x_test.append(blur)
            y_test.append(y[0])

        with open('x_train.pkl', 'wb') as f:
            pickle.dump(x_train, f)

        with open('y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)

        with open('x_test.pkl', 'wb') as f:
            pickle.dump(x_test, f)

        with open('y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)

    else:
        with open('x_train.pkl', 'rb') as f:
            x_train = pickle.load(f)

        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)

        with open('x_test.pkl', 'rb') as f:
            x_test = pickle.load(f)

        with open('y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)

    return x_train, y_train, x_test, y_test



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save = False

    print("getting data")
    x_train, y_train, x_test, y_test = get_data(save)
    # train_data, test_data = get_data(save)

    train_dataset = BlurredImageDataset(x_train, y_train)
    test_dataset = BlurredImageDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, pin_memory=True)

    print("starting model")
    model = Sharp().to(device)

    learning_rate = 1e-1
    batch_size = 64
    epochs = 40

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 1e-1,
            weight_decay=1e-8)

    for i in range(epochs):
        print(f"Epoch {i+1}\n")

#         for (image, _) in train_dataloader:
#             image = image.to("cuda")
#             print("got here")
#             pred = model(image)
# 
#             # cv.imshow("frame1", np.reshape(pred.cpu(), (28, 28)))
#             loss = loss_fn(pred, image)
# 
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print("ok")

        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)


#         cv.imshow("frame1", x_test[0])
#         cv.imshow("frame", x_train[0])
#         key = cv.waitKey()
# 
#         if key == 113:
#           quit()

        # plt.imshow(out)
        # plt.show()

    # for idx, x in enumerate(x_train):
    #     cv.imshow("frame1", x)
    #     cv.imshow("frame", fx_train[idx])

    #     key = cv.waitKey()

    #     if key == 113:
    #         quit()

