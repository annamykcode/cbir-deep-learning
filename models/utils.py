import numpy as np
import torch


def fit(dataloader, model, loss_fn, optimizer, device, print_loss=False):
    """ Fit deep-learning model.

    Args:
        dataloader:
            pytorch DataLoader object.
        model:
            deep-learning model, as pytorch object.
        loss_fn:
            loss function, as pytorch object.
        optimizer:
            optimizer function, as pytorch object.
        device:
            device where the deep-learning model will run (cpu, gpu), as string.
        print_loss:
            print loss on every batch, as boolean (default False)
    """
    size = len(dataloader.dataset)
    model.train()  # put on train mode
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)

        # compute prediction
        pred = model(X)

        # compute loss
        loss = loss_fn(pred, Y)

        # reset the gradients
        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update parameters
        optimizer.step()

        if print_loss and batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def predict(dataloader, model, loss_fn, device):
    """ Predict with deep-learning model.

    Args:
        dataloader:
            pytorch DataLoader object.
        model:
            deep learning model, as pytorch object.
        loss_fn:
            loss function, as pytorch object.
        device:
            device where the deep-learning model will run (cpu, gpu), as string.

    Returns:
         test loss, as float.
         predictions, as a list of integers.
         ground truth, as a list of integers.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss = 0

    pred_concat = []
    y_concat = []

    model.eval()  # put on evaluation mode
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, Y).item()

            # predictions to one-hot vectors
            for label in pred.argmax(1):
                pred_concat.append(label.item())

            # ground truth to one-hot vectors
            for label in Y:
                y_concat.append(label.item())

    test_loss /= num_batches

    return test_loss, pred_concat, y_concat


def label_to_vector(label, mapping):
    """ Translate a string label to one-hot vector.

    Args:
        label:
            class label, as string.
        mapping:
            string (label) to index mapping, as dictionary.

    Returns:
        label one-hot vector, as numpy array.
    """
    vec = np.zeros(len(mapping))
    vec[mapping[label]] = 1

    return vec
