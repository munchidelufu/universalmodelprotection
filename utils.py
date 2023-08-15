import os
import time
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl
import torchvision.transforms as transforms
from torch.optim import Adam
from torch import Module, device
from sklearn import model_selection
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def seed_everything(seed: int):
    """Set a random seed for reproducibility of results.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timer(func: callable) -> callable:
    """A timer decorator to time the execution of a function.

    Args:
        func (_type_): Functions that require timing.
    Returns:
        function: The decorated function.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start):.2f} seconds to execute.")
        return result

    return wrapper


def save_result(path: str, data: object) -> None:
    """Serialize data from memory to local.

    Args:
        path (str): local path, no exist then new path.
        data (object): data waiting to serialize
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    #
    with open(path, mode="wb") as file:
        pkl.dump(obj=data, file=file)
        print(f"save to {path} successfully!")


def load_result(path: str) -> object:
    """Deserialize data from local to memory.

    Args:
        path (str): local path.

    Raises:
        FileNotFoundError: path spell error or unexist.

    Returns:
        object: object
    """
    if not os.path.exists(path):
        print(f"{path} not found!")
    with open(path, "rb") as file:
        data = pkl.load(file=file)
    return data


def denormalize(batch_data: torch.Tensor):
    transform_reverse = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in CIFAR_STD]),
            transforms.Normalize(mean=[-m for m in CIFAR_MEAN], std=[1, 1, 1]),
        ]
    )
    data = transform_reverse(batch_data)
    data = torch.clamp(input=data, min=0.0, max=1.0)
    return data


def normalize(batch_data: torch.Tensor):
    normal = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    data = normal(batch_data)
    return data


def calculate_auc(list_a, list_b) -> int:
    """Calculate the area under the ROC curve (AUC)

    Args:
        list_a (_type_): containing the predicted values ​for the samples.
        list_b (_type_): containing the predicted values ​​for the samples.

    Returns:
        int: value of auc.
    """
    l1, l2 = len(list_a), len(list_b)
    y_true, y_score = [], []
    for _ in range(l1):
        y_true.append(0)
    for _ in range(l2):
        y_true.append(1)
    y_score.extend(list_a)
    y_score.extend(list_b)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
    return round(auc(fpr, tpr), 2)


def split_dataset_by_trial(
    dataset,
    split_path="./bci_data/split_dataset_by_trial",
    split_ratio=[0.5, 0.5],
    shuffle=True,
):
    """According to the proportion, the data set is divided into two parts according to the trial group

    Args:
        dataset (_type_): Dataset waiting for split.
        split_path (str, optional): The path to data partition information. Defaults to "./data/split_dataset_by_trial".
        split_ratio (list, optional): Split proportion. Defaults to [0.5, 0.5].
        shuffle (bool, optional): Whether to disrupt. Defaults to True.

    Returns:
        _type_: Data information of trian and attack
    """
    if not os.path.exists(split_path):
        os.makedirs(split_path, exist_ok=True)

        info = dataset.info

        trial_ids = list(set(info["trial_id"]))
        train_trial_ids, attack_trial_ids = model_selection.train_test_split(
            trial_ids, test_size=split_ratio[0], random_state=2023, shuffle=shuffle
        )

        train_info = []
        for train_trial_id in train_trial_ids:
            train_info.append(info[info["trial_id"] == train_trial_id])
        train_info = pd.concat(train_info, ignore_index=True)

        attack_info = []
        for attack_trial_id in attack_trial_ids[:-1]:
            attack_info.append(info[info["trial_id"] == attack_trial_id])
        attack_info = pd.concat(attack_info, ignore_index=True)

        train_info.to_csv(os.path.join(split_path, "train.csv"), index=False)
        attack_info.to_csv(os.path.join(split_path, "attack.csv"), index=False)

    train_info = pd.read_csv(os.path.join(split_path, "train.csv"))
    attack_info = pd.read_csv(os.path.join(split_path, "attack.csv"))
    return train_info, attack_info


def train(
    model: Module,
    data_loader: DataLoader,
    loss_fn: Module,
    optimizer: Adam,
) -> float:
    total_batches = len(data_loader)
    loss_record = []
    #
    model.train()
    for _, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)
        output = model(b_x)
        loss = loss_fn(output, b_y)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def test(model: Module, data_loader: DataLoader, loss_fn: Module):
    total_sample_num = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct_sample_num = 0, 0
    with torch.no_grad():
        for batch_data in data_loader:
            b_x = batch_data[0].to(device)
            b_y = batch_data[1].to(device)
            output = model(b_x)
            test_loss += loss_fn(output, b_y).item()

            correct_sample_num += (
                (output.argmax(1) == b_y).type(torch.float).sum().item()
            )
    #
    test_loss /= num_batches
    #
    test_accuracy = correct_sample_num / total_sample_num
    return test_loss, test_accuracy
