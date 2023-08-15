import utils
import copy
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset, MAHNOBDataset
from bci_models import Conformer, DeepConvNet, ShallowConvNet


def load_model(model_name):
    if model_name == "conformer":
        model = Conformer(n_classes=2)
    elif model_name == "deepnet":
        model = DeepConvNet(num_classes=2)
    elif model_name == "shallownet":
        model = ShallowConvNet(num_classes=2)
    return model


def load_deap_dataset(model_name: str, label: str = "valence"):
    root_path = "./bci_data/normalize"
    io_path = f"./bci_data/{model_name}"
    dataset = DEAPDataset(
        io_path=io_path,
        root_path=root_path,
        offline_transform=transforms.MeanStdNormalize(axis=1),
        online_transform=transforms.Compose([transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose(
            [
                transforms.Select(label),
                transforms.Binary(5.0),
            ]
        ),
        num_worker=4,
    )
    return dataset


def load_mahnob_dataset(model_name: str, label: str = "feltVlnc"):
    root_path = "/bci_data/dataset/Mahnob_HCI_tagging/Sessions/"
    io_path = f"./bci_data_tl/{model_name}"
    dataset = MAHNOBDataset(
        io_path=io_path,
        root_path=root_path,
        offline_transform=transforms.MeanStdNormalize(axis=1),
        online_transform=transforms.Compose([transforms.ToTensor(), transforms.To2d()]),
        label_transform=transforms.Compose(
            [
                transforms.Select(label),
                transforms.Binary(5.0),
            ]
        ),
        num_worker=4,
    )
    return dataset


def load_split_dataset(model_name="conformer", label="valence"):
    dataset = load_deap_dataset(model_name=model_name, label=label)
    train_info, attack_info = utils.split_dataset_by_trial(dataset=dataset)

    train_dataset = copy.deepcopy(dataset)
    train_dataset.info = train_info

    attack_dataset = copy.deepcopy(dataset)
    attack_dataset.info = attack_info
    return train_dataset, attack_dataset


if __name__ == "__main__":
    train, test = load_split_dataset()
    print(len(train))
