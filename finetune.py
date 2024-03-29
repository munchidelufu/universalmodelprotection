import os
import torch
import argparse
from torch.nn import Module
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheeg.model_selection import KFoldGroupbyTrial

import utils
import bci_loader


def fit(
    index: int,
    model: Module,
    kfold: KFoldGroupbyTrial,
    dataset: Dataset,
    args: argparse.ArgumentParser,
):
    model.to(device)
    #
    if index < 5:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(
            model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    loss_fn = CrossEntropyLoss()
    #
    save_dir = f"./bci_model/finetune/{args.model_name}/finetune_{index}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir.replace("/bci_model/", "/bci_result/"))
    best_acc = 0
    for split_idx, (train_dataset, test_dataset) in enumerate(kfold.split(dataset)):
        #
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        #
        best_test_acc = 0
        for epoch_id in range(args.epochs):
            #
            train_loss = utils.train(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            writer.add_scalar(
                "Loss/Train", train_loss, (split_idx * args.epochs) + epoch_id
            )
            #
            test_loss, test_acc = utils.test(
                model=model, data_loader=test_loader, loss_fn=loss_fn
            )
            writer.add_scalar(
                "Loss/Test", test_loss, (split_idx * args.epochs) + epoch_id
            )
            #
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                torch.save(
                    model.state_dict(), os.path.join(save_dir, f"model_{split_idx}.pth")
                )
        #
        model.load_state_dict(
            torch.load(os.path.join(save_dir, f"model_{split_idx}.pth"))
        )
        _, test_acc = utils.test(model, test_loader, loss_fn)
        writer.add_scalar("Acc/Test", test_acc, split_idx + 1)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="conformer")
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--label", type=str, default="valence", choices=["valence", "arousal"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    #
    utils.seed_everything(2023)
    #
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)

    _, attack_dataset = bci_loader.load_split_dataset(
        model_name=args.model_name, label=args.label
    )
    #
    kfold = KFoldGroupbyTrial(
        n_splits=5, shuffle=True, split_path=f"./bci_data/{args.model_name}/attack"
    )
    #
    for index in range(0, 10):
        #
        model = bci_loader.load_model(model_name=args.model_name)
        model.load_state_dict(
            torch.load(
                f=f"./bci_model/source/{args.model_name}/model_best.pth",
                map_location=device,
            )
        )
        fit(
            index=index,
            model=model,
            kfold=kfold,
            dataset=attack_dataset,
            args=args,
        )
