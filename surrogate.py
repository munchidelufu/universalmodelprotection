import argparse
import os
import torch
from torch.optim import Adam
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheeg.model_selection import KFoldGroupbyTrial
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
import bci_loader


def train(
    tea_model: Module,
    stu_model: Module,
    data_loader: DataLoader,
    loss_fn: Module,
    optimizer: Adam,
) -> float:
    total_batches = len(data_loader)
    tea_model.to(device)
    stu_model.to(device)
    #
    tea_model.eval()
    stu_model.train()

    loss_record = []
    for batch_id, batch_data in enumerate(data_loader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)

        t_output = tea_model(b_x)
        pred = t_output.argmax(1)
        s_output = stu_model(b_x)
        loss = loss_fn(s_output, pred)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        loss = loss.detach().item()
        loss_record.append(loss)
        if batch_id % 20 == 0:
            accuracy = 100 * (s_output.argmax(1) == b_y).sum().item() / len(b_y)
    mean_train_loss = sum(loss_record) / total_batches
    return mean_train_loss


def fit(
    index: int,
    tea_model: Module,
    stu_model: Module,
    dataset: Dataset,
    kfold: KFoldGroupbyTrial,
    args: argparse.ArgumentParser,
):
    tea_model.to(device)
    stu_model.to(device)
    #
    optimizer = Adam(stu_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    loss_fn = CrossEntropyLoss()
    #
    save_dir = f"./bci_model/surrogate/{args.teacher_name}_{args.student_name}/surrogate_{index}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir.replace("/bci_model/", "/bci_result/"))
    best_acc = 0
    for split_idx, (train_dataset, test_dataset) in enumerate(kfold.split(dataset)):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        #
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
            train_loss = train(
                tea_model=tea_model,
                stu_model=stu_model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch_id=epoch_id,
            )
            writer.add_scalar(
                "Loss/Train", train_loss, (split_idx * args.epochs) + epoch_id
            )
            #
            test_loss, test_acc = utils.test(
                model=stu_model, data_loader=test_loader, loss_fn=loss_fn
            )
            writer.add_scalar(
                "Loss/Test", test_loss, (split_idx * args.epochs) + epoch_id
            )
            # 更新学习率
            lr_scheduler.step(test_loss)
            #
            if test_acc > best_test_acc:
                best_test_acc = test_acc

                torch.save(
                    stu_model.state_dict(),
                    os.path.join(save_dir, f"model_{split_idx}.pth"),
                )
        #
        stu_model.load_state_dict(
            torch.load(os.path.join(save_dir, f"model_{split_idx}.pth"))
        )
        loss, test_acc = utils.test(stu_model, test_loader, loss_fn=loss_fn)
        writer.add_scalar("Acc/Test", test_acc, split_idx + 1)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(stu_model.state_dict(), os.path.join(save_dir, "model_best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_name", type=str, default="conformer")
    parser.add_argument(
        "--student_name",
        type=str,
        choices=["deepnet", "shallownet"],
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--label", type=str, default="valence", choices=["valence", "arousal"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    #
    utils.seed_everything(2023)
    #
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    #
    tea_model = bci_loader.load_model(model_name=args.teacher_name)
    tea_model.load_state_dict(
        torch.load(
            f"./bci_model/source/{args.teacher_name}/model_best.pth",
            map_location=device,
        )
    )
    #
    train_dataset, _ = bci_loader.load_split_dataset(
        model_name=args.teacher_name, label=args.label
    )
    kfold = KFoldGroupbyTrial(
        n_splits=5,
        split_path=f"./bci_data/{args.teacher_name}/train",
    )

    for index in range(2):
        stu_model = bci_loader.load_model(model_name=args.student_name)
        fit(
            index=index,
            tea_model=tea_model,
            stu_model=stu_model,
            dataset=train_dataset,
            kfold=kfold,
            args=args,
        )
