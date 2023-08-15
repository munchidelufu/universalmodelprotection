import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from torcheeg.datasets.module.emotion_recognition.deap import DEAPDataset

import utils
import bci_loader


def test(model: Module, data_loader: DataLoader):
    model.eval()
    total_sample_num = len(data_loader.dataset)
    correct_num = 0
    with torch.no_grad():
        for batch_data in data_loader:
            b_x = batch_data[0].to(device)
            b_y = batch_data[1].to(device)
            output = model(b_x)
            correct_num += (output.argmax(1) == b_y).type(torch.float).sum().item()
    #
    test_accuracy = correct_num / total_sample_num
    return round(test_accuracy, 3)


class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


def find_smallest_neuron(hook_list, prune_list):
    activation_list = []
    for j in range(len(hook_list)):
        # Get the output activation value of this batchdata in the current pruning module
        activation = hook_list[j].output
        for i in range(activation.shape[1]):
            # Average the activations of batchdata in that channel for that module.
            activation_channel = torch.mean(torch.abs(activation[:, i, :, :]))
            activation_list.append(activation_channel)

    activation_list1 = []
    activation_list2 = []
    # Find the neuron with the smallest activation value, return its index and judge whether it can be pruned
    for n, data in enumerate(activation_list):
        if n in prune_list:
            pass
        else:
            activation_list1.append(n)
            activation_list2.append(data)

    activation_list2 = torch.tensor(activation_list2)
    prune_num = torch.argmin(activation_list2)
    prune_idx = activation_list1[prune_num]
    prune_list.append(prune_idx)
    return prune_idx


def run_model(model, dataloader):
    model.eval()
    for i, (x, y) in enumerate(dataloader):
        y = y.long()
        b_x, b_y = x.to(device), y.to(device)
        output = model(b_x)


def idx_change(idx, neuron_num):
    total = 0
    for i in range(neuron_num.shape[0]):
        total += neuron_num[i]
        if idx < total:
            layer_num = i
            layer_idx = idx - (total - neuron_num[i])
            break
    return layer_num, layer_idx


def prune_neuron(mask_list, idx, neuron_num):
    # Determine which channel of which module the current neuron index corresponds to
    layer_num, layer_idx = idx_change(idx, neuron_num)
    # Set the mask matrix of the corresponding neuron to 0
    mask_list[layer_num].weight_mask[layer_idx] = 0


def finetune_step(model, dataloader, criterion):
    model.train()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _, batch_data in enumerate(dataloader):
        b_x = batch_data[0].to(device)
        b_y = batch_data[1].to(device)

        outputs = model(b_x)
        optimizer.zero_grad()
        loss = criterion(F.softmax(outputs, dim=1), b_y)
        loss.backward()
        optimizer.step()


def fine_prune(
    model: Module,
    train_dataset: DEAPDataset,
    attack_dataset: DEAPDataset,
):
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False
    )
    attack_loader = DataLoader(
        dataset=attack_dataset, batch_size=args.batch_size, shuffle=False
    )
    model.to(device)
    # Prune the conv2d of the model, and register the forward hook function of the corresponding module
    module_list = []
    neuron_num = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_list.append(module)
            neuron_num.append(module.out_channels)

    neuron_num = np.array(neuron_num)
    # list of alternative pruned neurons
    neuron_list = []
    mask_list = []
    for i in range(neuron_num.shape[0]):
        neurons = list(range(neuron_num[i]))
        neuron_list.append(neurons)
        # Add a mask tag to the weight of the model
        prune_filter = prune.identity(module_list[i], "weight")
        mask_list.append(prune_filter)
    # This list records the neurons that were pruned
    prune_list = []
    init_accuracy = test(model, train_loader)
    acc = []
    # The number of all candidate neurons
    total_length = 0
    for i in range(len(neuron_list)):
        total_length += len(neuron_list[i])
    print("Total number of neurons is", total_length)
    flag = True
    for i in range(int(np.floor(args.prune_amount * total_length))):
        if flag:
            hook_list = []
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    hook_list.append(FeatureHook(module))
            run_model(model, train_loader)
        # Find the neuron with the smallest activation value
        idx = find_smallest_neuron(hook_list, prune_list)
        # Find the channel of the corresponding module according to the neuron index and set its mask matrix to 0
        prune_neuron(mask_list, idx, neuron_num)
        if i % 20 == 0:
            finetune_step(model, attack_loader, criterion=torch.nn.CrossEntropyLoss())
        if i % 20 == 0:
            new_accuracy = test(model, train_loader)
            print(
                f"neurons removed: {i}, init_accuracy: {init_accuracy}, new_accuracy: {new_accuracy}"
            )
            acc.append([i, new_accuracy])

        if (
            np.floor(20 * i / total_length) - np.floor(20 * (i - 1) / total_length)
        ) == 1:
            iter = int(np.floor(20 * i / total_length))
            save_dir = f"./bci_model/fineprune/{args.model_name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            for hook in hook_list:
                hook.close()
            torch.save(model, os.path.join(save_dir, f"prune_{str(iter)}.pth"))
            print(f"neuron removed: {i}, saving model! Model number is: {iter}")
            flag = True
        else:
            flag = False
    results = np.array([acc])
    save_dir = f"./bci_result/fineprune/{args.model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    utils.save_result(os.path.join(save_dir, "pruned_accuracy.pkl"), data=results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="conformer",
    )
    parser.add_argument("--gpu", type=int)
    parser.add_argument(
        "--label", type=str, default="valence", choices=["valence", "arousal"]
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prune_amount", type=float, default=0.5)
    args = parser.parse_args()
    utils.seed_everything(2023)
    #
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu)
    #
    model = bci_loader.load_model(model_name=args.model_name)
    model.load_state_dict(
        torch.load(
            f"./bci_model/source/{args.model_name}/model_best.pth", map_location=device
        )
    )
    train_dataset, attack_dataset = bci_loader.load_split_dataset(
        model_name=args.model_name
    )
    #
    results = fine_prune(
        model=model, train_dataset=train_dataset, attack_dataset=attack_dataset
    )

# python fineprune.py --gpu 7
