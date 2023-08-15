import os
import torch
import bci_loader
import torchvision


def load_cv_model(num, mode, device):
    if mode == "source":
        model = torchvision.models.vgg16_bn(weights=None)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(os.path.join("./cv_model", "vgg_model.pth"), device)
        )
    elif mode == "surrogate":
        model = torchvision.models.vgg16_bn(weights=None)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(f"./cv_model/surrogate/surrogate_{num}.pth", device)
        )
    elif mode == "model_extract_l":
        if num < 5:
            model = torchvision.models.vgg13(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        elif 5 <= num < 10:
            model = torchvision.models.resnet18(weights=None)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif 10 <= num < 15:
            model = torchvision.models.densenet121(weights=None)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)
        elif 15 <= num < 20:
            model = torchvision.models.mobilenet_v2(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(
                os.path.join("cv_model", "student_model_1_" + str(num) + ".pth"), device
            )
        )
    elif mode == "model_extract_p":
        if num < 5:
            model = torchvision.models.vgg13(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        elif 5 <= num < 10:
            model = torchvision.models.resnet18(weights=None)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif 10 <= num < 15:
            model = torchvision.models.densenet121(weights=None)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)
        elif 15 <= num < 20:
            model = torchvision.models.mobilenet_v2(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(
                os.path.join("cv_model", "student_model_kd_" + str(num) + ".pth"),
                device,
            )
        )
    elif mode == "irrelevant":
        if 10 > num >= 5:
            model = torchvision.models.resnet18(weights=None)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num < 5:
            model = torchvision.models.vgg13(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        elif 15 > num >= 10:
            model = torchvision.models.densenet121(weights=None)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)
        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(
                os.path.join("cv_model", "clean_model_" + str(num) + ".pth"), device
            )
        )
    elif mode == "transfer_learning":
        if num >= 5:
            model = torchvision.models.resnet18(weights=None)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num < 5:
            model = torchvision.models.vgg16_bn(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "cv_model", "finetune_model", "CIFAR10C_" + str(num) + ".pth"
                ),
                device,
            )
        )
    elif mode == "model_extract_adv":
        if 5 <= num < 10:
            model = torchvision.models.resnet18(weights=None)
            in_feature = model.fc.in_features
            model.fc = torch.nn.Linear(in_feature, 10)
        elif num < 5:
            model = torchvision.models.vgg13(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        elif 15 > num >= 10:
            model = torchvision.models.densenet121(weights=None)
            in_feature = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feature, 10)
        elif 20 > num >= 15:
            model = torchvision.models.mobilenet_v2(weights=None)
            in_feature = model.classifier[-1].in_features
            model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(
                os.path.join("cv_model", "adv_train", "adv_" + str(num) + ".pth"),
                device,
            )
        )
    elif mode == "fineprune":
        model = torch.load(
            "cv_model/Fine-Pruning/prune_model_" + str(num) + ".pth", device
        )
    elif mode == "finetune":
        model = torchvision.models.vgg16_bn(weights=None)
        in_feature = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_feature, 10)
        model.load_state_dict(
            torch.load(
                os.path.join("cv_model", "finetune_10", "finetune" + str(num) + ".pth"),
                device,
            )
        )
    return model


def load_bci_model(num: int, mode: str, device: torch.device) -> torch.nn.Module:
    if mode == "source":
        model = bci_loader.load_model(model_name="conformer")
        model.load_state_dict(
            torch.load(
                f"./bci_model/{mode}/conformer/model_best.pth", map_location=device
            )
        )
    elif mode == "finetune":
        model = bci_loader.load_model(model_name="conformer")
        model.load_state_dict(
            torch.load(
                f"./bci_model/{mode}/conformer/{mode}_{num}/model_best.pth",
                map_location=device,
            )
        )
    elif mode == "irrelevant":
        if num < 5:
            model = bci_loader.load_model(model_name="deepnet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/irrelevant/deepnet/irrelevant_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
        elif 10 > num >= 5:
            model = bci_loader.load_model(model_name="shallownet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/irrelevant/shallownet/irrelevant_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
    elif mode == "model_extract_l":
        if num < 5:
            model = bci_loader.load_model(model_name="deepnet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_deepnet/model_extract_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
        elif 10 > num >= 5:
            model = bci_loader.load_model(model_name="shallownet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_shallownet/model_extract_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
    elif mode == "model_extract_p":
        if num < 5:
            model = bci_loader.load_model(model_name="deepnet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_deepnet/model_extract_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
        elif 10 > num >= 5:
            model = bci_loader.load_model(model_name="shallownet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_shallownet/model_extract_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
    elif mode == "model_extract_adv":
        if num < 5:
            model = bci_loader.load_model(model_name="deepnet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_deepnet/model_extract_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
        elif 10 > num >= 5:
            model = bci_loader.load_model(model_name="shallownet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_shallownet/model_extract_{num%5}/model_best.pth",
                    map_location=device,
                )
            )
    elif mode == "surrogate":
        if num < 2:
            model = bci_loader.load_model(model_name="deepnet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_deepnet/{mode}_{num%2}/model_best.pth",
                    map_location=device,
                )
            )
        elif 4 > num >= 2:
            model = bci_loader.load_model(model_name="shallownet")
            model.load_state_dict(
                torch.load(
                    f"./bci_model/{mode}/conformer_shallownet/{mode}_{num%2}/model_best.pth",
                    map_location=device,
                )
            )
    elif mode == "transfer_learning":
        model = bci_loader.load_model(model_name="conformer")
        model.load_state_dict(
            torch.load(
                f"./bci_model/{mode}/conformer/{mode}_{num}/model_best.pth",
                map_location=device,
            )
        )
    elif mode == "fineprune":
        model = torch.load(
            f"./bci_model/{mode}/conformer/prune_{num}.pth",
            map_location=device,
        )
    return model
