COMPONENT = ["cc", "cw", "uc", "uw"]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

CV_MODEL_TO_NUM = {
    "source": 1,
    "model_extract_l": 20,
    "model_extract_p": 20,
    "model_extract_adv": 20,
    "transfer_learning": 10,
    "fineprune": 10,
    "finetune": 20,
    "model_extract_adv": 20,
    "irrelevant": 20,
}

BCI_MODEL_TO_NUM = {
    "source": 1,
    "model_extract_l": 10,
    "model_extract_p": 10,
    "model_extract_adv": 10,
    "transfer_learning": 10,
    "fineprune": 10,
    "finetune": 10,
    "model_extract_adv": 10,
    "irrelevant": 10,
}
