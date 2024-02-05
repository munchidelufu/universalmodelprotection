import os
import csv
import sys

sys.path.append("/data/xuth/deep_ipr")
import argparse
from functools import partial

import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import Module
from typing import Union, List
from torch.utils.data._utils import collate
from torch.utils.data import Dataset, DataLoader, Subset

from easydeepip.util import utils
from easydeepip.util.model_loader import CVModelLoader
from easydeepip.util.model_loader import BCIModelLoader
from easydeepip.util.model_loader import NLPModelLoader
from easydeepip.util.data_adapter import SplitDataConverter
from easydeepip.model_ip.model_ip import ModelIP
from easydeepip.quary_attack import QueryAttack

COMPONENT = ["cc", "cw", "uc", "uw"]

CV_MODEL_TO_NUM = {
    "source": 1,
    "model_extract_l": 15,
    "model_extract_adv": 15,
    "model_extract_p": 15,
    "transfer_learning": 10,
    "fine_prune": 10,
    "fine_tune": 20,
    "irrelevant": 15,
}


class FeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()


class MetaFingerprint:
    def __init__(
        self, field: str, model: Module, dataset: Dataset, device: torch.device
    ) -> None:
        """MetaSamples generated from the model's components for depicting its 'fingerprint'.
        We think the models components consist of model's trained parameters and the trainset.

        Args:
            field (str): 'cv' or 'bci'.
            model (Module): model to be depicting fingerprint.
            dataset (Dataset): model's trainset.
            device (torch.device): default 'cuda'.
        """
        self.field = field
        self.model = model
        self.dataset = dataset
        self.device = device

    @utils.timer
    def generate_meta_fingerprint_point(self, n: int):
        """Generating four meta-fingerprint samples for protected models
        Args:
            n (int): number of samples of the four types, where equal numbers are taken.
        """
        dataloader = DataLoader(dataset=self.dataset, shuffle=False, batch_size=64)
        correct_info, wrong_info = self.test_pro(
            model=self.model, dataloader=dataloader
        )
        correct_partial = partial(self.confidence_well, info=correct_info, n=n)
        for m in ["cc", "uc"]:
            correct_partial(mode=m)
        wrong_partial = partial(self.confidence_well, info=wrong_info, n=n)
        for m in ["cw", "uw"]:
            wrong_partial(mode=m)

    def confidence_well(self, info: list, mode: str, n: int):
        # Select n samples according to the confidence level of the model for this type of sample
        if mode in ["cc", "uw"]:
            reverse = False
        elif mode in ["cw", "uc"]:
            reverse = True
        k_loss_indexs = sorted(info, key=lambda x: x[0], reverse=reverse)[:n]
        _, indexs = zip(*k_loss_indexs)
        sub_dataset = Subset(self.dataset, indexs)
        data, label = [], []
        for item in sub_dataset:
            data.append(item[0])
            label.append(item[1])
        data = torch.stack(data, dim=0)
        label = torch.tensor(label)

        ModelIP.save_ip(
            to_file=f"./fingerprint/{self.field}/meta_{n}/original_{mode}.pkl",
            data=data,
            label=label,
        )

    def test_pro(self, model: Module, dataloader: DataLoader):
        """
        Collect the correct and misclassified sample information of the converged model in the training set

        Args:
            model (Module): model to be depicting fingerprint.
            dataloader (DataLoader): training set loader

        Returns:
            list: [info,...], info=(sample_loss, sample_index)
        """
        model.eval()
        model = model.to(self.device)
        correct_num = 0
        correct, wrong = [], []
        for _, batch_index in enumerate(dataloader._index_sampler):
            batch_data = collate.default_collate(
                [dataloader.dataset[idx] for idx in batch_index]
            )
            b_x = batch_data[0].to(self.device)
            b_y = batch_data[1].to(self.device)
            output = model(b_x)
            loss = F.cross_entropy(output, b_y, reduction="none")
            pred = torch.argmax(output, dim=-1)
            correct.extend(
                [
                    (loss[i].detach().cpu(), batch_index[i])
                    for i, label in enumerate(pred)
                    if label == b_y[i]
                ]
            )
            wrong.extend(
                [
                    (loss[i].detach().cpu(), batch_index[i])
                    for i, label in enumerate(pred)
                    if label != b_y[i]
                ]
            )
            correct_num += (pred == b_y).sum().item()
        model.cpu()
        assert correct_num == len(correct)
        return correct, wrong


class PerturbedFingerprint:
    def __init__(
        self,
        field: str,
        iters: Union[int, List[int]],
        lr: Union[float, List[float]],
        delta: float = 1e-5,
    ) -> None:
        """
        Initialize the hyper-parameters of the algorithm, support a set of hyperparameters.

        Args:
            field (str): 'cv' or 'bci'
            iters (Union[int, List[int]]): number (or numbers) of perturbed sample iterations generated by the gfsa algorithm
            lr (Union[float, List[float]]): learning rate of the gfsa algorithm
            delta (float): default
        """
        self.field = field
        self.iters = iters
        self.lr = lr
        self.delta = delta
        self.finger_components = COMPONENT
        if field == "cv":
            self.model_to_num = CV_MODEL_TO_NUM
        else:
            raise NotImplementedError

    def pfa(
        self,
        model: torch.nn.Module,
        input: torch.Tensor,
        finger_component: str = "cc",
    ):
        """
        Args:
            model (torch.nn.Module): The source model to be protected.
            input (torch.Tensor): meta sample for generating perturbed fingerprint samples.
            fingerprint_component (str, optional): Four components of Fingerprint data, which in ['cc','cw','uc', and 'uw'].
                Defaults to "cc".

        Returns:
            Tensor : The final perturbed fingerprint sample.
        """
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        input = torch.unsqueeze(input, dim=0)
        input_clone = input.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([input_clone], lr=self.lr)
        p = F.softmax(model(input_clone), dim=1)
        i = torch.argmax(p)
        cur_p = p[0][i]
        if finger_component.startswith("c"):
            j = torch.argmin(p)
            for _ in range(self.iters):
                p = F.softmax(model(input_clone), dim=1)
                loss = -1 * (p[0][i] - p[0][j]) / (1 - p[0][i] + self.delta)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        elif finger_component.startswith("u"):
            dis = p[0][i] - 1 / len(p[0])
            top = 2 * dis + cur_p
            j = torch.topk(p, k=2, dim=1)[1][:, 1]
            for _ in range(self.iters):
                p = F.softmax(model(input_clone), dim=1)
                if p[0][i] >= top:
                    break
                clamped_p = torch.clamp(p[0][i], min=cur_p, max=top)
                loss = -1 * torch.log(clamped_p)
                optimizer.zero_grad()
                grads = torch.autograd.grad(loss, input_clone, retain_graph=True)
                input_clone.data = input_clone.data - self.lr * grads[0]
        return input_clone.detach()

    @utils.timer
    def pfa_helper(self, model: torch.nn.Module, n: int):
        """
        Args:
            model (torch.nn.Module): The source model to be protected.
            fingerprint_component (str, optional): Four components of Fingerprint data, which in ['cc','cw','uc', and 'uw'].
                Defaults to "cc".
            verbose (bool, optional): Whether to print new labels for generated samples.
        """
        for fc in self.finger_components:
            meta_data_path = f"./fingerprint/{self.field}/meta_{n}/original_{fc}.pkl"
            meta_data = ModelIP.load_ip(meta_data_path)["data"]
            sample_record = []
            for i in range(len(meta_data)):
                pert_s = self.pfa(model, meta_data[i], fc)
                sample_record.append(pert_s)
            pert_datas = torch.cat(sample_record, dim=0)
            pert_labels = torch.argmax(model(pert_datas), dim=1)
            ModelIP.save_ip(
                to_file=f"./fingerprint/{self.field}/pert_{n}_{self.lr}_{self.iters}/original_{fc}.pkl",
                data=pert_datas,
                label=pert_labels,
            )
        return None


class FingerprintMatch:
    def __init__(
        self,
        field: str,
        meta: bool,
        device: torch.device,
        ip_erase: str,
        n: int,
        lr: float,
        iters: int,
    ) -> None:
        self.field = field
        self.finger_component = COMPONENT
        self.meta = meta
        self.n = n
        self.lr = lr
        self.iters = iters
        if field == "cv":
            self.model_num = CV_MODEL_TO_NUM
        else:
            raise NotImplementedError

        self.device = device
        save_dir = (
            f"./result/{self.field}/meta_{self.n}/"
            if meta
            else f"./result/{self.field}/pert_{self.n}_{self.lr}_{self.iters}/"
        )
        os.makedirs(save_dir, exist_ok=True)
        self.feature_path = os.path.join(save_dir, f"{ip_erase}_feature.csv")
        self.ip_erase = ip_erase

    def dump_feature(self):
        ml = (
            CVModelLoader(dataset_name="cifar100", device=self.device)
            if self.field == "cv"
            else BCIModelLoader(dataset_name="cifar100", device=self.device)
        )
        with open(self.feature_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            for model_type, num in self.model_num.items():
                for i in range(num):
                    if model_type in ["irrelevant"]:
                        i += 5

                    feature_record = []
                    # print(model_type, num)
                    model = ml.load_model(mode=model_type, index=i)
                    model.to(self.device)
                    model.eval()
                    for fc in self.finger_component:
                        fc_path = (
                            f"./fingerprint/{self.field}/meta_{self.n}/{self.ip_erase}_{fc}.pkl"
                            if self.meta
                            else f"./fingerprint/{self.field}/pert_{self.n}_{self.lr}_{self.iters}/{self.ip_erase}_{fc}.pkl"
                        )
                        finger = ModelIP.load_ip(fc_path)
                        data = finger["data"].to(self.device)
                        label = finger["label"].to(self.device)
                        pred = torch.argmax(model(data.to(self.device)), dim=1)
                        correct = (label == pred).sum().item()
                        feature_record.append(round(correct / len(pred), 2))
                    feature_record.append(model_type)
                    writer.writerow(feature_record)
        print(f"{ self.ip_erase} model feature dump to {self.feature_path}")

    def fingerprint_recognition(
        self, n_features: list = [0, 1, 2, 3], verbose: bool = False
    ):
        """
        Args:
            n_features (list): Default full finger. How many fingerprint components be choosed.
            verbose (bool, optional): Whether to print the auc between models. Defaults to False.
        """
        with open(self.feature_path, mode="r") as file:
            reader = csv.reader(file)
            features = [
                [float(row[i]) for i in n_features] + [row[4]] for row in reader
            ]

        source_feature = np.array([row[:-1] for row in features if row[-1] == "source"])
        irr_feature = np.array(
            [row[:-1] for row in features if row[-1] == "irrelevant"]
        )
        pro_feature = np.array(
            [row[:-1] for row in features if row[-1] == "model_extract_p"]
        )
        lab_feature = np.array(
            [row[:-1] for row in features if row[-1] == "model_extract_l"]
        )
        tl_feature = np.array(
            [row[:-1] for row in features if row[-1] == "transfer_learning"]
        )
        fp_feature = np.array([row[:-1] for row in features if row[-1] == "fine_prune"])
        ft_feature = np.array([row[:-1] for row in features if row[-1] == "fine_tune"])
        adv_feature = np.array(
            [row[:-1] for row in features if row[-1] == "model_extract_adv"]
        )

        def helper(input):
            input = np.array(input)
            simi_score = np.linalg.norm(input - source_feature[0], ord=2)
            return simi_score

        try:
            irr_simi = list(map(helper, irr_feature))
            pro_simi = list(map(helper, pro_feature))
            lab_simi = list(map(helper, lab_feature))
            tl_simi = list(map(helper, tl_feature))
            fp_simi = list(map(helper, fp_feature))
            ft_simi = list(map(helper, ft_feature))
            adv_simi = list(map(helper, adv_feature))

            pro_auc = utils.calculate_auc(list_a=pro_simi, list_b=irr_simi)
            lab_auc = utils.calculate_auc(list_a=lab_simi, list_b=irr_simi)
            tl_auc = utils.calculate_auc(list_a=tl_simi, list_b=irr_simi)
            fp_auc = utils.calculate_auc(list_a=fp_simi, list_b=irr_simi)
            ft_auc = utils.calculate_auc(list_a=ft_simi, list_b=irr_simi)
            adv_auc = utils.calculate_auc(list_a=adv_simi, list_b=irr_simi)
            if verbose:
                print(
                    "ft:",
                    ft_auc,
                    "fp:",
                    fp_auc,
                    "lab:",
                    lab_auc,
                    "pro:",
                    pro_auc,
                    "adv:",
                    adv_auc,
                    "tl:",
                    tl_auc,
                )
            auc_records = [ft_auc, fp_auc, lab_auc, pro_auc, adv_auc, tl_auc]
            return sum(auc_records) / len(auc_records)
        except ValueError:
            print(self.n, self.lr, self.iters)


class GridSearch:
    """AI is creating summary for"""

    def __init__(
        self, n: list, iters: list, lr: list, source_model: Module, device: torch.device
    ):
        self.n = n
        self.iters = iters
        self.lr = lr
        self.field = "cv"
        self.type = "meta"
        self.ip_erase = "original"
        self.device = device

        self.source_model = source_model

    def search(self):
        res = {}
        for i in tqdm(self.n, desc="n Search"):
            for lr in tqdm(self.lr, desc="lr Search", leave=False):
                for it in tqdm(self.iters, desc="iter Search", leave=False):
                    if (
                        os.path.exists(f"./fingerprint/{self.field}/pert_{i}_{lr}_{it}")
                        and len(
                            os.listdir(f"./fingerprint/{self.field}/pert_{i}_{lr}_{it}")
                        )
                        == 4
                    ):
                        pass
                    else:
                        pf = PerturbedFingerprint(field=self.field, iters=it, lr=lr)
                        pf.pfa_helper(model=self.source_model, n=i)
                    pert_fm = FingerprintMatch(
                        field=self.field,
                        meta=False,
                        device=self.device,
                        ip_erase=self.ip_erase,
                        n=i,
                        lr=lr,
                        iters=it,
                    )
                    pert_file = f"./result/{self.field}/pert_{i}_{lr}_{it}/{self.ip_erase}_feature.csv"
                    if os.path.exists(pert_file):
                        os.remove(pert_file)
                    pert_fm.dump_feature()
                    meta_fm = FingerprintMatch(
                        field=self.field,
                        meta=True,
                        device=self.device,
                        ip_erase=self.ip_erase,
                        n=i,
                        lr=lr,
                        iters=it,
                    )
                    meta_file = (
                        f"./result/{self.field}/meta_{i}/{self.ip_erase}_feature.csv"
                    )
                    if os.path.exists(meta_file):
                        os.remove(meta_file)
                    meta_fm.dump_feature()

                    try:
                        meta_avg = meta_fm.fingerprint_recognition(verbose=False)
                        pert_avg = pert_fm.fingerprint_recognition(verbose=False)
                        res[f"n_{i}_lr_{lr}_iter_{it}"] = (meta_avg + pert_avg) / 2
                    except FileNotFoundError:
                        print(i, lr, it)
        result = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        print(result)
        # with open("./grid_search.txt", "w") as file:
        #     file.write(str(result))
        print("meta_pert and search done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="cifar100")
    args = parser.parse_args()

    utils.seed_everything(2023)
    device = torch.device("cuda", args.gpu)
    # field = "cv"
    cv_model_loader = CVModelLoader(dataset_name="cifar100", device=device)
    source_model = cv_model_loader.load_model(mode="source")
    train_dataset, dev_dataset, test_dataset = SplitDataConverter.split(args.dataset)

    # mf = MetaFingerprint(
    #     field="cv", model=source_model, dataset=train_dataset, device=device
    # )
    # mf.generate_meta_fingerprint_point(20) # gen time 16.97s
    # for n in range(10, 101, 10):
    #     mf.generate_meta_fingerprint_point(n=n)  # 15.99s

    # pf = PerturbedFingerprint(
    #     field="cv",
    #     iters=20,
    #     lr=0.01,
    # )
    # pf.pfa_helper(source_model, 20)  # gen time 48.78s

    # device = torch.device("cuda", 1)
    # gd = GridSearch(
    #     n=list(range(10, 101, 10)),
    #     iters=list(range(10, 51, 10)),
    #     lr=[0.1, 0.01, 0.001],
    #     source_model=source_model,
    #     device=device,
    # )
    # gd.search()

    # ft: 1.0 fp: 0.89 lab: 1.0 pro: 1.0 adv: 1.0 tl: 0.62 original
    # ft: 1.0 fp: 0.81 lab: 1.0 pro: 1.0 adv: 1.0 tl: 0.42 erasure
    # @utils.timer
    # def time_count():
    #     fm = FingerprintMatch(
    #         "cv",
    #         meta=False,
    #         device=device,
    #         ip_erase="original",
    #         n=20,
    #         lr=0.01,
    #         iters=20,
    #     )
    #     fm.dump_feature()
    #     fm.fingerprint_recognition(verbose=True)

    # time_count()  # infer time 125.09s

    # ft: 1.0 fp: 0.81 lab: 1.0 pro: 1.0 adv: 1.0 tl: 0.5 original #125.02
    # ft: 1.0 fp: 0.81 lab: 1.0 pro: 1.0 adv: 1.0 tl: 0.42 erasure
    # mf = MetaFingerprint(
    #     field="cv", model=source_model, dataset=train_dataset, device=device
    # )
    # mf.generate_meta_fingerprint_point(n=10)  # 15.99s
    # pf = PerturbedFingerprint(
    #     field="cv",
    #     iters=50,
    #     lr=0.01,
    # )
    # pf.pfa_helper(source_model, 10)  # 62.87s

    # ft: 1.0 fp: 0.87 lab: 1.0 pro: 1.0 adv: 1.0 tl: 0.5 original
    # ft: 1.0 fp: 0.82 lab: 1.0 pro: 1.0 adv: 0.97 tl: 0.47 erasure
    # @utils.timer
    # def time_count():
    #     fm = FingerprintMatch(
    #         "cv", meta=True, device=device, ip_erase="original", n=20, lr=0.01, iters=20
    #     )
    #     fm.dump_feature()
    #     fm.fingerprint_recognition(verbose=True)

    # time_count() # infer time 126.27s

    # @utils.timer
    # def helper():
    #     fm.dump_feature()
    #     fm.fingerprint_recognition(verbose=True)

    # helper()
