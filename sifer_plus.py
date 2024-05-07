# Importing libraires
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


# Class Definition
class SiferPlus(nn.Module):
    def __init__(self, models, optim_params, device):
        super(SiferPlus, self).__init__()
        self.device = device
        self.initModels(models)
        self.initOptimParams(optim_params)

    def initModels(self, models):
        model_names = ["PreModel", "AuxModel", "MainModel"]

        for name in model_names:
            if not (name in models):
                raise KeyError(f"Model missing : {name} not found.")

        self.pre_model = models["PreModel"]
        self.pre_model.to(self.device)

        self.aux_model = models["AuxModel"]
        self.aux_model.to(self.device)

        self.main_model = models["MainModel"]
        self.main_model.to(self.device)

        print("Models Initialized")
        print(self.pre_model)
        print(self.aux_model)
        print(self.main_model)

    def initOptimParams(self, optim_params):
        self.mode = optim_params.get("Target", True)
        self.aux_lr = optim_params.get("AuxLR", 0.001)
        self.main_lr = optim_params.get("MainLR", 0.001)
        self.forget_lr = optim_params.get("ForgetGeneralLR", 0.01)

        self.forget_target_weights = optim_params.get(
            "ForgetTargetWeights", 5
        )  # Try to adapt to multiple targets.

        self.general_num_classes = optim_params.get("GenNumClasses", 1)
        self.target_num_classes = optim_params.get("TargetNumClasses", 1)

        self.aux_optim = torch.optim.SGD(
            list(self.pre_model.parameters()) + list(self.aux_model.parameters()),
            lr=self.aux_lr,
        )

        self.main_optim = torch.optim.SGD(
            list(self.pre_model.parameters()) + list(self.main_model.parameters()),
            lr=self.main_lr,
        )

        self.forget_optim = torch.optim.SGD(
            self.pre_model.parameters(), lr=self.forget_lr
        )

    def initTrainParams(self, train_params):
        try:
            num_epochs = train_params.get("NumEpochs", None)
            num_epochs = int(num_epochs)
        except ValueError:
            raise ValueError(
                f"NumEpochs requires Integer Input! Current Val :{num_epochs}"
            )

        forget_interval = train_params.get("ForgetAfter", 0)

        if forget_interval == 0:
            forget_interval = np.Inf
        try:
            self.main_loss = train_params["MainLoss"]
            assert callable(self.main_loss)
        except Exception as e:
            print(e)

        self.target_loss = train_params.get("TargetLoss", None)

        if not (callable(self.target_loss)) and (self.mode):
            raise ValueError("TargetLoss must be callable when target is present.")

        save_dir = train_params.get(
            "SaveDir",
            f"./models/target_{self.forget_target_weights}_forget_{forget_interval}",
        )

        if not (os.path.isdir(save_dir)):
            os.makedirs(save_dir)

        return num_epochs, forget_interval, save_dir

    def train(self, train_loader, train_params):
        num_epochs, forget_interval, save_dir = self.initTrainParams(train_params)
        for epoch in range(1, num_epochs + 1):

            print(f"\nWorking on Epoch {epoch}")
            running_main_loss = 0.0
            running_aux_loss = 0.0
            running_target_loss = 0.0
            num_batches = len(train_loader)

            for i, data in enumerate(train_loader, 1):
                self.aux_optim.zero_grad()
                self.main_optim.zero_grad()

                if self.mode:
                    inputs, labels, targets = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    targets = targets.to(self.device)
                    if len(targets.shape) == 1:
                        targets = targets.reshape(-1, 1)
                else:
                    inputs, labels = data

                pre_out = self.pre_model(inputs)

                if self.mode:
                    aux_out, aux_target = self.aux_model(pre_out)
                else:
                    aux_out = self.aux_model(pre_out)

                if (epoch * num_batches + i) % forget_interval != 0:
                    main_out = self.main_model(pre_out)
                    loss_main = self.main_loss(main_out, labels)

                    running_main_loss += loss_main.item()
                    loss_aux = self.main_loss(aux_out, labels)

                    temp = loss_aux.item()

                    if self.mode:
                        loss_aux = loss_aux + (
                            self.forget_target_weights
                            * self.target_loss(aux_target, targets)
                        )
                        # loss_target.backward()

                    running_aux_loss += temp
                    running_target_loss += loss_aux.item() - temp

                    loss_main.backward(retain_graph=True)
                    loss_aux.backward(retain_graph=True)

                    self.aux_optim.step()
                    self.main_optim.step()

                else:
                    self.forget_optim.zero_grad()

                    labels[:, :] = 1 / labels.shape[1]
                    targets[:, :] = 1 / targets.shape[1]

                    loss_forget = self.main_loss(aux_out, labels)

                    if self.mode:
                        loss_aux = loss_aux + (
                            self.forget_target_weights
                            * self.target_loss(aux_target, targets)
                        )
                        # loss_target.backward()

                    print(f"{loss_forget.item()=}")

                    loss_forget.backward(retain_graph=True)

                    self.forget_optim.step()

                print(
                    f"\rIteration {i}/{num_batches}: {running_main_loss/i=}, {running_aux_loss/i=}, {running_target_loss/i=}",
                    end="",
                )

            os.makedirs(os.path.join(save_dir, f"epoch_{epoch}"))
            torch.save(self.pre_model, os.path.join(save_dir, f"epoch_{epoch}/pre.pth"))
            torch.save(
                self.main_model, os.path.join(save_dir, f"epoch_{epoch}/main.pth")
            )
            torch.save(self.aux_model, os.path.join(save_dir, f"epoch_{epoch}/aux.pth"))


if __name__ == "__main__":
    from networks.models import PreModel, MainModel, AuxModel
    from utils.data import CelebATarget

    B_SIZE = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OPTIM_PARAMS = {
        "Target": True,
        "AuxLR": 1e-3,
        "MainLR": 1e-3,
        "ForgetGeneralLR": 1e-2,
        "ForgetTargetWeights": 5,
    }
    TRAIN_PARAMS = {
        "NumEpochs": 10,
        "ForgetAfter": 0,
        "MainLoss": nn.CrossEntropyLoss(),
        "TargetLoss": nn.BCELoss(),
    }

    data_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.PILToTensor()]
    )

    forget_steps = [100, 500]
    target_weights = [2, 1, 0]

    for target_w in target_weights:
        OPTIM_PARAMS["ForgetTargetWeights"] = target_w
        for forget_step in forget_steps:
            TRAIN_PARAMS["ForgetAfter"] = forget_step

            TRAIN_PARAMS["SaveDir"] = (
                f"./models/target_{OPTIM_PARAMS['ForgetTargetWeights']}_forget_{TRAIN_PARAMS['ForgetAfter']}"
            )

            models = {
                "PreModel": PreModel(),
                "MainModel": MainModel(),
                "AuxModel": AuxModel(),
            }
            train_data = CelebATarget(
                "./data/", transform=data_transform, split="train"
            )

            train_dataloader = DataLoader(train_data, batch_size=B_SIZE, shuffle=True)
            model = SiferPlus(models, OPTIM_PARAMS, DEVICE)

            model.train(train_dataloader, TRAIN_PARAMS)
