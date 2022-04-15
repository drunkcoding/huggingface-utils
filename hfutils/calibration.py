from sched import scheduler
from turtle import forward
from typing import Any, Dict, Union
from torch.functional import Tensor
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


m = torch.nn.Softmax(dim=-1)

# def agg_logits(hist, curr, pos, device):
#     if hist is not None:
#         hist = hist.to(device)
#         curr_prob, _ = torch.max(torch.float_power(m(curr), 2), dim=-1)
#         hist_prob, _ = torch.max(torch.float_power(m(hist), 2), dim=-1)

#         diff = torch.abs(hist_prob-curr_prob)
#         # print(diff)
#         for i in range(len(diff)):
#             if diff[i] > 0.2:
#                 if curr_prob[i] < hist_prob[i]:
#                     curr[i] = hist[i]
#             else:
#                 curr[i] = (hist[i] * pos + curr[i]) / (pos+1)
#     return curr


# def agg_logits(hist, curr, pos, device):
#     alpha = 0.6
#     if hist is not None:
#         hist = hist.to(device)
#         # return (hist * pos + curr) / (pos+1)
#         return hist * (1 - alpha) + curr * alpha
#     return curr


def agg_logits(hist, curr, alpha):
    if hist is not None:
        return hist * (1 - alpha) + curr * alpha
    return curr


class CalibrationLayer(torch.nn.Module):
    def __init__(self, out_size, hidden_size=100) -> None:
        super().__init__()

        self.g_layer = torch.nn.Sequential(
            torch.nn.Linear(out_size, hidden_size, bias=False),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, out_size, bias=False),
            torch.nn.Dropout(0.1),
        )

        for p in self.g_layer.parameters():
            if isinstance(p, nn.Linear):
                torch.nn.init.eye_(p.weight)

    def forward(self, logits):
        return self.g_layer(logits)


# class ModelWithCalibration(torch.nn.Module):
#     def __init__(self, model, out_size) -> None:
#         super().__init__()

#         self.model = model
#         self.calibration_layer = CalibrationLayer(out_size)

#     def forward(self, batch):
#         outputs =

from torch import optim

# def g_scaling(model_func, eval_dataloader, out_size) -> CalibrationLayer:
#     calibration_layer = CalibrationLayer(out_size)
#     nll_criterion = torch.nn.CrossEntropyLoss()
#     optimizer = optim.Adam(calibration_layer.parameters, lr=0.0001)
#     for epoch in range(200):
#         for batch in eval_dataloader:
#             logits, labels = model_func(batch)
#             logits = calibration_layer(logits)
#             loss = nll_criterion(logits, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     return calibration_layer


def g_scaling_helper(
    outputs: Dict, labels: torch.Tensor, epoches: Dict, out_size: int
) -> Dict:
    model_temperature = {}
    for key in outputs:
        labels = labels.to(outputs[key].device)
        model_temperature[key] = g_scaling(outputs[key], labels, epoches[key], out_size)
    return model_temperature


import optuna
from functools import partial


def objective(trial, outputs: torch.Tensor, labels: torch.Tensor, out_size: int):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    eps = trial.suggest_loguniform("eps", 1e-9, 1e-4)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-3, 1e-1)

    calibration_layer = CalibrationLayer(out_size)
    calibration_layer = calibration_layer.to(outputs.device)
    nll_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        calibration_layer.parameters(), lr=lr, eps=eps, weight_decay=weight_decay
    )
    assert labels.dtype == torch.int64
    for _ in range(100):
        logits = calibration_layer(outputs)
        loss = nll_criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def g_scaling(
    outputs: torch.Tensor, labels: torch.Tensor, epoch: int, out_size: int
) -> CalibrationLayer:

    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(objective, outputs=outputs, labels=labels, out_size=out_size),
        n_trials=100,
    )
    trial = study.best_trial.params

    calibration_layer = CalibrationLayer(out_size)
    calibration_layer = calibration_layer.to(outputs.device)
    nll_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        calibration_layer.parameters(),
        lr=trial["lr"],
        eps=trial["eps"],
        weight_decay=trial["weight_decay"],
    )
    # scheduler = CosineAnnealingLR(optimizer, epoch, eta_min=3e-5)
    assert labels.dtype == torch.int64
    for _ in tqdm(range(epoch)):
        logits = calibration_layer(outputs)
        loss = nll_criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    print("g_scaling loss", loss)
    return calibration_layer


def temperature_scale(logits: torch.Tensor, temperature: Any,) -> torch.Tensor:
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits

    if not isinstance(temperature, torch.Tensor):
        temperature = torch.nn.Parameter(
            torch.ones(1, device=logits.device) * temperature
        )

    temperature = temperature.unsqueeze(1).expand(logits.shape).to(logits.device)
    return logits / temperature


def temperature_scaling_helper(
    outputs: Dict, labels: torch.Tensor, devices: Dict
) -> Dict:
    model_temperature = {}
    for key in outputs:
        labels = labels.to(devices[key])
        temperature = (
            temperature_scaling(outputs[key], labels).detach().cpu().numpy().tolist()[0]
        )
        # bar = 1.5
        # temperature = bar + (temperature - bar) / 2 if temperature > bar else temperature
        model_temperature[key] = torch.nn.Parameter(
            torch.ones(1, device=devices[key]) * temperature
        )
    return model_temperature


def temperature_scaling(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # assert outputs.device == labels.device
    device = outputs.device
    labels = labels.to(device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=500)
    # outputs = outputs.to(device)
    # labels = labels.to(device)

    nll_criterion = torch.nn.CrossEntropyLoss().to(device)
    ece_criterion = ECELoss().to(device)

    before_temperature_nll = nll_criterion(outputs, labels).item()
    before_temperature_ece = ece_criterion(outputs, labels).item()
    print(
        "Before temperature - NLL: %.3f, ECE: %.3f"
        % (before_temperature_nll, before_temperature_ece)
    )

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(outputs, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    after_temperature_nll = nll_criterion(
        temperature_scale(outputs, temperature), labels
    ).item()
    after_temperature_ece = ece_criterion(
        temperature_scale(outputs, temperature), labels
    ).item()
    print("Optimal temperature: %.3f" % temperature.item())
    print(
        "After temperature - NLL: %.3f, ECE: %.3f"
        % (after_temperature_nll, after_temperature_ece)
    )

    return temperature.cpu().detach().cpu()
