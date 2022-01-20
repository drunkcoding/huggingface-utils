from typing import Any, Union
from torch.functional import Tensor
import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


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


def agg_logits(hist, curr, pos, device):
    alpha = 0.6
    if hist is not None:
        hist = hist.to(device)
        # return (hist * pos + curr) / (pos+1)
        return hist * (1 - alpha) + curr * alpha
    return curr


def temperature_scale(
    logits: torch.Tensor,
    temperature: Any,
) -> torch.Tensor:
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits

    if not isinstance(temperature, torch.Tensor):
        temperature = torch.nn.Parameter(
            torch.ones(1, device=logits.device) * temperature
        )

    temperature = (
        temperature.unsqueeze(1)
        .expand(logits.size(0), logits.size(1))
        .to(logits.device)
    )
    return logits / temperature


def temperature_scaling(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert outputs.device == labels.device
    device = outputs.device
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
