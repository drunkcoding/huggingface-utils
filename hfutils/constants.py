from transformers import *
import numpy as np
import torch

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "axb": ("sentence1", "sentence2"),
    "axg": ("premise", "hypothesis"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
}

TASK_TO_LABELS = {
    "cola": ("true", "false"),
    "mnli": ("true", "false", "not"),
    "mrpc": ("true", "false"),
    "qnli": ("true", "false"),
    "qqp": ("true", "false"),
    "rte": ("true", "false"),
    "sst2": ("true", "false"),
    "stsb": None,
    "wnli": ("true", "false"),
    "axb": ("true", "false"),
    "axg": ("true", "false"),
    "boolq": ("yes", "no"),
    "cb": ("true", "false", "not"),
}

MODEL_TASK_TO_GEN = {
    "cola": {
        "t5": "text2text-generation",
        "default": "text-classification",
    },
    "rte": {
        "t5": "text2text-generation",
        "default": "text-classification",
    },
    "sst2": {
        "t5": "text2text-generation",
        "default": "text-classification",
    },
}

MODEL_TASK_TO_CLASS = {
    "cola": {
        "t5": T5ForConditionalGeneration,
        "default": AutoModelForSequenceClassification,
    },
    "rte": {
        "t5": T5ForConditionalGeneration,
        "default": AutoModelForSequenceClassification,
    },
    "sst2": {
        "t5": T5ForConditionalGeneration,
        "default": AutoModelForSequenceClassification,
    },
}

ENSEMBLE_ORDER = ["small", "base", "large", "xl", "PLACEHOLDER"]

NP_TYPE_TO_PT = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int,
    np.int64: torch.long,
    np.bool8: torch.bool,
}

MODEL_KEYS = [
    "S",
    "M",
    "L",
    "XL",
    "2XL",
    "3XL",
    "4XL",
    "5XL",
]

def np_to_torch_dtype(np_dtype):
    if np_dtype == bool:
        return torch.bool
    elif np_dtype == np.bool8:
        return torch.bool
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.float16:
        return torch.half
    elif np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return None
    return None

