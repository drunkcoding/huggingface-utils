

from typing import Tuple
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, default_data_collator

from hfutils.arg_parser import HfArguments
from hfutils.constants import TASK_TO_KEYS, TASK_TO_LABELS

# class DatasetProcessor:
#     def __init__(self, task_name, raw_datasets):
#         # Labels
#         if task_name is not None:
#             is_regression = task_name == "stsb"
#             if not is_regression:
#                 self.label_list = raw_datasets["train"].features["label"].names
#                 self.num_labels = len(label_list)
#             else:
#                 self.num_labels = 1
#         else:
#             # Trying to have good defaults here, don't hesitate to tweak to your needs.
#             is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
#             if is_regression:
#                 self.num_labels = 1
#             else:
#                 # A useful fast method:
#                 # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
#                 self.label_list = raw_datasets["train"].unique("label")
#                 self.label_list.sort()  # Let's sort it for determinism
#                 self.num_labels = len(label_list)

#         # Some models have set the order of the labels to use, so let's make sure we do use it.
#         self.label_to_id = {v: i for i, v in enumerate(self.label_list)}
#         # label_to_id = None
#         # if (
#         #     model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
#         #     and data_args.task_name is not None
#         #     and not is_regression
#         # ):
#         #     # Some have all caps in their config, some don't.
#         #     label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
#         #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
#         #         label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
#         #     else:
#         #         logger.warning(
#         #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
#         #             f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
#         #             "\nIgnoring the model labels as a result.",
#         #         )
#         # elif data_args.task_name is None and not is_regression:
#         #     label_to_id = {v: i for i, v in enumerate(label_list)}


# def default_preprocess_function(examples, task_name):
#     sentence1_key = TASK_TO_KEYS[task_name][0]
#     sentence2_key = None if len(TASK_TO_KEYS[task_name]) == 1 else TASK_TO_KEYS[task_name][1]
    
#     args = (
#         (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
#     )
#     result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

#     # Map labels to IDs (not necessary for GLUE tasks)
#     if self.label_to_id is not None and "label" in examples:
#         result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
#     return result

# def t5_preprocess_function(examples, task_name):
#     sentence1_key = TASK_TO_KEYS[task_name][0]
#     sentence2_key = None if len(TASK_TO_KEYS[task_name]) == 1 else TASK_TO_KEYS[task_name][1]
#     sentence1_examples = examples[sentence1_key]
#     sentence2_examples = None if sentence2_key is None else examples[sentence2_key]
#     processed_examples = []
#     for i in range(len(sentence1_examples)):
#         elements = [
#                 task_name, 
#                 sentence1_key+":",
#                 sentence1_examples[i],
#             ]
#         if sentence2_examples is not None:
#             elements += [
#                 sentence2_key+":",
#                 sentence2_examples[i],
#             ]
#         processed_examples.append(" ".join(elements))

#     texts = (
#         (processed_examples,)
#     )
#     result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True, return_tensors="np")

#     if "label" in examples:

#         labels = examples["label"]
#         labels = [label2text(task_name, label) for label in labels]

#         # Setup the tokenizer for targets
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(labels, max_length=2, padding=padding, truncation=True, return_tensors="np")

#         # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
#         # padding in the loss.
#         if padding == "max_length":
#             labels["input_ids"] = [
#                 [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
#             ]

#         result["labels"] = labels["input_ids"]
#     # del result['label']
#     return result

from transformers import AutoFeatureExtractor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path, split="train"):
        transform = []

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path
        )

        if feature_extractor.do_resize:
            transform.append(
                RandomResizedCrop(feature_extractor.size) if split == "train" else Resize(feature_extractor.size)
            )

        transform.append(RandomHorizontalFlip() if split == "train" else CenterCrop(feature_extractor.size))
        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x.convert("RGB"))

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        # self.inp = self.inp.pin_memory()
        # self.tgt = self.tgt.pin_memory()
        return {"pixel_values": self.inp, 'labels': self.tgt}

def split_train_test(dataset, test_size):
    index = np.array([x for x in range(len(dataset))])
    train_index, test_index = train_test_split(index, test_size=test_size, random_state=0, shuffle=True)

    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)

    # train_sampler = SequentialSampler(train_dataset)
    # test_sampler = SequentialSampler(test_dataset)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_dataset, test_dataset


def vit_collate_fn(batch):
    # print("==batchsize====", len(batch))
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = torch.tensor(transposed_data[1])
    return {"pixel_values": inp, 'labels': tgt}

class ViTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_extractor, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.device = device

    def __getitem__(self, idx):
        encodings = self.feature_extractor(self.dataset[idx][0], return_tensors='pt')   
        encodings['pixel_values'] = encodings['pixel_values'].squeeze(0).to(self.device)
        return (encodings, torch.tensor(self.dataset[idx][1], dtype=torch.long).to(self.device))

    def __len__(self):
        return len(self.dataset)

class TorchVisionDataset(torch.utils.data.Dataset):
    def __init__(self, input, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.input = input
        self.device = device

    def __getitem__(self, idx):
        item = {'x': self.input[idx][0].clone().detach().to(self.device)}
        return (item, self.input[idx][1] if self.input[idx][1] != None else None)

    def __len__(self):
        return len(self.input)

def format_texts(examples, data_args):
    texts = []
    fields = TASK_TO_KEYS[data_args.task_name]
    for key in fields:
        if key is not None:
            texts.append(examples[key])
    return tuple(texts)

def do_padding(data_args):
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    return padding

def label2seq(data_args, label):
    # easy_labels = ("true", "false")
    # return easy_labels[label]
    if TASK_TO_LABELS[data_args.task_name] is None:
        return label
    else:
        return TASK_TO_LABELS[data_args.task_name][label]
        # return easy_labels[label]

def default_preprocess(
    examples,
    data_args,
    tokenizer=None,
    label_to_id=None,
):
    texts = format_texts(examples, data_args)
    fields = TASK_TO_KEYS[data_args.task_name]

    processed_examples = []
    for i in range(len(texts[0])):
        elements = [data_args.task_name]
        for j in range(len(texts)):
            if fields[j] is None: continue
            elements.append(fields[j]+":")
            elements.append(texts[j][i])
        processed_examples.append(" ".join(elements))

    if tokenizer is None:
        return {
            "input_text": processed_examples
        }

    result = tokenizer(*texts, padding=do_padding(data_args), max_length=data_args.max_length, truncation=True)

    if "label" in examples:
        if label_to_id is not None:
            # Map labels to IDs (not necessary for GLUE tasks)
            result["labels"] = [label_to_id[l] for l in examples["label"]]
        else:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
    return result

def seq2seq_preprocess(
    examples,
    data_args,
    tokenizer=None,
    label_to_id=None,
):
    texts = format_texts(examples, data_args)
    fields = TASK_TO_KEYS[data_args.task_name]

    processed_examples = []
    for i in range(len(texts[0])):
        elements = [data_args.task_name]
        for j in range(len(texts)):
            if fields[j] is None: continue
            elements.append(fields[j]+":")
            elements.append(texts[j][i])
        processed_examples.append(" ".join(elements))

    if tokenizer is None:
        return {
            "input_text": processed_examples
        }

    texts = (
        (processed_examples,)
    )
    padding = do_padding(data_args)
    result = tokenizer(*texts, padding=padding, max_length=data_args.max_length, truncation=True)

    if "label" in examples:

        labels = examples["label"]
        labels = [label2seq(data_args, label) for label in labels]

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labels, max_length=2, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        result["labels"] = labels["input_ids"]

    return result