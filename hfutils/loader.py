from posixpath import split
import numpy as np
import time
from torch.utils.data.dataloader import DataLoader
from functools import partial
from transformers import AutoConfig, AutoTokenizer, AutoModel
from datasets import concatenate_datasets, load_dataset, load_metric
from transformers.data.data_collator import DataCollatorForSeq2Seq, DataCollatorWithPadding, default_data_collator

from hfutils.arg_parser import HfArguments
from hfutils.constants import TASK_TO_KEYS, TASK_TO_LABELS
from hfutils.preprocess import default_preprocess, seq2seq_preprocess

class ModelLoader:
    def __init__(self, args: HfArguments) -> None:
        self.model_args = args.model_args
        self.data_args = args.data_args

        self.model_cls = AutoModel

    def set_custom_model_class(self, model_cls):
        self.model_cls = model_cls

    def load(self, deepspeed=True, load_tokenizer=True, load_model=True):
        model_args = self.model_args
        data_args = self.data_args
        # labels = TASK_TO_LABELS[data_args.task_name]
        # config = AutoConfig.from_pretrained(
        #     model_args.model_name_or_path,
        #     num_labels=len(labels) if labels is not None else None,
        #     finetuning_task=data_args.task_name,
        # )

        tokenizer = None
        model = None

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                from_slow=True,
            )
        if load_model:
            model = self.model_cls.from_pretrained(
                model_args.model_name_or_path,
                # config=config,
            )

            if deepspeed:
                from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
                model = load_state_dict_from_zero_checkpoint(model, model_args.model_name_or_path)
        
        return tokenizer, model

class DatasetLoader:
    def __init__(self, args: HfArguments) -> None:
        self.data_args = args.data_args
        self.model_args = args.model_args

        #HARD CODE seq2seq model
        self.is_seq2seq = "t5" in self.model_args.model_name_or_path
    
    def load(self, tokenizer=None, partition:str="train", create_dataloader:bool=False):
        data_args = self.data_args
        raw_datasets = load_dataset(data_args.dataset_name.lower(), data_args.task_name.lower())       

        preprocessor = partial(seq2seq_preprocess if self.is_seq2seq else default_preprocess, 
            tokenizer=tokenizer,
            data_args=data_args)

        raw_datasets = raw_datasets.map(
            preprocessor,
            batched=True,
            desc="Running tokenizer on dataset",
        )

        dataset = raw_datasets[partition].shuffle(seed=int(time.time()))        

        if create_dataloader:

            if data_args.pad_to_max_length:
                # If padding was already done ot max length, we use the default data collator that will just convert everything
                # to tensors.
                data_collator = default_data_collator
            else:
                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                if self.is_seq2seq:
                    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
                else:
                    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

            dataset = DataLoader(
                dataset, 
                shuffle=True, 
                collate_fn=data_collator, 
                batch_size=self.data_args.train_bsz if partition == "train" else self.data_args.eval_bsz
            )

        return dataset

def load_glue_task_val(preprocess_function, task_name):
    eval_name = "validation_matched" if task_name == "mnli" else "validation"
    dataset = load_dataset(
            "glue", task_name, split=eval_name
    )
    print("glue", task_name, dataset)
    dataset = dataset.map(
        partial(preprocess_function, task_name=task_name),
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=["idx", "label"] + list(TASK_TO_KEYS[task_name])
    )
    return dataset

def load_glue_val(preprocess_function):
    raw_datasets = None
    for task_key in TASK_TO_LABELS.keys():
        eval_name = "validation_matched" if task_key == "mnli" else "validation"
        dataset = load_dataset(
            "glue", task_key, split=eval_name
        )
        print("glue", task_key, dataset)
        dataset = dataset.map(
            partial(preprocess_function, task_name=task_key),
            batched=True,
            desc="Running tokenizer on dataset",
            remove_columns=["idx", "label"] + list(TASK_TO_KEYS[task_key])
        )
        if raw_datasets is None:
            raw_datasets = dataset
        else:
            raw_datasets = concatenate_datasets([raw_datasets, dataset])
    print("load_glue_val", raw_datasets)

    return raw_datasets

def load_glue_test(preprocess_function):
    raw_datasets = None
    for task_key in TASK_TO_LABELS.keys():
        eval_name = "test_matched" if task_key == "mnli" else "test"
        dataset = load_dataset(
            "glue", task_key, split=eval_name
        )
        print("glue", task_key, dataset)
        dataset = dataset.map(
            partial(preprocess_function, task_name=task_key),
            batched=True,
            desc="Running tokenizer on dataset",
            remove_columns=["idx", "label"] + list(TASK_TO_KEYS[task_key])
        )
        if raw_datasets is None:
            raw_datasets = dataset
        else:
            raw_datasets = concatenate_datasets([raw_datasets, dataset])
    print("load_glue_val", raw_datasets)

    return raw_datasets

def preprocess_function(examples, task_name, label_to_id):
    # Tokenize the texts
    sentence1_key = TASK_TO_KEYS[task_name][0]
    sentence2_key = None if len(TASK_TO_KEYS[task_name]) == 1 else TASK_TO_KEYS[task_name][1]
    
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
    #     print(examples)
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    else:
        result["labels"] = examples["label"]
    return result

def t5_preprocess_function(examples, tokenizer, task_name, padding="max_length", max_length=128):
        # Tokenize the texts
        sentence1_key = TASK_TO_KEYS[task_name][0]
        sentence2_key = None if len(TASK_TO_KEYS[task_name]) == 1 else TASK_TO_KEYS[task_name][1]
        sentence1_examples = examples[sentence1_key]
        sentence2_examples = None if sentence2_key is None else examples[sentence2_key]
        processed_examples = []
        for i in range(len(sentence1_examples)):
            elements = [
                    task_name, 
                    sentence1_key+":",
                    sentence1_examples[i],
                ]
            if sentence2_examples is not None:
                elements += [
                    sentence2_key+":",
                    sentence2_examples[i],
                ]
            processed_examples.append(" ".join(elements))

        texts = (
            (processed_examples,)
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True, return_tensors="np")

        if "label" in examples:

            labels = examples["label"]
            labels = [TASK_TO_LABELS[task_name][label] for label in labels]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(labels, max_length=2, padding=padding, truncation=True, return_tensors="np")

            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            result["labels"] = labels["input_ids"]

        return result

class DatasetExtractor:
    def __init__(self, model_type, dataset_name, task_name=None):
        self.dataset_name = dataset_name
        self.task_name = task_name

        if task_name is not None:
            self.dataset = load_dataset(dataset_name, task_name)
        else:
            raw_datasets = None
            for task_key in TASK_TO_LABELS.keys():
                dataset = load_dataset(
                    dataset_name, task_key
                )
                eval_name = "validation_matched" if task_key == "mnli" else "validation"
                test_name = "test_matched" if task_key == "mnli" else "test"
                dataset = dataset.map(
                    partial(preprocess_function, task_name=task_key),
                    batched=True,
                    desc="Running tokenizer on dataset",
                    remove_columns=["idx", "label"] + list(TASK_TO_KEYS[task_key])
                )
                if raw_datasets is None:
                    raw_datasets = dataset
                else:
                    raw_datasets['train'] = concatenate_datasets(
                        [raw_datasets['train']] 
                        + [dataset['train']] 
                        * int(np.ceil(max(10000 / len(dataset['train']), 1)))
                    )
                    raw_datasets['validation'] = concatenate_datasets([raw_datasets['validation'], dataset[eval_name]])
                    raw_datasets['test'] = concatenate_datasets([raw_datasets['test'], dataset[test_name]])
            print("raw_datasets", raw_datasets)

    def __call__(self, split_name=None):
        pass
