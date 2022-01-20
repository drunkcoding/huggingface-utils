import argparse
import sys
import os

from dataclasses import dataclass, field
from typing import List, Optional
from transformers import HfArgumentParser
from hfutils.options import DatasetArguments, InstanceArguments, ModelArguments, RayServeArguments



class RayArguments:
    parser = HfArgumentParser((RayServeArguments, DatasetArguments))
    def __init__(self):
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            self.serve_args, self.data_args  = self.parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            self.serve_args, self.data_args  = self.parser.parse_args_into_dataclasses()

class HfArguments:
    parser = HfArgumentParser((ModelArguments, DatasetArguments))
    def __init__(self):
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            self.model_args, self.data_args = self.parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            self.model_args, self.data_args = self.parser.parse_args_into_dataclasses()

class DeployArguments:
    parser = HfArgumentParser((ModelArguments, DatasetArguments, InstanceArguments))
    def __init__(self):
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            self.model_args, self.data_args, self.ins_args = self.parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            self.model_args, self.data_args, self.ins_args = self.parser.parse_args_into_dataclasses()

class GlueArgParser(argparse.ArgumentParser):

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    def __init__(self, *args, **kwargs):
        super(GlueArgParser, self).__init__(*args, **kwargs)

        # self.add_argument(
        #     "--cfg",
        #     type=str,
        #     help="Test Configuration",
        #     required=True,
        # )

        self.add_argument(
            "--task_name",
            type=str.lower,
            default=None,
            help="The name of the glue task to train on.",
            choices=list(self.task_to_keys.keys()),
        )
        self.add_argument(
            "--model_name_or_path",
            type=str,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
            required=True,
        )
        self.add_argument(
            "--eval_file", type=str, default=None, help="A csv or a json file containing the training data."
        )
        self.add_argument(
            "--max_length",
            type=int,
            default=128,
            help=(
                "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
            ),
        )
        self.add_argument(
            "--per_device_eval_batch_size",
            type=int,
            default=1,
            help="Batch size (per device) for the evaluation dataloader.",
        )
        self.add_argument(
            "--pad_to_max_length",
            action="store_true",
            help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
        )

    def parse(self):
        args = self.parse_args()
        return args
