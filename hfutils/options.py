from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hfutils.constants import TASK_TO_KEYS

import torch

@dataclass
class RayServeArguments:
    cfg: str = field(
        metadata={"help": "The Ray model configuration file path."},
    )

@dataclass
class DatasetArguments:
    task_name: str.lower = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(TASK_TO_KEYS.keys())},
    )
    dataset_name: str = field(
        default=None, 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_bsz: int = field(
        default=2,
        metadata={
            "help": "Batch size of training, default 2."
        },
    )
    eval_bsz: int = field(
        default=2,
        metadata={
            "help": "Batch size of evaluation, default 2."
        },
    )

    max_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    num_labels: int = field(
        default=2,
        metadata={
            "help": "Number of classification labels."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    local_rank: int = field(
        default=-1,
        metadata={
            "help": "Place holder for deepspeed launcher."
        },
    )


@dataclass
class InstanceArguments:
    num_replica: int = field(
        default=1,
        metadata={
            "help": "Number of model instance replications."
        },
    )

    update_cfg: bool = field(
        default=False,
        metadata={
            "help": "Whether to update configuration file. Will not update version."
        },
    )

    update_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to update model softlink. Will update version."
        },
    )

    device_map: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Device map of replication, default on cuda:0, take in list"
        },
    )

@dataclass
class ReplicationOptions:
    replica_id: int
    key: str
    device: torch.device
    # handle: Any = field(default=None)


@dataclass
class ParallelOptions:
    num_stages: int
    parallel_stage: int
    first_parallel_layer: int
    last_parallel_layer: int

    # MODEL REPLICAS
    replication_options: List[ReplicationOptions]
    rr_counter: int = field(default=0)


@dataclass
class EnsembleOptions:
    ensemble_weight: float
    ensemble_pos: int
    threshold: float
    temperature: float
    name: str
    ckpt_path: str
    skip_connection: bool
    parallel: bool

    # MODEL PARALLEL
    parallel_options: List[ParallelOptions]

    # DEPLOYMENT
    ray_actor_options: Dict
    scheduler: str.lower
