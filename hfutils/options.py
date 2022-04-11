from email.policy import default
from typing import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from hfutils.constants import TASK_TO_KEYS

import torch

@dataclass
class RayServeArguments:
    cfg: str = field(
        metadata={"help": "The Ray model configuration file path."},
    )

    deployment: str = field(
        metadata={"help": "Name of the deployment in the configuration file."},
    )

    # tag: str = field(
    #     metadata={"help": "Tagging service monitoring, must be unique, otherwise delete entry from database."},
    # )

@dataclass
class DeepSpeedArguments:
    deepspeed_config: str = field(
        default=None,
        metadata={"help": "DeepSpeed configuration path."},
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

    tag: str = field(
        default="",
        metadata={"help": "Tagging service monitoring, must be unique, otherwise delete entry from database."},
    )

    # local_rank: int = field(
    #     default=-1,
    #     metadata={
    #         "help": "Place holder for deepspeed launcher."
    #     },
    # )


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
    replica: int
    key: str
    # device: torch.device
    # ray_actor_options: Dict

@dataclass
class ParallelOptions:
    stages: int
    ppos: int

    # MODEL REPLICAS
    replications: List[str]
    rr_counter: int = field(default=0)

@dataclass
class EnsembleOptions:
    epos: int
    th: float
    # temp: float
    name: str
    # path: str
    # util_params: List[float]

    # MODEL PARALLEL
    parallel_options: List[ParallelOptions]


@dataclass
class SystemOptions:
    alpha: float  # ensemble exp smooth weight
    ens: int  # number of total ensembles
    type: int  # number of total ensembles

    ensemble_options: List[EnsembleOptions]

@dataclass
class ModelConfig:
    name: str  # model full name
    path: str  # model checkpoint path
    type: str  # model type, e.g., t5 or bert
    stages: int  # number of parallel stages
    ppos: int  # current stage
    epos: int  # current ensemble
    # ens: int  # number of total ensembles
    # alpha: float  # ensemble exp smooth weight
    temp: float  # temperature scaling
    # th: float  # confidence threshold
    util_params: List[float]  # kernel utilization parameter
    # device: torch.device  # device assigned
    ray_actor_options: Dict
    # replica: int
    key: str

@dataclass
class HostOptions:
    host: str
    # alpha: float  # ensemble exp smooth weight
    # ens: int  # number of total ensembles
    type: int  # number of total ensembles
    placement: Dict[str, List[ModelConfig]]

# @dataclass
# class ReplicationOptions:
#     replica_id: int
#     key: str
#     device: torch.device
#     # handle: Any = field(default=None)


# @dataclass
# class ParallelOptions:
#     num_stages: int
#     parallel_stage: int
#     first_parallel_layer: int
#     last_parallel_layer: int

#     # MODEL REPLICAS
#     replication_options: List[ReplicationOptions]
#     rr_counter: int = field(default=0)


# @dataclass
# class EnsembleOptions:
#     ensemble_weight: float
#     ensemble_pos: int
#     threshold: float
#     temperature: float
#     name: str
#     ckpt_path: str
#     skip_connection: bool
#     parallel: bool

#     # MODEL PARALLEL
#     parallel_options: List[ParallelOptions]

#     # DEPLOYMENT
#     scheduler: str.lower
#     ray_actor_options: Dict = None
