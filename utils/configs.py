import os
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class DataArguments:
    dataset_name: str = field(
        default='M-5T',
        metadata={"help": "The name of the dataset to use. e.g., C10-5T"},
    )
    task_id: int = field(
        default=0,
        metadata={"help": "The task id to use. e.g., 0"},
    )
    image_size: int = field(
        default=32,
    )
    class_order: Literal[0, 1, 2, 3, 4] = field(
        default=0
    )
    tot_samples_for_eval: int = field(
        default=2048,
    )
    noncl: bool = field(
        default=False,
        metadata={"help": "non-continual learning"},
    )

@dataclass
class ModelArguments:
    model_arch: Literal['ddim', 'gan'] = field(
        default='ddim'
    )
    method: Literal['noncl', 'naive', 'er', 'agem', 'generative_replay', 'l2', 'noncl', 'ewc', 'kd', 'si', 'mas', 'lora', 'ensemble'] = field(
        default='naive'
    )

    # Diffusion
    diffusion_time_steps: int = field(default=1000)
    inference_steps: int = field(default=50)

@dataclass
class TrainArguments:
    seed: int = field(default=42)
    eval: bool = field(default=False)
    num_train_epochs: int = field(default=200)
    per_device_train_batch_size: int = field(default=1024)
    per_device_eval_batch_size: int = field(default=512)
    validation_batch_size: int = field(default=256)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.0001)
    logging_dir: str = field(default='logs')
    logging_steps: int = field(default=100)
    lr_warmup_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    mixed_precision: str = field(default=None)
    gradient_accumulation_steps: int = field(default=1)

    check_done: bool = field(default=True)

    # ensemble
    ensemble: bool = field(default=False)

    # replay-related
    replay: bool = field(default=False)
    memory_size: int = field(default=200)
    replay_batch_size: int = field(default=256)
    
    # generative replay (don't need to set replay=True)
    generative_replay: bool = field(default=False)

    # ER
    er: bool = field(default=False)

    # AGEM
    agem: bool = field(default=False)

    # L2
    L2: bool = field(default=False)
    L2_weight: float = field(default=500.0)

    # EWC
    ewc: bool = field(default=False)
    ewc_weight: float = field(default=50000.0)

    # MAS
    mas: bool = field(default=False)
    mas_weight: float = field(default=5.0)

    # SI
    si: bool = field(default=False)
    si_weight: float = field(default=5.0)
    si_epsilon: float = field(default=0.01)

    # KD
    kd: bool = field(default=False)
    kd_weight: float = field(default=1.0)

def update_configs(model_args, data_args, train_args):
    
    if model_args.method == 'noncl':
        data_args.noncl = True

    train_args.logging_dir = f'logs/model_arch={model_args.model_arch}/method={model_args.method}/dataset_name={data_args.dataset_name}/class_order={data_args.class_order}/seed={train_args.seed}/task_id={data_args.task_id}'
    
    if train_args.eval:
        train_args.logging_dir += '/eval'

    os.makedirs(train_args.logging_dir, exist_ok=True)

    if data_args.task_id > 0:
        train_args.prev_dir = f'logs/model_arch={model_args.model_arch}/method={model_args.method}/dataset_name={data_args.dataset_name}/class_order={data_args.class_order}/seed={train_args.seed}/task_id={data_args.task_id-1}'
    else:
        train_args.prev_dir = None

    train_args.all_dirs = [
        f'logs/model_arch={model_args.model_arch}/method={model_args.method}/dataset_name={data_args.dataset_name}/class_order={data_args.class_order}/seed={train_args.seed}/task_id={i}'
        for i in range(data_args.task_id)
    ]

    return model_args, data_args, train_args
    