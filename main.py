from transformers import HfArgumentParser

from utils.configs import ModelArguments, DataArguments, TrainArguments, update_configs
from dataloader.data import get_dataloader
from networks.model import get_model
from approaches.train import Trainer
from approaches.eval import evaluate

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args, data_args, training_args = update_configs(model_args, data_args, training_args)
    
    print(model_args, data_args, training_args)
    
    dataloaders = get_dataloader(data_args, training_args)
    model = get_model(model_args, data_args, training_args)

    if training_args.eval:
        evaluate(model, dataloaders, model_args, data_args, training_args)
    else:
        trainer = Trainer(model, dataloaders, model_args, data_args, training_args)
        trainer.train()

if __name__ == '__main__':
    main()