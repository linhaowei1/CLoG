import logging
from tqdm.auto import tqdm
import os
import random
import torch
from torch.optim import Adam
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from utils.metrics import Validator

@torch.no_grad()
def evaluate(model, dataloaders, model_args, data_args, training_args):

    accelerator = Accelerator()

    logging.basicConfig(filename=os.path.join(training_args.logging_dir, 'eval_log.txt'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
    
    logging.info("Running CIL...")

    os.makedirs(training_args.logging_dir, exist_ok=True)

    model.model = accelerator.prepare(model.model)


    model.eval()

    for task_id in range(data_args.task_id):
        
        validator = Validator(data_args, training_args)
        validator.to(accelerator.device)

        samples = []
        all_labels = []
        while len(samples) < data_args.tot_samples_for_eval:
            
            # FIXME: assume uniform label distribution
            labels = random.choices(
                data_args.all_task_labels[task_id],
                k=training_args.per_device_eval_batch_size
            )

            samples.extend(
                model.sample(
                    training_args.per_device_eval_batch_size, 
                    training_args.seed + len(samples),
                    labels=torch.tensor(labels, device=accelerator.device, dtype=torch.long)
                )
            )

            all_labels.extend(labels)

        eval_logs = validator.evaluate(samples[:data_args.tot_samples_for_eval], all_labels, dataloaders['all_test_loader'][task_id])
        
        rows = cols = 16
        make_image_grid(
            samples[:rows * cols], 
            rows=rows, 
            cols=cols
        ).save(f'{training_args.logging_dir}/{task_id}.png')

        logging.info(f"task {task_id}: {str(eval_logs)}")

        del validator

