import itertools
from copy import deepcopy
import torch

from datasets import load_dataset, load_from_disk
from torchvision import transforms
from torch.utils.data import DataLoader

from .seq import get_sequence_map

def get_dataloader(data_args, training_args):

    def get_preprocess(data_args):
        if 'ImageNet' in data_args.dataset_name:
            return transforms.Compose(
                [                
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )    
        else:
            return transforms.Compose(
                [
                    transforms.Resize([data_args.image_size, data_args.image_size]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
    
    def transform(examples):
        preprocess = get_preprocess(data_args)
        images = [preprocess(torch.tensor(image) if 'ImageNet' in data_args.dataset_name else image.convert('RGB')) for image in examples["images"]]
        return {"images": images, 'labels': examples['labels']}
    
    data_args.task_num = int(data_args.dataset_name.split('-')[-1].replace('T', ''))
    
    if 'C10-' in data_args.dataset_name:
        dataset = load_dataset('cifar10')
        dataset = dataset.rename_column('img', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 10

    elif data_args.dataset_name.startswith('FM-'):
        dataset = load_dataset('fashion_mnist')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 10

    elif data_args.dataset_name.startswith('M-'):
        dataset = load_from_disk('./hf_home/datasets/mnist')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 10

    elif data_args.dataset_name.startswith('ImageNet'):
        dataset = load_from_disk('./dataloader/imagenet-1k-64')
        dataset = dataset.rename_column('X_train', 'images')
        dataset = dataset.rename_column('Y_train', 'labels')
        dataset['test'] = dataset['validation']
        data_args.tot_class_num = 1000

    elif data_args.dataset_name.startswith('flowers'):
        dataset = load_dataset('nelorth/oxford-flowers')
        dataset = dataset.rename_column('image', 'images')
        dataset = dataset.rename_column('label', 'labels')
        data_args.tot_class_num = 100
    else:
        raise NotImplementedError

    data_args.class_num = data_args.tot_class_num // data_args.task_num
    data_args.task_labels = [data_args.task_id * data_args.class_num + i for i in range(data_args.class_num)]
    data_args.all_task_labels = [
        [i * data_args.class_num + j for j in range(data_args.class_num)] for i in range(data_args.task_num)
    ]

    data_args.sequence = get_sequence_map(data_args.dataset_name, data_args.class_order)

    # preprocess
    def filter(dataset, labels):

        filtered_dataset = deepcopy(dataset)

        final_label = [data_args.sequence[x] for x in dataset['train']['labels']]
        indices = [i for i in range(len(final_label)) if final_label[i] in labels]
        filtered_dataset['train'] = dataset['train'].select(indices)

        final_label = [data_args.sequence[x] for x in dataset['test']['labels']]
        indices = [i for i in range(len(final_label)) if final_label[i] in labels]
        filtered_dataset['test'] = dataset['test'].select(indices)
        
        return filtered_dataset
    

    # continual learning setting

    if training_args.eval:
        
        dataset_list = [
            filter(dataset, task_labels) for task_labels in data_args.all_task_labels[:data_args.task_id]
        ]

        for dataset in dataset_list:
            dataset.set_transform(transform)
        
        dataloader_dict = {
            'all_train_loader': [DataLoader(dataset_list[task_id]['train'], batch_size=training_args.per_device_train_batch_size, shuffle=True) for task_id in range(data_args.task_id)],
            'all_test_loader': [DataLoader(dataset_list[task_id]['test'], batch_size=training_args.per_device_eval_batch_size, shuffle=False) for task_id in range(data_args.task_id)],
        }
    
    elif data_args.noncl:

        data_args.task_labels = list(itertools.chain.from_iterable(data_args.all_task_labels[:data_args.task_id+1]))
        dataset = filter(dataset, data_args.task_labels)
        dataset.set_transform(transform)

        dataloader_dict = {
            'train_loader': DataLoader(dataset['train'], batch_size=training_args.per_device_train_batch_size, shuffle=True),
            'test_loader': DataLoader(dataset['test'], batch_size=training_args.per_device_eval_batch_size, shuffle=False),
        }
    
    else:
        dataset = filter(dataset, data_args.task_labels)
        dataset['train'] = dataset['train'].map(lambda x: {'labels': data_args.sequence[x]}, input_columns='labels')
        dataset['test'] = dataset['test'].map(lambda x: {'labels': data_args.sequence[x]}, input_columns='labels')

        dataset.set_transform(transform)
        dataloader_dict = {
            'train_loader': DataLoader(dataset['train'], batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=8),
            'test_loader': DataLoader(dataset['test'], batch_size=training_args.per_device_eval_batch_size, shuffle=False, num_workers=8),
        }
    
    return dataloader_dict

