from .diffusion import ddim, lora_ddim

def get_model(model_args, data_args, training_args):
    '''
        model factory function
        A model must implement:
            train_step(x, y, task_id=None) -> loss
            sample(bs, seed) -> image
            load()
            save()
    '''
    if 'ddim' in model_args.model_arch and 'lora' in model_args.method:
        model = lora_ddim.Learner(model_args, data_args, training_args)
    elif 'ddim' in model_args.model_arch:
        model = ddim.Learner(model_args, data_args, training_args)
    else:
        raise NotImplementedError
    
    return model