import torch
import torch.nn.functional as F
import random
import itertools
from copy import deepcopy

from diffusers import UNet2DModel, DDIMScheduler
from .pipeline_ddim import MyDDIMPipeline
from diffusers.utils import make_image_grid
from ..memory import Memory
from ..regularize.ewc import EWC
from ..regularize.mas import MAS
from ..regularize.si import SI
from ..replay.agem import AGEM
from .base import BaseLearner

class Learner(BaseLearner):

    def __init__(self, model_args, data_args, training_args):

        super(Learner, self).__init__()
        
        if data_args.image_size == 32:
            
            block_out_channels = (128, 256, 256, 256)
            down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")

        elif data_args.image_size == 64:
            
            block_out_channels = (128, 256, 384, 512)
            down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
            up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")

        elif data_args.image_size == 128:

            block_out_channels = (128, 128, 256, 384, 512)
            down_block_types = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D")
            up_block_types = ("UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")

        self.model = UNet2DModel(
            sample_size=data_args.image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            downsample_type='resnet',
            upsample_type='resnet',
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            num_class_embeds=data_args.tot_class_num,
            dropout=0.1,
        )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=model_args.diffusion_time_steps,
        )
        self.pipeline = MyDDIMPipeline(
            unet=self.model, scheduler=self.scheduler
        )
        self.inference_steps = model_args.inference_steps
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        
        load_from_prev_dir_flag = False
        # case 1: evaluation
        if self.training_args.eval:
            load_from_prev_dir_flag = True
        elif self.data_args.task_id > 0 and self.model_args.method != 'noncl' and self.model_args.method != 'ensemble':
            load_from_prev_dir_flag = True

        if load_from_prev_dir_flag:
            self.load(self.training_args.prev_dir)

        # --------- replay-based methods --------- #
        
        # experience replay 
        if self.model_args.method == 'er':
            self.memory = Memory(self.training_args.memory_size, data_args)
            if self.data_args.task_id > 0:
                self.memory.load(self.training_args.prev_dir)
        
        # generative replay
        if self.model_args.method == 'generative_replay' and self.data_args.task_id > 0:
            self.teacher_model = deepcopy(self.model)
            self.teacher_labels = list(itertools.chain.from_iterable(data_args.all_task_labels[:self.data_args.task_id]))
            self.teacher_pipeline = MyDDIMPipeline(
                unet=self.teacher_model, scheduler=self.scheduler,
            )

        # --------- regularization-based methods --------- #
        
        # AGEM
        if self.model_args.method == 'agem':
            self.agem = AGEM(self.model)

        # L2 regularization
        if self.model_args.method == 'l2':
            self.teacher_model = deepcopy(self.model)
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        # EWC: load fisher matrix
        if self.model_args.method == 'ewc':
            self.ewc = EWC()
            if self.data_args.task_id > 0:
                self.ewc.load(self.training_args.prev_dir)
                self.teacher_model = deepcopy(self.model)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
        
        if self.model_args.method == 'mas':
            self.mas = MAS()
            if self.data_args.task_id > 0:
                self.mas.load(self.training_args.prev_dir)
                self.teacher_model = deepcopy(self.model)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

        if self.model_args.method == 'si':
            self.si = SI()
            
            # notice that in SI, we load the teacher model at the first task.
            self.teacher_model = deepcopy(self.model)   

            if self.data_args.task_id > 0:
                self.si.load(self.training_args.prev_dir)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

        if self.model_args.method == 'kd' and self.data_args.task_id > 0:
            self.teacher_model = deepcopy(self.model)
            for p in self.teacher_model.parameters():
                p.requires_grad = False

    def train_step(self, x, y):
        
        loss = 0.

        # --------- pre-computed methods: replay-based --------- #

        if self.training_args.er and not self.memory.empty():
            x_replay, y_replay = self.memory.sample(self.training_args.replay_batch_size)
            x = torch.cat([x, x_replay.to(x.device)], dim=0)
            y = torch.cat([y, y_replay.to(y.device)], dim=0)
        
        if self.training_args.generative_replay and self.data_args.task_id > 0:
            self.teacher_model = self.teacher_model.to(self.model.device)
            # sample from teacher model
            labels = random.choices(
                self.teacher_labels,
                k=self.training_args.replay_batch_size
            )
            y_replay = torch.tensor(labels, device=self.teacher_model.device, dtype=torch.long)
            x_replay = self.teacher_pipeline(
                batch_size=self.training_args.replay_batch_size,
                labels=y_replay,
                num_inference_steps=self.inference_steps,
                output_type='tensor',
                show_progress=False
            ).images
            x = torch.cat([x, x_replay], dim=0)
            y = torch.cat([y, y_replay], dim=0)

        # --------- vanilla training snippet --------- #
        
        noise = torch.randn(x.shape, device=x.device)
        timesteps = torch.randint(
            0, self.model_args.diffusion_time_steps, (x.shape[0],), device=x.device, dtype=torch.int64
        )
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.model(noisy_x, timesteps, class_labels=y, return_dict=False)[0]
        loss += F.mse_loss(noise_pred, noise)


        # --------- pre-computed methods: regularization-based --------- #

        if self.model_args.method == 'l2' and self.data_args.task_id > 0:
            l2_loss = 0
            self.teacher_model = self.teacher_model.to(self.model.device)
            for (n_teacher, p_teacher), (n, p) in zip(self.teacher_model.named_parameters(), self.model.named_parameters()):
                # assert n_teacher == n or n_teacher in n or n in n_teacher
                l2_loss += F.mse_loss(p, p_teacher) * self.training_args.L2_weight
            loss = loss + l2_loss

        if self.model_args.method == 'ewc' and self.data_args.task_id > 0:
            self.teacher_model = self.teacher_model.to(self.model.device)
            ewc_loss = self.ewc.loss(self.model, self.teacher_model, weight=self.training_args.ewc_weight)
            loss = loss + ewc_loss

        if self.model_args.method == 'si' and self.data_args.task_id > 0:
            si_loss = self.si.loss(self.model, self.teacher_model, weight=self.training_args.si_weight)
            loss = loss + si_loss

        if self.model_args.method == 'mas' and self.data_args.task_id > 0:
            self.teacher_model = self.teacher_model.to(self.model.device)
            mas_loss = self.mas.loss(self.model, self.teacher_model, weight=self.training_args.mas_weight)
            loss = loss + mas_loss

        if self.model_args.method == 'kd' and self.data_args.task_id > 0:
            self.teacher_model = self.teacher_model.to(self.model.device)
            with torch.no_grad():
                noise_pred_teacher = self.teacher_model(noisy_x, timesteps, class_labels=y, return_dict=False)[0]
            kd_loss = F.mse_loss(noise_pred, noise_pred_teacher) * self.training_args.kd_weight
            loss = loss + kd_loss

        return loss

    def regularize_with_gradient_before_step(self):

        if self.model_args.method == 'agem' and not self.memory.empty():
            x_replay, y_replay = self.memory.sample(self.training_args.replay_batch_size)
            x_replay, y_replay = x_replay.to(self.model.device), y_replay.to(self.model.device)
            noise = torch.randn(x_replay.shape, device=x_replay.device)
            timesteps = torch.randint(
                0, self.model_args.diffusion_time_steps, (x_replay.shape[0],), device=x_replay.device, dtype=torch.int64
            )
            noisy_x = self.scheduler.add_noise(x_replay, noise, timesteps)
            noise_pred = self.model(noisy_x, timesteps, class_labels=y_replay, return_dict=False)[0]
            replay_loss = F.mse_loss(noise_pred, noise)
            self.agem.penalty(self.model, replay_loss)

    
    def regularize_with_gradient_after_step(self):

        if self.model_args.method == 'si':
            self.si.update_W(self.model)


    @torch.no_grad()
    def sample(self, bs, seed, labels):
        
        # FIXME: make this part extensible!

        if self.model_args.method == 'ensemble' and self.training_args.eval:
            task_id = (labels[0] // self.data_args.class_num).item()
            self.model.load_state_dict(
                torch.load(
                    f'{self.training_args.all_dirs[task_id]}/model.pth',
                    map_location=self.model.device
                )
            )

        self.pipeline.to(self.model.device)
        image = self.pipeline(
            batch_size=bs,
            labels=labels,
            num_inference_steps=self.inference_steps,
            generator=torch.manual_seed(seed),
            output_type='pil'
        ).images
        return image
    
    def load(self, path):
        self.model.load_state_dict(torch.load(f'{path}/model.pth', map_location='cpu'))

    def save(self, path, dataloader):

        labels = random.choices(
            self.data_args.task_labels,
            k=self.training_args.per_device_eval_batch_size
        )
        
        images = self.sample(
            self.training_args.per_device_eval_batch_size, 
            self.training_args.seed,
            labels=torch.tensor(labels, device=self.model.device, dtype=torch.long)
        )
        
        make_image_grid(
            images[:int(self.training_args.per_device_eval_batch_size ** 0.5)*int(self.training_args.per_device_eval_batch_size ** 0.5)], 
            rows=int(self.training_args.per_device_eval_batch_size ** 0.5), 
            cols=int(self.training_args.per_device_eval_batch_size ** 0.5)
        ).save(f'{path}/samples.png')

        torch.save(self.model.state_dict(), f'{path}/model.pth')
        

        if self.model_args.method == 'er':
            for _, batch in enumerate(dataloader):
                x, y = batch['images'], batch['labels']
                self.memory.add(x, y)

            self.memory.save(path)

        if self.model_args.method == 'ewc' or self.model_args.method == 'mas':
            for _, batch in enumerate(dataloader):
                x, y = batch['images'], batch['labels']
                noise = torch.randn(x.shape, device=x.device)
                timesteps = torch.randint(
                    0, self.model_args.diffusion_time_steps, (x.shape[0],), device=x.device, dtype=torch.int64
                )
                noisy_x = self.scheduler.add_noise(x, noise, timesteps)
                loss = 0.
                with torch.enable_grad():
                    noise_pred = self.model(noisy_x, timesteps, class_labels=y, return_dict=False)[0]
                    if self.model_args.method == 'ewc':
                        loss += F.mse_loss(noise_pred, noise)
                        loss.backward()
                        self.ewc.update(self.model, x.shape[0])
                    elif self.model_args.method == 'mas':
                        loss = torch.sum(noise_pred.norm(2, dim=-1))
                        loss.backward()
                        self.mas.update(self.model, x.shape[0])
            
            if self.model_args.method == 'ewc':
                self.ewc.save(path)
            elif self.model_args.method == 'mas':
                self.mas.save(path)

        if self.model_args.method == 'si':
            self.si.update_omega(self.model, self.teacher_model, epsilon=self.training_args.si_epsilon)
            self.si.save(path)

        