import numpy as np
import os
import torch
import torch.nn.functional as F

class AGEM:

    def __init__(self, model):
        
        self.grad_xy, self.grad_er = {}, {}

        for n, param in model.named_parameters():
            self.grad_xy[n] = torch.zeros_like(param)
            self.grad_er[n] = torch.zeros_like(param)


    def store_grad(self, named_params, grads):
        
        for n, p in named_params():
            
            grads[n] = grads[n].to(p.grad.device)

            if p.grad is not None:
                grads[n].copy_(p.grad.data)
        
    def dot_product(self, x, y):
        return sum([torch.dot(x[n].view(-1), y[n].view(-1)) for n in x.keys()])

    def penalty(self, model, replay_loss):

        self.store_grad(model.named_parameters, self.grad_xy)
        
        # this may not support accelerator
        replay_loss.backward()
        self.store_grad(model.named_parameters, self.grad_er)

        dot_prod = self.dot_product(self.grad_xy, self.grad_er)

        if dot_prod.item() < 0:
            g_tilde = self.project(gxy=self.grad_xy, ger=self.grad_er)
            self.overwrite_grad(model.named_parameters, g_tilde)
        else:
            self.overwrite_grad(model.named_parameters, self.grad_xy)

    def project(self, gxy, ger):
        corr = self.dot_product(gxy, ger) / self.dot_product(ger, ger)
        return gxy - corr * ger
    
    def overwrite_grad(self, params, new_grad):
        count = 0
        for n, p in params():
            if p.grad is not None:
                this_grad = new_grad[n].contiguous().view(p.grad.data.size())
                p.grad.data.copy_(this_grad)
            count += 1
