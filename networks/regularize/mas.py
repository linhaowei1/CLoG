import os
import torch
import torch.nn.functional as F

class MAS:

    def __init__(self):
        self.omega = {}

    def save(self, path):
        torch.save(self.omega, os.path.join(path, 'omega.pth'))

    def load(self, path):
        self.omega = torch.load(os.path.join(path, 'omega.pth'), map_location='cpu')
    
    def update(self, model_with_grad, num_samples):

        for n, p in model_with_grad.named_parameters():
            
            if p.grad is not None:

                if n not in self.omega:
                    self.omega[n] = p.grad.data.abs() / num_samples
                else:
                    self.omega[n] = self.omega[n].to(p.grad.device) + p.grad.data.abs() / num_samples

    def loss(self, model, teacher_model, weight=1.0):
        loss = 0
        for (n_teacher, p_teacher), (n, p) in zip(teacher_model.named_parameters(), model.named_parameters()):
            loss += torch.sum(self.omega[n].to(p.device) * (p_teacher.to(p.device) - p).pow(2)) / 2

        return loss * weight