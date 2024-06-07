import os
import torch
import torch.nn.functional as F

class EWC:

    def __init__(self):
        self.fisher = {}
        self.cnt = 0

    def save(self, path):
        for n in self.fisher.keys():
            self.fisher[n] = self.fisher[n] / self.cnt
            self.fisher[n].requires_grad = False

        torch.save(self.fisher, os.path.join(path, 'fisher.pth'))

    def load(self, path):
        self.fisher = torch.load(os.path.join(path, 'fisher.pth'), map_location='cpu')
    
    def update(self, model_with_grad, num_samples):

        for n, p in model_with_grad.named_parameters():
            if p.grad is not None:
                if n not in self.fisher:
                    self.fisher[n] = p.grad.data ** 2 * num_samples
                else:
                    self.fisher[n] += p.grad.data ** 2 * num_samples

        self.cnt += num_samples

    def loss(self, model, teacher_model, weight=1.0):
        loss = 0
        for (n_teacher, p_teacher), (n, p) in zip(teacher_model.named_parameters(), model.named_parameters()):
            loss += ((p - p_teacher.to(p.device)) ** 2 * self.fisher[n].to(p.device)).sum()

        return loss * weight