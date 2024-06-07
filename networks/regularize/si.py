import os
import torch
import torch.nn.functional as F

class SI:

    def __init__(self):
        self.omega = {}
        self.W = {}
        self.p_old = {}

    def save(self, path):
        torch.save(self.omega, os.path.join(path, 'omega.pth'))

    def load(self, path):
        self.omega = torch.load(os.path.join(path, 'omega.pth'), map_location='cpu')
    
    def update_omega(self, model, teacher_model, epsilon=0.1):
        '''
            After completing training on a task, update the per-param regularization strength
            [W]         <dict> estimated param-sepcific contribution to changes in total loss of completed task
            [epsilon]   <float> dampening parameter for stabilizing training
        '''

        for (n_teacher, p_teacher), (n, p) in zip(teacher_model.named_parameters(), model.named_parameters()):

            if p.requires_grad:

                delta_p = p.detach().clone() - p_teacher.to(p.device)
                omega_add = self.W[n] / (delta_p ** 2 + epsilon)
                if n not in self.omega:
                    self.omega[n] = omega_add
                else:
                    self.omega[n] = omega_add + self.omega[n].to(p.device)


    def update_W(self, model_with_grad):

        for n, p in model_with_grad.named_parameters():

            if n not in self.W:
                self.W[n] = p.clone().detach().zero_()
                self.p_old[n] = p.clone().detach()
            else:
                if p.grad is not None:
                    self.W[n] += (-p.grad * (p.detach() - self.p_old[n]))
                self.p_old[n] = p.clone().detach()
            
            # print(self.W[n].min())
            
        


    def loss(self, model, teacher_model, weight=1.0):
        loss = 0
        for (n_teacher, p_teacher), (n, p) in zip(teacher_model.named_parameters(), model.named_parameters()):
            loss += ((p - p_teacher.to(p.device)) ** 2 * self.omega[n].to(p.device)).sum()

        return loss * weight