import torch
import numpy as np

from torchmetrics.image.fid import FrechetInceptionDistance

class Validator:
    
    def __init__(self, data_args, training_args):
        
        self.training_args = training_args
        self.data_args = data_args
        
        self.FID = FrechetInceptionDistance(reset_real_features=False).set_dtype(torch.float64)
        self.FID_prepared = False

    def to(self, device):
        self.FID.to(device)

    @torch.no_grad()
    def _compute_fid(self, samples, dataloader):
        
        if not self.FID_prepared:
            for data in dataloader:
                images = data['images'].to(self.FID.device)
                images = (images + 1) * 127.5
                images = images.to(torch.uint8)
                self.FID.update(images, real=True)
            self.FID_prepared = True

        images = np.concatenate([np.array(sample)[np.newaxis, ...] for sample in samples], axis=0).transpose(0, 3, 1, 2)
        images = torch.tensor(images)
        for i in range(0, len(images), self.training_args.validation_batch_size):
            self.FID.update(images[i:i+self.training_args.validation_batch_size].to(self.FID.device).to(torch.uint8), real=False)

        fid = self.FID.compute().item()
        self.FID.reset()

        return fid
    
    def evaluate(self, samples, labels, dataloader):
        fid = self._compute_fid(samples, dataloader)
        return {'fid': fid, 'metric_for_validation': fid}
