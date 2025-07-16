import math
import os

import torch
from aim import Run
from utils import reproducibility

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from aim import Image, Run
from PIL import Image
import torch.nn as nn
import cv2


class NMSELoss:
    def __init__(self):
        pass

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        mse_loss = np.mean((predictions - targets) ** 2)  
        target_variance = np.mean(targets ** 2)  
        nmse = mse_loss / target_variance 
        return nmse

class Trainer:
    def __init__(
        self,
        vqgan: torch.nn.Module,
        vqgan2: torch.nn.Module,
        transformer: torch.nn.Module,
        run: Run,
        config: dict,
        experiment_dir: str = "experiments/250312",
        seed: int = 42,
        device: str = "cuda"
    ) -> None:

        self.vqgan = vqgan
        self.vqgan2 = vqgan2
        self.transformer = transformer

        self.run = run
        self.config = config
        self.experiment_dir = experiment_dir
        self.seed = seed
        self.device = device

        print(f"[INFO] Setting seed to {seed}")
        reproducibility(seed)

        print(f"[INFO] Results will be saved in {experiment_dir}")
        self.experiment_dir = experiment_dir

    def generate_angle(self, dataloader: torch.utils.data.DataLoader, n_images: int = 100, latent_channels=1024):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)  

        i=0
        criterion = NMSELoss()
        Loss_NMSE=0
        for index, imgs in enumerate(dataloader):

            print('index',index)

            RGB = Variable(imgs[0]).to(device=self.device)
            CIR = Variable(imgs[1]).to(device=self.device)

            logits, target = self.transformer(CIR,RGB)
            probs = F.softmax(logits, dim=-1)

            ix = torch.argmax(probs, dim=-1)
            

            sampled_imgs = self.transformer.z_to_image(ix,latent_channels=latent_channels)
            sampled_imgs = sampled_imgs.repeat(1, 3, 1, 1)
            CIR = CIR.repeat(1, 3, 1, 1)

            merged = torch.cat([RGB, sampled_imgs, CIR], dim=0)

            mean = 0.5
            std = 0.5
            merged = merged * std + mean

            merged1 = merged * 2 * math.pi
            merged1 = merged1.clamp(0, 2 * math.pi).cpu().numpy()

            merged = merged * 255.0
            merged = merged.clamp(0, 255).cpu().numpy().astype(np.uint8)

            Loss_NMSE = Loss_NMSE + criterion(merged1[1,:,:,:], merged1[2,:,:,:])


            images = []
            for j in range(merged.shape[0]):
                img = merged[j]
                img = np.transpose(img, (1, 2, 0))

                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)

                images.append(img)


            gray_image = cv2.cvtColor(images[1], cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
            images[1] = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

            gray_image = cv2.cvtColor(images[2], cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
            images[2] = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

            merged_image = np.concatenate(images, axis=1)  
            image = Image.fromarray(merged_image)
            image.save(os.path.join(self.experiment_dir, f"generated_{i}.jpg"))

            i=i+1
            if i==n_images:
                break

