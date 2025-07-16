"""
https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
"""

# Importing Libraries
import os

import imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from aim import Image, Run
from utils import weights_init
from VQGAN.trainer.vqgan_wt import Discriminator

import time


def channel_adopter(x):
    if x.shape[1] == 2:
        return torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    elif x.shape[1] == 3:
        return x
    else:
        return x


class VQGANTrainer:
    """Trainer class for VQGAN, contains step, train methods"""

    def __init__(
            self,
            model: torch.nn.Module,
            run: Run,
            # Training parameters
            device: str or torch.device = "cuda",
            learning_rate: float = 2.25e-05,
            beta1: float = 0.5,
            beta2: float = 0.9,
            # Loss parameters
            perceptual_loss_factor: float = 1.0,
            rec_loss_factor: float = 1.0,
            # Discriminator parameters
            disc_factor: float = 1.0,
            disc_start: int = 100,
            # Miscellaneous parameters
            experiment_dir: str = "./experiments",
            expriment_save_name: str = "reconstruction_RGB.gif",
            perceptual_model: str = "vgg",
            save_every: int = 10,
            alpha_loss_factor: float = 1.0,
    ):

        self.run = run
        self.device = device

        # VQGAN parameters
        self.vqgan = model

        # Discriminator parameters
        self.discriminator = Discriminator(img_channels=self.vqgan.img_channels).to(
            self.device
        )
        self.discriminator.apply(weights_init)

        # Loss parameters
        self.perceptual_loss = lpips.LPIPS(net='vgg').to(self.device)

        # Optimizers
        self.opt_vq, self.opt_disc = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

        # Hyperprameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.rec_loss_factor = rec_loss_factor
        self.alpha_loss_factor = alpha_loss_factor      

        # Save directory
        self.expriment_save_dir = experiment_dir
        self.expriment_save_name = expriment_save_name

        # Miscellaneous
        self.global_step = 0
        self.sample_batch = None
        self.gif_images = []
        self.save_every = save_every

    def configure_optimizers(
            self, learning_rate: float = 2.25e-05, beta1: float = 0.5, beta2: float = 0.9
    ):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.quant_conv.parameters())
            + list(self.vqgan.post_quant_conv.parameters()),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )

        return opt_vq, opt_disc

    def step(self, imgs: torch.Tensor) -> torch.Tensor:  # 把图像输给vqgan
        """Performs a single training step from the dataloader images batch

        For the VQGAN, it calculates the perceptual loss, reconstruction loss, and the codebook loss and does the backward pass.

        For the discriminator, it calculates lambda for the discriminator loss and does the backward pass.

        Args:
            imgs: input tensor of shape (batch_size, channel, H, W)

        Returns:
            decoded_imgs: output tensor of shape (batch_size, channel, H, W)
        """

        # Getting decoder output
        decoded_images, _, q_loss = self.vqgan(imgs)  # 返回了codebook loss
    
        """
        =======================================================================================================================
        VQ Loss
        """
        # imgs0=channel_adopter(imgs)
        # decoded_images0 = channel_adopter(decoded_images)

        if imgs.shape[1] == 4 and decoded_images.shape[1] == 4:
            # RGB部分
            imgs_rgb = imgs[:, :3, ...]
            decoded_images_rgb = decoded_images[:, :3, ...]
            perceptual_loss = self.perceptual_loss(imgs_rgb, decoded_images_rgb)
            rec_loss = torch.abs(imgs_rgb - decoded_images_rgb)
            # Alpha部分
            imgs_alpha = imgs[:, 3:, ...]
            decoded_images_alpha = decoded_images[:, 3:, ...]
            alpha_loss = torch.abs(imgs_alpha - decoded_images_alpha).mean()
            # 合并loss
            perceptual_rec_loss = (
                self.perceptual_loss_factor * perceptual_loss
                + self.rec_loss_factor * rec_loss
                + self.alpha_loss_factor * alpha_loss
            )
        else:
            perceptual_loss = self.perceptual_loss(imgs, decoded_images)
            rec_loss = torch.abs(imgs - decoded_images)
            perceptual_rec_loss = (
                self.perceptual_loss_factor * perceptual_loss
                + self.rec_loss_factor * rec_loss
            )
        perceptual_rec_loss = perceptual_rec_loss.mean()  # 组合起来的重构loss

        """
        =======================================================================================================================
        Discriminator Loss
        """
        disc_real = self.discriminator(imgs)
        disc_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )

        g_loss = -torch.mean(disc_fake)

        λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1.0 - disc_real))
        d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        # ======================================================================================================================
        # Tracking metrics

        self.run.track(
            perceptual_rec_loss,
            name="Perceptual & Reconstruction loss",
            step=self.global_step,
            context={"stage": "vqgan"},
        )

        self.run.track(
            vq_loss, name="VQ Loss", step=self.global_step, context={"stage": "vqgan"}
        )
        self.run.track(
            gan_loss, name="GAN Loss", step=self.global_step, context={"stage": "vqgan"}
        )

        # =======================================================================================================================
        # Backpropagation

        self.opt_vq.zero_grad()
        vq_loss.backward(
            retain_graph=True
        )  # retain_graph is used to retain the computation graph for the discriminator loss

        self.opt_disc.zero_grad()
        gan_loss.backward()

        self.opt_vq.step()
        self.opt_disc.step()

        return decoded_images, vq_loss, gan_loss

    def train(
            self,
            dataloader: torch.utils.data.DataLoader,
            epochs: int = 1,
    ):
        """Trains the VQGAN for the given number of epochs

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader to use.
            epochs (int, optional): number of epochs to train for. Defaults to 100.
        """

        for epoch in range(epochs):
            for index, imgs in enumerate(dataloader):

                # Training step
                imgs = imgs.to(self.device)

                #print('imgs',imgs.shape)
                # 计算训练时间
                #start_time = time.time()
                decoded_images, vq_loss, gan_loss = self.step(imgs)
                #end_time = time.time()
                #training_duration = end_time - start_time
                #print(f"Training time: {training_duration:.6f} seconds")
                # Updating global step
                self.global_step += 1

                if index % self.save_every == 0:

                    print(
                        f"Epoch: {epoch + 1}/{epochs} | Batch: {index}/{len(dataloader)} | VQ Loss : {vq_loss:.4f} | Discriminator Loss: {gan_loss:.4f}"
                    )

                    # Only saving the gif for the first 2000 save steps
                    '''if self.global_step // self.save_every <= 2000:
                        self.sample_batch = (
                            imgs[:] if self.sample_batch is None else self.sample_batch
                        )

                        with torch.no_grad():
                            """
                            Note : Lots of efficiency & cleaning needed here
                            """

                            gif_img = (
                                torchvision.utils.make_grid(
                                    torch.cat(
                                        (
                                            self.sample_batch,
                                            self.vqgan(self.sample_batch)[0],
                                        ),
                                    )
                                )
                                .detach()
                                .cpu()
                                .permute(1, 2, 0)
                                .numpy()
                            )

                            gif_img = (gif_img - gif_img.min()) * (
                                    255 / (gif_img.max() - gif_img.min())
                            )
                            gif_img = gif_img.astype(np.uint8)

                            self.run.track(
                                Image(
                                    torchvision.utils.make_grid(
                                        torch.cat(
                                            (
                                                imgs,
                                                decoded_images,
                                            ),
                                        )
                                    ).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                                ),
                                name="VQGAN Reconstruction",
                                step=self.global_step,
                                context={"stage": "vqgan"},
                            )

                            self.gif_images.append(gif_img)

                        imageio.mimsave(
                            os.path.join(self.expriment_save_dir, self.expriment_save_name),
                            self.gif_images,
                            fps=5,
                        )
                        print("reconstruction.gif has been generated")'''
