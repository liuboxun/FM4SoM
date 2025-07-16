# Importing Libraries
import os
import cv2
import imageio
import torch
import numpy as np
import torchvision
from aim import Run
from PIL import Image
from utils import reproducibility

from trainer import TransformerTrainer, VQGANTrainer
from torch.autograd import Variable
import torch.nn.functional as F
import time


class Trainer:
    def __init__(
        self,
        vqgan: torch.nn.Module,
        vqgan2: torch.nn.Module,
        transformer: torch.nn.Module,
        run: Run,
        config: dict,
        experiment_dir: str = "experiments",
        seed: int = 42,
        device: str = "cuda:1",
    ) -> None:

        self.vqgan = vqgan
        self.vqgan2 = vqgan2
        self.transformer = transformer

        self.run = run
        self.config = config
        self.experiment_dir = experiment_dir
        self.seed = seed
        self.device = torch.device("cuda:1")

        print(f"[INFO] Setting seed to {seed}")
        reproducibility(seed)

        print(f"[INFO] Results will be saved in {experiment_dir}")
        self.experiment_dir = experiment_dir

#### 微调工作的train
#### 63-50_400s
    def train_vqgan_50_400s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_400s/vqgan_pl_fh_epoch50_bs32_0531_70-50_400s.pt")
        )


    def train_vqgan2_50_rgbd_400s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_400s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_400s.pt")  
        )

    def train_transformers_50_400s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_400s/transformer_fh_epoch50_bs32_0531_70-50_400s.pt")
        )

    def generate_crossmodal_63_50_400s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            all_nmse.append(nmse)
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_400s/fh_{i}_epoch50_0531_70-50_400s.jpg"), merged)

            i=i+1
            if i==n_images:
                break

    ### 63-50_500s
    def train_vqgan_50_500s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_500s/vqgan_pl_fh_epoch50_bs32_0531_70-50_500s.pt")
        )


    def train_vqgan2_50_rgbd_500s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_500s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_500s.pt")  
        )

    def train_transformers_50_500s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_500s/transformer_fh_epoch50_bs32_0531_70-50_500s.pt")
        )

    def generate_crossmodal_63_50_500s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            all_nmse.append(nmse)
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_500s/fh_{i}_epoch50_0531_70-50_500s.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    ### 63-50_300s
    def train_vqgan_50_300s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_300s/vqgan_pl_fh_epoch50_bs32_0531_70-50_300s.pt")
        )


    def train_vqgan2_50_rgbd_300s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_300s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_300s.pt")  
        )

    def train_transformers_50_300s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_300s/transformer_fh_epoch50_bs32_0531_70-50_300s.pt")
        )

    def generate_crossmodal_63_50_300s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            all_nmse.append(nmse)
            print(f"NMSE: {nmse}")
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_300s/fh_{i}_epoch50_0531_70-50_300s.jpg"), merged)

            i=i+1
            if i==n_images:
                break
        print(f"average NMSE: {np.mean(all_nmse)}") 


    ### 63-50_200s
    def train_vqgan_50_200s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_200s/vqgan_pl_fh_epoch50_bs32_0531_70-50_200s.pt")
        )


    def train_vqgan2_50_rgbd_200s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_200s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_200s.pt")  
        )

    def train_transformers_50_200s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_200s/transformer_fh_epoch50_bs32_0531_70-50_200s.pt")
        )

    def generate_crossmodal_63_50_200s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            all_nmse.append(nmse)
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_200s/fh_{i}_epoch50_0531_70-50_200s.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    ### 63-50_100s
    def train_vqgan_50_100s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_100s/vqgan_pl_fh_epoch50_bs32_0531_70-50_100s.pt")
        )


    def train_vqgan2_50_rgbd_100s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_100s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_100s.pt")  
        )

    def train_transformers_50_100s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_100s/transformer_fh_epoch50_bs32_0531_70-50_100s.pt")
        )

    def generate_crossmodal_63_50_100s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            all_nmse.append(nmse)
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_100s/fh_{i}_epoch50_0531_70-50_100s.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    ### 63-50_50s
    def train_vqgan_50_50s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_50s/vqgan_pl_fh_epoch50_bs32_0531_70-50_50s.pt")
        )


    def train_vqgan2_50_rgbd_50s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_50s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_50s.pt")  
        )

    def train_transformers_50_50s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_50s/transformer_fh_epoch50_bs32_0531_70-50_50s.pt")
        )

    def generate_crossmodal_63_50_50s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_50s/fh_{i}_epoch50_0531_70-50_50s.jpg"), merged)

            i=i+1
            if i==n_images:
                break

    ### 63-50_40s
    def train_vqgan_50_40s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_40s/vqgan_pl_fh_epoch50_bs32_0531_70-50_40s.pt")
        )


    def train_vqgan2_50_rgbd_40s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_40s/vqgan2_RGB_D_fh_epoch50_bs32_0531_70-50_40s.pt")  
        )

    def train_transformers_50_40s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_40s/transformer_fh_epoch50_bs32_0531_70-50_40s.pt")
        )

    def generate_crossmodal_63_50_40s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            all_nmse.append(nmse)
            print(f"NMSE: {nmse}")
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_40s/fh_{i}_epoch50_0531_70-50_40s.jpg"), merged)

            i=i+1
            if i==n_images:
                break

    ### 63-50_30s
    def train_vqgan_50_30s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_30s/vqgan_pl_fh_epoch50_bs32_0514_70-50_30s.pt")
        )


    def train_vqgan2_50_rgbd_30s(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_30s/vqgan2_RGB_D_fh_epoch50_bs32_0514_70-50_30s.pt")  
        )

    def train_transformers_50_30s(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "fh_checkpoints", "70-50_30s/transformer_fh_epoch50_bs32_0514_70-50_30s.pt")
        )

    def generate_crossmodal_63_50_30s(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            all_nmse.append(nmse)
            print(f"NMSE: {nmse}")
            print(f"average NMSE: {np.mean(all_nmse)}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_50_30s/fh_{i}_epoch50_0531_70-50_30s.jpg"), merged)

            i=i+1
            if i==n_images:
                break

    def train_vqgan_50(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "50_full/vqgan_pl_epoch50_bs32_0531_50m.pt")
        )
    def train_vqgan2_50(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "50_full/vqgan2_RGB_epoch1_bs8_0531_50m.pt")
        )

    def train_vqgan2_50_rgbd(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "50_full/vqgan2_RGB_D_epoch50_bs32_0531_50m.pt")  
        )

    def train_transformers_50(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "50_full/transformer_epoch50_bs32_0531_50m.pt")
        )

    


    





    def train_vqgan_63(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "63_full/vqgan_pl_epoch50_bs32_0531_63m.pt")
        )
    def train_vqgan2_63(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "vqgan2_RGB_wt_epoch50_bs8_0512_63mtest.pt")
        )


    def train_vqgan2_63_rgbd(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "63_full/vqgan2_RGB_D_epoch50_bs32_0531_63m.pt")  
        )

    def train_transformers_63(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "63_full/transformer_epoch50_bs32_0531_63m.pt")
        )

    def train_transformers_63_28(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "63_28_full/transformer_epoch50_bs32_0703_63_28.pt")
        )


    def train_vqgan_70(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "70_full/vqgan_pl_epoch50_bs32_0531_70m.pt")
        )


    def train_vqgan2_70_rgbd(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "70_full/vqgan2_RGB_D_epoch50_bs32_0531_70m.pt")  
        )

    def train_vqgan2_70(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs8_0417_70m.pt")
        )

    def train_transformers_70(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "wt_checkpoints", "70_full/transformer_epoch50_bs32_0531_70m.pt")
        )





    def train_vqgan_80(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs8_0417_80m.pt")
        )
    def train_vqgan2_80(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs8_0417_80m.pt")
        )

    def train_transformers_80(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs8_0417_80m.pt")
        )




    def train_vqgan_506370(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs2_0417_50+63+70m.pt")
        )
    def train_vqgan2_506370(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs2_0417_50+63+70m.pt")
        )

    def train_transformers_506370(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs2_0417_50+63+70m.pt")
        )




    def train_vqgan_506380(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs2_0417_50+63+80m.pt")
        )
    def train_vqgan2_506380(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs2_0417_50+63+80m.pt")
        )

    def train_transformers_506380(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs2_0417_50+63+80m.pt")
        )




    def train_vqgan_507080(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs2_0417_50+70+80m.pt")
        )
    def train_vqgan2_507080(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs2_0417_50+70+80m.pt")
        )

    def train_transformers_507080(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs2_0417_50+70+80m.pt")
        )





    def train_vqgan_637080(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs2_0417_63+70+80m.pt")
        )
    def train_vqgan2_637080(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs2_0417_63+70+80m.pt")
        )

    def train_transformers_637080(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs2_0417_63+70+80m.pt")
        )


    def train_vqgan_all_height(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs2_0418_allheight.pt")
        )
    def train_vqgan2_all_height(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs2_0418_allheight.pt")
        )

    def train_transformers_all_height(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs2_0418_allheight.pt")
        )


    def train_vqgan_6380(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):
        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            expriment_save_name="reconstruction_pl_epoch1_bs2_0316.gif",
            **self.config["vqgan_pl"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan_pl_epoch60_bs8_0417_63+80m.pt")
        )
    def train_vqgan2_6380(self, dataloader: torch.utils.data.DataLoader, epochs: int = 1):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan2.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model=self.vqgan2,
            run=self.run,
            device=self.device,
            expriment_save_name = "reconstruction_RGB_epoch1_bs2_0316.gif",
            **self.config["vqgan_RGB"],
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan2.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan2_RGB_epoch60_bs8_0417_63+80m.pt")
        )

    def train_transformers_6380(
            self, dataloader: torch.utils.data.DataLoader, epochs: int = 1
    ):

        print(f"[INFO] Training Transformer on {self.device} for {epochs} epoch(s).")

        self.vqgan.eval()
        self.transformer = self.transformer.to(self.device)

        self.transformer_trainer = TransformerTrainer(
            model=self.transformer,
            run=self.run,
            device=self.device,
            experiment_dir=self.experiment_dir,
            **self.config["transformer"],
        )

        self.transformer_trainer.train(dataloader=dataloader, epochs=epochs)

        self.transformer.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "transformer_epoch60_bs8_0417_63+80m.pt")
        )

    def generate_images(self, n_images: int = 5):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.transformer = self.transformer.to(self.device)


        for i in range(n_images):
            start_indices = torch.zeros((4, 0)).long().to(self.device)
            sos_tokens = torch.ones(start_indices.shape[0], 1) * 0

            sos_tokens = sos_tokens.long().to(self.device)
            sample_indices = self.transformer.sample(
                start_indices, sos_tokens, steps=256
            )

            sampled_imgs = self.transformer.z_to_image(sample_indices)
            torchvision.utils.save_image(
                sampled_imgs,
                os.path.join(self.experiment_dir, f"generated_{i}_epoch60_0417_63+70m.jpg"),
                nrow=4,
            )

    def generate_crossmodal_50_50(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            all_nmse.append(nmse)
            avg_nmse = np.mean(all_nmse)
            print(f"NMSE: {nmse}")
            print(f"average NMSE: {avg_nmse}")
            
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/50_full/bs32_{i}_epoch50_0531_50_50m.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    def generate_crossmodal_63_63(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            all_nmse.append(nmse)
            avg_nmse = np.mean(all_nmse)
            print(f"average NMSE: {avg_nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/63_full/bs32_{i}_epoch50_0531_63_63m.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    def generate_crossmodal_63_50(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGB)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            #RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0513/63_50_500s/wt_{i}_epoch50_0513_63-50_500s.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    def generate_crossmodal_70_70(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        all_nmse = []
        for index, imgs in enumerate(dataloader):

            RGBD = Variable(imgs[0]).to(device=self.device)
            RGB = RGBD[:, :3, ...]
            depth = RGBD[:, 3:, ...]
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGBD)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            depth_image_3c = depth.repeat(1, 3, 1, 1)
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            all_nmse.append(nmse)
            avg_nmse = np.mean(all_nmse)
            print(f"average NMSE: {avg_nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            depth_image_3c,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0531/70_full/bs32_{i}_epoch50_0531_70_70m.jpg"), merged)

            i=i+1
            if i==n_images:
                break




    def generate_crossmodal_80_80(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGB)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0417/generated_{i}_epoch60_0417_80-80m.jpg"), merged)

            i=i+1
            if i==n_images:
                break

    def generate_crossmodal_506370_80(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGB)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0417/generated_{i}_epoch60_0417_506370-80m.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    def generate_crossmodal_506380_70(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGB)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0417/generated_{i}_epoch60_0417_506380-70m.jpg"), merged)

            i=i+1
            if i==n_images:
                break

    def generate_crossmodal_507080_63(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGB)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            print(f"NMSE: {nmse}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0417/generated_{i}_epoch60_0417_507080-63m.jpg"), merged)

            i=i+1
            if i==n_images:
                break


    def generate_crossmodal_637080_50(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        total_nmse = 0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)
            logits, target = self.transformer(pl,RGB)
            probs = F.softmax(logits, dim=-1)
            ix = torch.argmax(probs, dim=-1)
            sampled_imgs = self.transformer.z_to_image(ix)
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            tru = torch.sum(pl_va ** 2)
            nmse = error / tru
            total_nmse = total_nmse + nmse
            print(f"NMSE: {nmse}")
            if i == n_images - 1:
                print(f"Average NMSE: {total_nmse / n_images}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0417/generated_{i}_epoch60_0417_637080-50m.jpg"), merged)

            i=i+1
            if i==n_images:
                break




    def generate_crossmodal(self, dataloader: torch.utils.data.DataLoader, n_images: int = 20, step: int = 256):

        print(f"[INFO] Generating {n_images} images...")

        self.vqgan.to(self.device)
        self.vqgan2.to(self.device)
        self.transformer = self.transformer.to(self.device)

        i=0
        total_nmse = 0
        for index, imgs in enumerate(dataloader):

            RGB = Variable(imgs[0]).to(device=self.device)
            pl = Variable(imgs[1]).to(device=self.device)


            # RGB_indices=self.transformer.encode_to_z2(RGB)
            #print('RGB',RGB.shape)
            #print('pl', pl.shape)
            #torch.cuda.synchronize()
            #start_time = time.time()
            logits, target = self.transformer(pl,RGB)
            #torch.cuda.synchronize()
            #end_time = time.time()
            #inference_duration = end_time - start_time
            #print(f"vqgan Inference time: {inference_duration:.6f} seconds")

            #print('logits',logits.shape) #[1, 64, 512]
            probs = F.softmax(logits, dim=-1)
            #print('probs', probs.shape) #[1, 64, 512]

            ix = torch.argmax(probs, dim=-1)
            #print('ix', ix.shape) #[1, 64]

            # ix = torch.multinomial(  # 按概率采样一个 token
            #     probs, num_samples=64
            # )  # Note : not sure what's happening here
            #torch.cuda.synchronize()
            # 计算推理时间
            # start_time = time.time()
            print(ix.shape) # [1, 64]
            sampled_imgs = self.transformer.z_to_image(ix)
            #print('sampled_imgs', sampled_imgs.shape) #[1, 1, 128, 128]
            #torch.cuda.synchronize()
            #end_time = time.time()
            #inference_duration = end_time - start_time
            #print(f"plvqgan Inference time: {inference_duration:.6f} seconds")
            pl_image_3c = pl.repeat(1, 3, 1, 1)  # 形状变为[1, 3, H, W]
            sample_image_3c = sampled_imgs.repeat(1, 3, 1, 1)
            pl_va  = (pl - pl.min()) * (255 / (pl.max() - pl.min()))
            pl_va = pl_va.cpu()
            sampled_imgs_va =(sampled_imgs - sampled_imgs.min()) * (255 / (sampled_imgs.max() - sampled_imgs.min()))
            sampled_imgs_va = sampled_imgs_va.cpu()
            #print(pl_va)
            '''error = np.sum((pl_va - sampled_imgs_va) ** 2)
            print(error)
            normB = np.linalg.norm(pl_va.cpu().numpy().flatten())
            print('normB',normB)
            nmse_value = np.linalg.norm(error.cpu().numpy().flatten()) / normB
            print(f"NMSE: {nmse_value}")'''
            #print(pl_va)
            #print(sampled_imgs_va)
            #print((pl_va - sampled_imgs_va)**2)
            error = torch.sum((pl_va - sampled_imgs_va) ** 2)
            #print('error',error)
            tru = torch.sum(pl_va ** 2)
            #print('tru',tru)
            nmse = error / tru
            total_nmse = total_nmse + nmse
            print(f"NMSE: {nmse}")
            if i==n_images-1:
                print(f"Average NMSE: {total_nmse/n_images}")
            merged = (
                torchvision.utils.make_grid(
                    torch.cat(
                        (
                            RGB,
                            sample_image_3c,
                            pl_image_3c,
                        ),
                    )
                )
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            merged = (merged - merged.min()) * (
                    255 / (merged.max() - merged.min())
            )
            merged = merged.astype(np.uint8)
            imageio.imsave(os.path.join(self.experiment_dir, f"0716/generated_{i}_epoch60_70_28.jpg"), merged)

            i=i+1
            if i==n_images:
                break