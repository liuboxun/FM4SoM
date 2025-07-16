# Importing Libraries
# Importing Libraries
import argparse

import yaml
from aim import Run

from dataloader import load_dataloader
# from sentry_sdk.utils import epoch
from trainer import Trainer
from transformer import VQGANTransformer
from vqgan import VQGAN
import torch


def main(args, config):
    vqgan_RGB = VQGAN(**config["architecture"]["vqgan_RGB"])
    # print(vqgan_RGB)
    vqgan_pl = VQGAN(**config["architecture"]["vqgan_pl"])
    # print(vqgan_pl)
    vqgan_RGB.load_state_dict(
        torch.load(f'/home/smr/smr_base_model/VQGAN/experiments/checkpoints/vqgan2_RGB_epoch60_bs8_0417_70m.pt'))
    vqgan_pl.load_state_dict(
        torch.load(f'/home/smr/smr_base_model/VQGAN/experiments/checkpoints/vqgan_pl_epoch60_bs8_0417_70m.pt'))

    transformer = VQGANTransformer(
        vqgan=vqgan_pl,
        vqgan2=vqgan_RGB,
        **config["architecture"]["transformer"],
        device=args.device
    )
    transformer.load_state_dict(
        torch.load(f'/home/smr/smr_base_model/VQGAN/experiments/checkpoints/transformer_epoch60_bs8_0417_70m.pt'))
    dataloader_generate = load_dataloader(name="RGB2PL_70_kk", batch_size=1, image_size=128)
    trainer = Trainer(
        vqgan_pl,
        vqgan_RGB,
        transformer,
        run=None,
        config=config["trainer"],
        seed=args.seed,
        device=args.device,
    )

    trainer.generate_crossmodal(dataloader_generate, step=32)  # 生成最终图像


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default_crossmodal.yml",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["mnist", "cifar", "custom", "RGB2PL_70_15ghz_test"],
        default="RGB2PL_70_15ghz_test",
        help="Dataset for the model",
    )
    parser.add_argument(  # 添加设备选择参数
        "--device", type=str, default="cuda:1",
        choices=["cpu", "cuda"],
        help="Device to train the model on"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Seed for Reproducibility",
    )

    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)
