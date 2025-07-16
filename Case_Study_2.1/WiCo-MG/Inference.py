import argparse
import yaml

from dataloader import load_dataloader
from trainer import Trainer
from transformer import VQGANTransformer
from vqgan import VQGAN



def main(args, config):

    vqgan_RGB = VQGAN(**config["architecture"]["vqgan_RGB"])  # 从配置文件中初始化VQGAN模型
    vqgan_CIR = VQGAN(**config["architecture"]["vqgan_CIR"])  # 从配置文件中初始化VQGAN模型
    vqgan_CIR.load_checkpoint("./experiments/checkpoints/vqgan.pt", device=args.device)
    vqgan_RGB.load_checkpoint("./experiments/checkpoints/vqgan2.pt", device=args.device)

    transformer = VQGANTransformer(  # 初始化VQGAN-Transformer联合模型
        vqgan=vqgan_CIR,
        vqgan2=vqgan_RGB, # 传入已创建的VQGAN实例
        **config["architecture"]["transformer"],  # 从配置读取transformer参数
        device=args.device  # 设置运行设备
    )
    transformer.load_checkpoint("./experiments/checkpoints/transformer.pt", device=args.device)

    dataloader_generate = load_dataloader(name=args.dataset_name, batch_size=1,
                                 image_size=config["architecture"]["vqgan_CIR"]["img_size"], num_files=1) 

    experiment_dir= "experiments"
    trainer = Trainer(  # 创建训练管理对象
        vqgan_CIR, vqgan_RGB, transformer,  # 传入模型
        run=None,  # 绑定实验记录
        config=config["trainer"],  # 读取训练相关配置
        seed=args.seed,  # 设置随机种子
        device=args.device,  # 指定运行设备
        experiment_dir= experiment_dir
    )

    trainer.generate_angle(dataloader_generate,n_images=1,latent_channels=512)  # 生成最终图像


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(  # 添加配置文件路径参数
        "--config_path", type=str,
        default="configs/default_crossmodal.yml",
        help="path to config file"
    )
    parser.add_argument(  # 添加数据集选择参数
        "--dataset_name", type=str,
        choices=["mnist", "cifar", "custom", "RGB2CIR", "RGB2power", "RGB2dodtheta", "RGB2dodphi", "RGB2delay"],
        default="RGB2DoA",
        help="Dataset for the model"
    )
    parser.add_argument(  # 添加设备选择参数
        "--device", type=str, default="cuda:2",
        choices=["cpu", "cuda"],
        help="Device to train the model on"
    )
    parser.add_argument(  # 添加随机种子参数
        "--seed", type=str, default=42,
        help="Seed for Reproducibility"
    )
    parser.add_argument(  # 添加设备选择参数
        "--batch_size", type=int, default=16,
    )
    parser.add_argument(  # 添加设备选择参数
        "--epochs_vqgan", type=int, default=50,
    )
    parser.add_argument(  # 添加设备选择参数
        "--epochs_transformer", type=int, default=200,
    )
    args = parser.parse_args()  # 解析命令行参数

    # 7. 配置文件加载
    with open(args.config_path) as f:  # 打开配置文件
        config = yaml.load(f, Loader=yaml.FullLoader)  # 用yaml加载配置

    main(args, config)  # 执行主函数