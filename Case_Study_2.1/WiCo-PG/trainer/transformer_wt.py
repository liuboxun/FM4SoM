# Importing Libraries
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Run, Image
from torch.autograd import Variable
import time

class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        run: Run,
        experiment_dir: str = "experiments",
        device: str = "cuda",
        learning_rate: float = 4.5e-06,
        beta1: float = 0.9,
        beta2: float = 0.95,
    ):
        self.run = run
        self.experiment_dir = experiment_dir

        self.model = model
        self.device = device
        self.optim = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

    def configure_optimizers( #自定义哪些参数需要权重衰减（weight decay），哪些不需要。它的核心思想是针对不同类型的层（例如 nn.Linear、nn.LayerNorm、nn.Embedding），分别应用不同的 weight decay 规则，然后使用 AdamW 作为优化器
        self, learning_rate: float = 4.5e-06, beta1: float = 0.9, beta2: float = 0.95
    ):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        # Enabling weight decay to only certain layers
        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(beta1, beta2)
        )
        return optimizer


    def train(self, dataloader: torch.utils.data.DataLoader, epochs: int):
        for epoch in range(epochs):

            for index, imgs in enumerate(dataloader):
                self.optim.zero_grad()
                RGB = Variable(imgs[0]).to(device=self.device)
                pl = Variable(imgs[1]).to(device=self.device)
                #print('RGB',RGB.shape) #[4,3,128,128]
                #print('pl', pl.shape) #[4,1,128,128]
                logits, targets = self.model(pl, RGB) # 模型前向传播获取输出
                # print('logits',logits.shape) #[bs,64,2048]
                # print('targets',targets.shape) #[bs,64]
                # print("logits.type()",logits.type())
                # print("targets.type()",targets.type())
                #print('logits, targets',logits.shape, targets.shape)    #torch.Size([16, 64, 512]) torch.Size([16, 64])
                #print('logits, targets', logits.reshape(-1, logits.size(-1)).shape, targets.reshape(-1).shape)    #torch.Size([1024, 512]) torch.Size([1024])
                # 计算训练时间
                #start_time = time.time()
                loss = F.cross_entropy( # 计算交叉熵损失（展平处理适配序列任务）
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                loss.backward()
                self.optim.step()
                #end_time = time.time()
                #training_duration = end_time - start_time
                #print(f"Training time: {training_duration:.6f} seconds")
                self.run.track( # 记录损失指标（使用实验跟踪工具）
                    loss,
                    name="Cross Entropy Loss",
                    step=index,
                    context={"stage": "transformer"},
                )

                if index % 10 == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | Cross Entropy Loss : {loss:.4f}"
                    )

                    '''_, sampled_imgs = self.model.log_images(RGB[0][None]) # 生成并记录样本图像（用于可视化监控）

                    self.run.track(
                        Image( # 将张量转换为PIL兼容格式
                            torchvision.utils.make_grid(sampled_imgs)
                            .mul(255) # 反归一化 [0,1] -> [0,255]
                            .add_(0.5) # 数值修正（四舍五入准备）
                            .clamp_(0, 255) # 确保像素值合法
                            .permute(1, 2, 0) # 维度转换 (C,H,W) -> (H,W,C)
                            .to("cpu", torch.uint8)
                            .numpy()
                        ),
                        name="Transformer Images",
                        step=index,
                        context={"stage": "transformer"},
                    )'''