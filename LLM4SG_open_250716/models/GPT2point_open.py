import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config



class GPT4SCAgrid(nn.Module):
    def __init__(self, configs, device, normal_channel=False):
        super(GPT4SCAgrid, self).__init__()

        self.fc1 = nn.Linear(1, 64)  # 从1维输入到64维
        self.bn11 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)  # 从64维到128维
        self.bn12 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)  # 从128维到256维
        self.bn13 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 768)  # 从128维到256维
        self.bn14 = nn.BatchNorm1d(768)

        num_patches = 100
        embed_dim = 768
        self.conv_patch = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=8, stride=8)
        self.bn_patch = nn.BatchNorm2d(embed_dim)
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # [1, 100, 768]

        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len + 256 - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        #        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1


        if configs.pretrain:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                    output_hidden_states=True)  # loads a pretrained GPT-2 base model
        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        print("gpt2 = {}".format(self.gpt2))

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        self.fc_freq = nn.Linear(101, 100)

        self.conv1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(1)

        #        各层转移到device上
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    #    def forward(self, x, itr):
    def forward(self, f, x):

        torch.set_printoptions(threshold=torch.inf)

        freq = self.fc1(f)
        freq = self.bn11(freq)
        freq = nn.functional.relu(freq)
        freq = self.fc2(freq)
        freq = nn.functional.relu(self.bn12(freq))
        freq = self.fc3(freq)
        freq = nn.functional.relu(self.bn13(freq))
        freq = self.fc4(freq)
        freq = nn.functional.relu(self.bn14(freq))

        x = self.conv_patch(x)
        x = self.bn_patch(x)
        x = nn.functional.relu(x)

        outputs = x.flatten(2).transpose(1, 2)
        outputs = outputs + self.position_encoding  # 加上位置编码

        batch_size, patches, patch_size = outputs.size()

        # 在第二维拼接
        freq = freq.unsqueeze(1)
        outputs = torch.cat((outputs, freq), dim=1)

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = outputs.permute(0, 2, 1)  # 重新排列维度，从 [B, 101, 768] 到 [B, 768, 101]
        outputs = self.fc_freq(outputs)
        outputs = outputs.permute(0, 2, 1)  # 再次排列维度，从 [B, 768, 100] 到 [B, 100, 768]


        # 卷积层实现
        outputs = outputs.view(batch_size, 10, 10, 768)
        outputs = outputs.permute(0, 3, 1, 2)  # 变成 (B, 768, 10, 10)

        outputs = nn.functional.leaky_relu(self.bn1(self.conv1(outputs)))
        outputs = nn.functional.leaky_relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))

        outputs = outputs.squeeze(1)

        return outputs






