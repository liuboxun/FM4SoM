"""
   final-version
"""
import sys
import os
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import AutoModel, LlamaForCausalLM, ViTConfig
from src.MLoRA_SoM_open_source.model.modeling_gpt2 import GPT2Model
from src.MLoRA_SoM_open_source.peft import MMOELoraConfig2, get_peft_model
from einops import rearrange
from peft import LoraConfig
# from transformers import LlamaModel
import time
import torchvision.models as models

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Res_block_complex(nn.Module):
    def __init__(self, in_planes):
        super(Res_block_complex, self).__init__()

        self.linear1 = nn.Conv1d(in_planes, in_planes, 3, 1, 1)
        self.linear2 = nn.Conv1d(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        rs1 = self.relu(self.linear1(x))
        rs1 = self.linear2(rs1)
        rs = torch.add(x, rs1)
        return rs


def generate_res_block_complex(in_channel, num_res=1):
    layers = []
    for i in range(num_res):
        layers.append(Res_block_complex(in_planes=in_channel))
        # layers.append(nn.BatchNorm1d(in_channel))
    return nn.Sequential(*layers)


class Muti_task_Adapter(nn.Module):

    def __init__(self, input_lens=16, input_dims=768, output_dims=768, n_adapter=3, task_id=1):
        super(Muti_task_Adapter, self).__init__()
        self.back_bone1 = generate_res_block_complex(input_lens, num_res=n_adapter)
        self.back_bone2 = generate_res_block_complex(input_lens, num_res=n_adapter)
        self.norm = nn.LayerNorm(output_dims)
        self.relu = nn.ReLU()
        self.dim_projection = nn.Sequential(
            # nn.Linear(input_dims, input_dims),
            # nn.ReLU(),
            nn.Linear(input_dims, output_dims),
        )
        # self.dim_projection2 = nn.Linear(768, 768)

    def forward(self, x):
        x = self.dim_projection(x)
        h = self.back_bone1(x)
        out = self.back_bone2(self.relu(h))
        out = self.norm(out)
        # out = out + self.dim_projection2(out)
        return out


class Model(nn.Module):

    def __init__(self, llm_name='gpt2', llm_layers=6, task_num=3, expert_num=8,
                 lora_r=8, prev_len=32, pred_len=32,
                 gpu_id=0, train_stage=1,
                 Nt=8, num_polit=8, K=64, patch_num=4,
                 d_model=512,
                 is_llm_rand_inital=0, is_llm_frozen=0, is_llm_inference=1,
                 adapter_num=[1, 1],
                 stride=1, dropout=0.1, peft='moe', is_sparse=0,
                 task_range=None, is_multi_modality=True,
                 num_beam=128):
        super(Model, self).__init__()
        assert len(task_range) == task_num
        self.device = torch.device('cuda:{}'.format(gpu_id))
        print(self.device)
        numtoken = prev_len
        self.is_multi_modality = is_multi_modality
        self.llm_name = llm_name
        self.task_num = task_num
        self.expert_num = expert_num
        self.peft = peft
        self.lora_r = lora_r
        self.adapter_num = adapter_num
        self.dropout = dropout

        self.Nt = Nt
        self.prev_len = prev_len
        self.pred_len = pred_len
        self.num_polit = num_polit
        self.num_data = K - num_polit
        self.K = K  # num of subcarrier
        self.d_model = d_model
        self.num_beam = num_beam  # num of DFT codebook

        self.patch_num = patch_num
        self.stride = stride
        self.num_token = numtoken
        self.tau = K // num_polit
        self.is_moe = 0
        self.is_llm_rand_inital = is_llm_rand_inital
        self.is_llm_frozen = is_llm_frozen
        self.is_sparse = is_sparse
        self.is_llm_inference = is_llm_inference
        # Config
        if train_stage == 1:
            self.is_llm_frozen = 1
        self.config = {
            'train_stage': train_stage,
            'task_num': self.task_num,
            'llm_name': self.llm_name,
            'is_llm_inference': self.is_llm_inference,
            'is_llm_rand_inital': self.is_llm_rand_inital,
            'is_llm_frozen': self.is_llm_frozen,
            'peft': self.peft,
            'is_sparse': self.is_sparse,
            'lora_r': self.lora_r,
            'expert_num': self.expert_num,
            'adapter_num': self.adapter_num
        }

        self.input_dim = K * 2
        # 1.Preprocess
        self.activate = nn.ReLU()
        self.Linear_t1 = nn.Linear(num_polit * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.Linear_t2 = nn.Linear(prev_len * 2, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.Linear_t3 = nn.Linear(K * 2, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.len_token = Nt + 1 if is_multi_modality else Nt

        # 2.Muti-task-adapter

        # 3.LLM
        if self.llm_name == 'gpt2':
            self.llm = GPT2Model.from_pretrained('./Weights/gpt2',
                                                 output_attentions=True, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 768
            self.is_llama = 0

        elif self.llm_name == 'gpt2-medium':
            self.llm = GPT2Model.from_pretrained('xxxxx',
                                                 output_attentions=True, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 1024
            self.is_llama = 0

        elif self.llm_name == 'gpt2-large':
            self.llm = GPT2Model.from_pretrained('xxxxx',
                                                 output_attentions=True, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 1280
            self.is_llama = 0

        elif self.llm_name == 'gpt2-xl':
            self.llm = GPT2Model.from_pretrained('xxxxx',
                                                 output_attentions=True, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 1600
            self.is_llama = 0

        else:
            self.llm = GPT2Model.from_pretrained('./Weights/gpt2',
                                                 output_attentions=True, output_hidden_states=True)
            self.llm.h = self.llm.h[:llm_layers]
            self.hidden_dim_gpt = 768
            self.is_llama = 0

        if self.is_multi_modality:
            self.res18 = models.resnet18(pretrained=True)
            self.res18.eval()
            for param in self.res18.parameters():
                param.requires_grad = False
            self.res18_align_layer_1 = nn.Linear(1000, self.hidden_dim_gpt)
            self.res18_align_layer_2 = nn.Linear(1000, self.hidden_dim_gpt)
            self.res18_align_layer_3 = nn.Linear(1000, self.hidden_dim_gpt)

        # 4.Muti-task-adapter
        self.adapter_in_t1 = Muti_task_Adapter(input_lens=self.Nt, input_dims=d_model, output_dims=self.hidden_dim_gpt,
                                               n_adapter=adapter_num[0])
        self.adapter_in_t2 = Muti_task_Adapter(input_lens=self.Nt, input_dims=d_model, output_dims=self.hidden_dim_gpt,
                                               n_adapter=adapter_num[0])
        self.adapter_in_t3 = Muti_task_Adapter(input_lens=self.Nt, input_dims=d_model, output_dims=self.hidden_dim_gpt,
                                               n_adapter=adapter_num[0])

        self.adapter_out_t1 = Muti_task_Adapter(input_lens=self.Nt, input_dims=self.hidden_dim_gpt, output_dims=K * 2,
                                                n_adapter=adapter_num[1])
        self.adapter_out_t2 = Muti_task_Adapter(input_lens=self.Nt, input_dims=self.hidden_dim_gpt,
                                                output_dims=pred_len * 2,
                                                n_adapter=adapter_num[1])
        self.adapter_out_t3 = Muti_task_Adapter(input_lens=self.Nt, input_dims=self.hidden_dim_gpt,
                                                output_dims=self.hidden_dim_gpt,
                                                n_adapter=adapter_num[1])

        # 5.output-projection
        self.output_d2_t1 = nn.Linear(self.K * 2, self.K * 2)
        self.output_d2_t2 = nn.Linear(self.pred_len * 2, self.pred_len * 2)

        self.output_d2_t3 = nn.Linear(self.hidden_dim_gpt, 1)
        self.output_d1_t3 = nn.Linear(self.Nt, 2)

        # 6. set train parms
        self.stage = train_stage
        self.Set_train_parms(self.config)

    def Set_train_parms(self, config):
        print(f"Training Stage: {config['train_stage']}")
        print(f"Adapter Number: {config['adapter_num']}")
        print(f"LLM stage: | is_exist: {config['is_llm_inference']}"
              f" | is_frozen: {config['is_llm_frozen']}"
              f" | is_rand_init: {config['is_llm_rand_inital']}"
              f" | peft: {config['peft']}"
              f" | is_sparse: {config['is_sparse']}"
              f" | lora_r: {config['lora_r']}")
        target_module = ['c_fc', 'c_proj']
        if self.peft == 'moe':
            self.peft_config = MMOELoraConfig2(
                fan_in_fan_out=True,
                task_type="CAUSAL_LM",
                target_modules=target_module,
                inference_mode=False,
                r=self.lora_r, lora_alpha=self.lora_r * 2,
                lora_dropout=self.dropout,
                modules_to_save=[],
                task_num=self.task_num,
                task_embedding_dim=256,
                expert_num=self.expert_num,
                is_sparse=config['is_sparse']
            )
        else:
            self.peft_config = LoraConfig(
                r=self.lora_r // self.expert_num,  # the dimension of the low-rank matrices
                lora_alpha=self.lora_r // self.expert_num * 2,  # scaling factor for the weight matrices
                lora_dropout=self.dropout,  # dropout probability of the LoRA layers
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_module,
                fan_in_fan_out=True
            )
        print(target_module)
        if config['is_llm_rand_inital']:
            for name, param in self.llm.named_parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.1)

        if config['is_llm_frozen'] == 1:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                param.requires_grad = False
        else:
            if config['train_stage'] == 1:
                for i, (name, param) in enumerate(self.llm.named_parameters()):
                    param.requires_grad = False
            elif config['train_stage'] == 2:
                if self.peft == 'moe':
                    self.is_moe = 1
                self.llm = get_peft_model(self.llm, self.peft_config)

    def llm_forward(self, x, task_id=1):
        if self.is_moe:
            outputs = self.llm(inputs_embeds=x, task_id=task_id).last_hidden_state
        else:
            outputs = self.llm(inputs_embeds=x).last_hidden_state
        return outputs

    def forward(self, x1=None, x2=None, x3=None, rgb=None, task_range=None):
        if task_range is None:
            task_range = [1, 2, 3]
        if self.is_multi_modality:
            rgb_feature = self.res18(rgb)
        if 1 in task_range:
            # T1, b, 8, 8, 2  -> b, 8, 64
            B, N, K0, _ = x1.shape
            x1 = rearrange(x1, 'b n k o -> b n (k o)')
            x1 = self.Linear_t1(x1)
            x1 = self.adapter_in_t1(x1)
            if self.is_multi_modality:
                rgb_feature_t1 = self.res18_align_layer_1(rgb_feature).unsqueeze(1)
                x1 = torch.cat([rgb_feature_t1, x1], dim=1)
            h1 = self.llm_forward(x1, task_id=1)
            h1 = h1[:, -self.Nt:, :]
            y1 = self.adapter_out_t1(h1)
            y1 = rearrange(y1, 'b n (k o)-> b n k o', o=2)
        else:
            y1 = None

        if 2 in task_range:
            # T2, b, 8, 32, 2  -> b, 8, 32
            B, N, K1, _ = x2.shape
            x2 = rearrange(x2, 'b n k o -> b n (k o)')
            x2 = self.Linear_t2(x2)
            x2 = self.adapter_in_t2(x2)
            if self.is_multi_modality:
                rgb_feature_t2 = self.res18_align_layer_2(rgb_feature).unsqueeze(1)
                x2 = torch.cat([rgb_feature_t2, x2], dim=1)
            h2 = self.llm_forward(x2, task_id=2)
            h2 = h2[:, -self.Nt:, :]
            y2 = self.adapter_out_t2(h2)
            y2 = rearrange(y2, 'b n (k o)-> b n k o', o=2)
        else:
            y2 = None

        if 3 in task_range:
            # T3, b, 8, 64, 2  -> b, 2
            B, N, K2, _ = x3.shape
            x3 = rearrange(x3, 'b n k o -> b n (k o)')
            x3 = self.Linear_t3(x3)
            x3 = self.adapter_in_t3(x3)
            if self.is_multi_modality:
                rgb_feature_t3 = self.res18_align_layer_3(rgb_feature).unsqueeze(1)
                x3 = torch.cat([rgb_feature_t3, x3], dim=1)
            h3 = self.llm_forward(x3, task_id=3)
            h3 = h3[:, -self.Nt:, :]
            y3 = self.adapter_out_t3(h3)
            y3 = self.output_d2_t3(self.activate(self.output_d1_t3(y3.permute(0, 2, 1)).permute(0, 2, 1)))
            y3 = y3.squeeze(2)
        else:
            y3 = None

        return y1, y2, y3


if __name__ == '__main__':
    import torch

    device = torch.device('cuda:3')
    task_range = [1, 2, 3]

    inputs_1 = torch.rand(2, 8, 8, 2).to(device)
    inputs_2 = torch.rand(2, 8, 32, 2).to(device)
    inputs_3 = torch.rand(2, 8, 64, 2).to(device)
    inputs_rgb = torch.rand(2, 3, 224, 224).to(device)

    model = Model(gpu_id=3, task_num=len(task_range), lora_r=8, expert_num=8,
                  K=64, num_polit=8, Nt=8, num_beam=128, prev_len=32, pred_len=32,
                  peft='moe', train_stage=1, llm_name='gpt2',
                  is_llm_inference=1, is_llm_frozen=0, is_sparse=0, is_llm_rand_inital=0,
                  task_range=task_range, adapter_num=[1, 1]).to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (total / 1e6))
    total_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of learnable parameter: %.5fM" % (total_learn / 1e6))

    out = model(inputs_1, inputs_2, inputs_3, inputs_rgb, task_range=task_range)
    for i in task_range:
        print(i, out[i - 1].shape)

    model.stage = 2
    model.Set_train_parms({**model.config, 'train_stage': 2})
    out = model(inputs_1, inputs_2, inputs_3, inputs_rgb, task_range=task_range)
    for i in task_range:
        print(i, out[i - 1].shape)
