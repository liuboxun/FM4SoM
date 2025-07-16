# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.mingpt import GPT
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import time


class MyGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('/home/smr/smr_base_model/LLM4PP/gpt2', output_attentions=True,
                                  output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:6]

        self.in_layer = nn.Linear(64, 768)
        self.relu = nn.ReLU()
        self.in_layer2 = nn.Linear(768, 768)
        
        self.out_layer = nn.Linear(768, 131072)

    
    def model_train(self):
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.train()

    def model_eval(self):
        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.eval()

    def forward(self, x):
        x = self.in_layer(x)
        x = self.relu(x)
        x = self.in_layer2(x)
        x = self.gpt2(inputs_embeds=x).last_hidden_state
        x = self.out_layer(x)

        x = x.reshape(-1, 64, 2048)
        return x, None

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        """Loads the checkpoint from the given path."""

        self.load_state_dict(torch.load(path))