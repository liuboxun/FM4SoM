import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.mingpt import GPT


class CrossModalGPT(GPT):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__(vocab_size, block_size, n_layer, n_head, n_embd)
        self.cross_attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head)  # 跨模态注意力

    def forward(self, x, context=None):
        x = self.tok_emb(x)  # CIR token embedding
        if context is not None:
            context = self.tok_emb(context)  # RGB token embedding
            x, _ = self.cross_attn(x, context, context)  # CIR 关注 RGB
        return super().forward(x)

class VQGANTransformer(nn.Module):
    def __init__(
        self,
        vqgan: nn.Module, #CIR
        vqgan2: nn.Module, #RGB
        device: str = "cuda",
        sos_token: int = 0,
        pkeep: float = 0.5,
        block_size: int = 512,
        n_layer: int = 12,
        n_head: int = 16,
        n_embd: int = 1024,
    ):
        super().__init__()

        self.sos_token = sos_token
        self.device = device

        self.vqgan = vqgan #CIR
        self.vqgan2 = vqgan2 #RGB

        self.n_embd = n_embd

        self.transformer = GPT(
            vocab_size=self.vqgan.num_codebook_vectors,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )

        self.pkeep = pkeep

    @torch.no_grad()
    def encode_to_z(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_z2(self, x: torch.tensor) -> torch.tensor:

        quant_z, indices, _ = self.vqgan2.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(
        self, indices: torch.tensor, p1: int = 8, p2: int = 8, latent_channels: int = 512
    ) -> torch.Tensor:

        ix_to_vectors = self.vqgan.codebook.codebook(indices).reshape(
            indices.shape[0], p1, p2, latent_channels
        )

        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:

        _, indices = self.encode_to_z(x) 

        _, indices_y = self.encode_to_z2(y)  

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token 
        sos_tokens = sos_tokens.long().to(self.device)

        mask = torch.bernoulli( 
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )  
        mask = mask.round().to(dtype=torch.int64)

        random_indices = torch.randint_like( 
            indices, high=self.transformer.config.vocab_size
        ) 


        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1) 

        target = indices
        logits, _ = self.transformer(indices_y)  
        return logits, target 

   
    def top_k_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        
        v, ix = torch.topk(logits, k) #取 logits 中前 k 大的值 v 及其索引 ix。
        out = logits.clone()
        out[out < v[..., [-1]]] = -float( 
            "inf"
        )  
        return out

    @torch.no_grad()
    def sample( 
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        steps: int = 256,
        temperature: float = 1.0,
        top_k: int = 100,
    ) -> torch.Tensor:

        self.transformer.eval()
        
        x = torch.cat((c, x), dim=1)  

        for k in range(steps):
            logits, _ = self.transformer(x)  

            logits = (
                logits[:, -1, :] / temperature
            )  

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k) 
            
            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(
                probs, num_samples=1
            )  

            x = torch.cat((x, ix), dim=1) 

        x = x[:, c.shape[1] :] 

        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x:torch.Tensor):

        log = dict()

        _, indices = self.encode_to_z(x) # Getting the indices of the quantized encoding
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token # 创建序列起始符
        sos_tokens = sos_tokens.long().to(self.device) # 转换为整数类型并送设备

        start_indices = indices[:, : indices.shape[1] // 2] # 取前50%潜在编码作为条件
        sample_indices = self.sample( # 自回归生成后半段编码
            start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1]
        )
        
        half_sample = self.z_to_image(sample_indices, latent_channels=self.n_embd) # 解码为图像

        start_indices = indices[:, :0] # 空条件（生成完全自主）
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices, latent_channels=self.n_embd) # 解码为图像

        x_rec = self.z_to_image(indices, latent_channels=self.n_embd) # 完整编码直接解码（验证重建能力）

        log["input"] = x 
        log["rec"] = x_rec 
        log["half_sample"] = half_sample 
        log["full_sample"] = full_sample 

        return log, torch.concat((x, x_rec, half_sample, full_sample)) 


    def load_checkpoint(self, path: str, device: str = "cuda"):
        checkpoint = torch.load(path, map_location=device) 
        self.load_state_dict(checkpoint) 
        print(f"Checkpoint loaded from {path} to {device}")

    def save_checkpoint(self, path):
        """Saves the checkpoint to the given path."""

        torch.save(self.state_dict(), path)
