import torch
import torch.nn as nn


class CodeBook(nn.Module):

    def __init__(
        self, num_codebook_vectors: int = 1024, latent_dim: int = 256, beta: int = 0.25
    ):
        super().__init__()

        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim #latent_dim=latent_channels
        self.beta = beta

        # creating the codebook, nn.Embedding here is simply a 2D array mainly for storing our embeddings, it's also learnable
        self.codebook = nn.Embedding(num_codebook_vectors, latent_dim)

        # Initializing the weights in codebook in uniform distribution
        self.codebook.weight.data.uniform_(
            -1 / num_codebook_vectors, 1 / num_codebook_vectors
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        # Channel to last dimension and copying the tensor to store it in a contiguous ( in a sequence ) way
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(
            -1, self.latent_dim
        )  # b*h*w * latent_dim, will look similar to codebook in fig 2 of the paper
        # calculating the distance between the z to the vectors in flattened codebook, from eq. 2
        # (a - b)^2 = a^2 + b^2 - 2ab
        distance = (
            torch.sum(
                z_flattened**2, dim=1, keepdim=True
            )  # keepdim = True to keep the same original shape after the sum
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2
            * torch.matmul(
                z_flattened, self.codebook.weight.t()
            )  # 2*dot(z, codebook.T)
        )

        # getting indices of vectors with minimum distance from the codebook
        min_distance_indices = torch.argmin(distance, dim=1)

        z_q = self.codebook(min_distance_indices).view(z.shape)


        loss = torch.mean(
            (z_q.detach() - z) ** 2 #优化z
            # detach() to avoid calculating gradient while backpropagating
            + self.beta
            * torch.mean(
                (z_q - z.detach()) ** 2 #优化z_q
            )  # commitment loss, detach() to avoid calculating gradient while backpropagating
        )

        # Not sure why we need this, but it's in the original implementation and mentions for "preserving gradients"
        z_q = z + (z_q - z).detach()

        # reshapring to the original shape
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_distance_indices, loss
