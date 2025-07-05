import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Cross-Stitch Layer for task feature fusion
class CrossStitchLayer(nn.Module):
    def __init__(self, num_tasks, in_channels):
        super(CrossStitchLayer, self).__init__()
        alpha_init = torch.full((num_tasks, num_tasks), 0.1 / (num_tasks - 1))
        for i in range(num_tasks):
            alpha_init[i, i] = 0.9
        self.alpha = alpha_init
        self.alpha = torch.tril(self.alpha)
        self.num_tasks = num_tasks
        self.in_channels = in_channels

    def forward(self, inputs):
        assert len(inputs) == self.num_tasks, "Mismatch in the number of inputs and tasks"

        # Initialize the output with zeros
        output = [torch.zeros_like(inputs[0]) for i in range(self.num_tasks)]

        # Apply cross-stitch operation to fuse task-specific features
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                output[i] += self.alpha[i, j] * inputs[j]

        return output


# Small CNN for each task to extract features
class TaskCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(TaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        # self.fc = nn.Linear(out_channels*8, 256)  # Flattened size for output

    def forward(self, x):
        input = x
        x = F.relu(self.conv1(x))
        # x = self.pool(x)
        x = F.relu(self.conv2(x)) + input
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.fc(x)
        return x


class Res_block_2d(nn.Module):
    def __init__(self, in_planes):
        super(Res_block_2d, self).__init__()

        self.linear1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.linear2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        rs1 = self.relu(self.linear1(x))
        rs1 = self.linear2(rs1)
        rs = torch.add(x, rs1)
        return rs


# Multi-task CNN with Cross-Stitch operation
class SingleTaskCNNWithCrossStitch(nn.Module):
    def __init__(self, is_combine=1, num_tasks=6, in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1):
        super(SingleTaskCNNWithCrossStitch, self).__init__()
        self.num_tasks = num_tasks
        self.is_combine = is_combine
        self.task_cnn = nn.ModuleList(
            [TaskCNN(in_channels, out_channels, kernel_size, stride, padding) for _ in range(num_tasks)])

        self.cross_stitch = CrossStitchLayer(num_tasks, out_channels * 2 * 4 * 4)

    def forward(self, inputs):
        task_features = [self.task_cnn[i](inputs[i]) for i in range(self.num_tasks)]
        if self.is_combine:
            task_features = self.cross_stitch(task_features)
        return task_features


# Multi-task CNN with Cross-Stitch operation
class NetworkCrossStitch(nn.Module):
    def __init__(self, is_combine=1, is_mm=True, num_task=3, hidden_channel=16, num_layer=4,
                 prev_len=32, pred_len=32, Nt=16, K=64, num_polit=8, num_beam=128):
        super(NetworkCrossStitch, self).__init__()
        self.hidden_channel = hidden_channel
        self.num_layer = num_layer
        self.num_task = num_task
        self.is_combine = is_combine
        self.prev_len = prev_len
        self.Nt = Nt
        self.num_polit = num_polit
        self.is_mm = is_mm
        self.raw_channel = 3 if self.is_mm else 2

        self.input_aligned = nn.ModuleList([
            nn.Linear(num_polit, K),
            nn.Linear(prev_len, K),
            nn.Linear(K, K),
        ])
        self.channel_up = nn.ModuleList([
            nn.Conv2d(self.raw_channel, hidden_channel, 3, 1, 1),
            nn.Conv2d(self.raw_channel, hidden_channel, 3, 1, 1),
            nn.Conv2d(self.raw_channel, hidden_channel, 3, 1, 1),
        ])

        if self.is_mm:
            self.img_process = nn.Sequential(
                Res_block_2d(3),
                nn.Conv2d(3, 64, 3, 1, 1),
                Res_block_2d(64),
                # nn.Conv2d(64, 128, 3, 1, 1),
                nn.AdaptiveAvgPool2d(1)
            )
            self.img_linear_t1 = nn.Linear(64, Nt * num_polit)
            self.img_linear_t2 = nn.Linear(64, Nt * prev_len)
            self.img_linear_t3 = nn.Linear(64, Nt * K)
        self.activate = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(num_features=hidden_channel)
        self.norm2 = nn.BatchNorm2d(num_features=hidden_channel)
        self.norm3 = nn.BatchNorm2d(num_features=hidden_channel)
        self.layers = nn.ModuleList([SingleTaskCNNWithCrossStitch(is_combine=self.is_combine,
                                                                  num_tasks=num_task,
                                                                  in_channels=hidden_channel,
                                                                  out_channels=hidden_channel)
                                     for i in range(num_layer)])
        self.out2 = nn.Sequential(
            nn.Conv2d(hidden_channel, 2, 3, 1, 1),
            nn.Linear(K, pred_len)
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(hidden_channel, 2, 3, 1, 1),
            nn.Linear(K, K)
        )
        self.out3 = nn.Sequential(
            nn.Linear(hidden_channel * Nt * K, 2)
        )


    def forward(self, x1=None, x2=None, x3=None, img=None, ):
        # Extract individual inputs from the input dictionary
        # img
        if self.is_mm:
            img_feature = self.img_process(img).squeeze()
        # T1
        B, N, K0, _ = x1.shape
        x1 = rearrange(x1, 'b n k o -> b o n k')
        if self.is_mm:
            img_t1 = self.img_linear_t1(img_feature)
            img_t1 = rearrange(img_t1, 'b (o n k) -> b o n k', k=K0, n=N)
            x1 = torch.cat([img_t1, x1], dim=1)
        x1_a = self.activate(self.input_aligned[0](x1))
        x1_a = self.channel_up[0](x1_a)
        # T2
        B, N, K1, _ = x2.shape
        x2 = rearrange(x2, 'b n k o -> b o n k')
        if self.is_mm:
            img_t2 = self.img_linear_t2(img_feature)
            img_t2 = rearrange(img_t2, 'b (o n k) -> b o n k', k=K1, n=N)
            x2 = torch.cat([img_t2, x2], dim=1)
        x2_a = self.activate(self.input_aligned[1](x2))
        x2_a = self.channel_up[1](x2_a)
        # T3
        B, N, K2, _ = x3.shape
        x3 = rearrange(x3, 'b n k o -> b o n k')
        if self.is_mm:
            img_t3 = self.img_linear_t3(img_feature)
            img_t3 = rearrange(img_t3, 'b (o n k) -> b o n k', k=K2, n=N)
            x3 = torch.cat([img_t3, x3], dim=1)
        x3_a = self.activate(self.input_aligned[2](x3))
        x3_a = self.channel_up[2](x3_a)

        #
        hidden_state = [self.norm1(x1_a), self.norm2(x2_a), self.norm3(x3_a)]
        for h in self.layers:
            hidden_state = h(hidden_state)
        output = [self.out1(hidden_state[0]), self.out2(hidden_state[1]),
                  self.out3(hidden_state[2].flatten(1))]

        output[0] = rearrange(output[0], 'b o n k-> b n k o')
        output[1] = rearrange(output[1], 'b o n k-> b n k o')

        return output[0], output[1], output[2]
