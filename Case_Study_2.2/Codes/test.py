# coding=utf-8
import torch
from torch.optim import AdamW, SGD, Adam
import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import time
import collections


class Tester:
    def __init__(self, args, writer, model, optimizer, scheduler,
                 test_data, device, early_stop=5, test_fre=5):
        self.args = args
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_data = test_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        # self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=args.weight_decay)
        self.log_interval = args.log_interval
        self.best_nmse_random = 1e9
        self.warmup_steps = 5
        self.min_lr = args.min_lr
        self.best_nmse = 1e9
        self.early_stop = early_stop
        self.test_fre = test_fre

        self.mask_list = {'random': [0.85], 'temporal': [0.5], 'fre': [0.5]}

    def Test_iter(self, test_data, mask_ratio, mask_strategy, seed=None, dataset=''):
        error_nmse = 0
        num = 0
        # start time
        for _, batch in enumerate(test_data):
            # print(batch.shape)
            loss, _, pred, target, mask = self.model_forward(batch, mask_ratio, mask_strategy, seed=seed, data=dataset,
                                                             mode='forward')
            dim1 = pred.shape[0]
            pred_mask = pred.squeeze(dim=2)
            target_mask = target.squeeze(dim=2)

            y_pred = pred_mask[mask == 1].reshape(-1, 1).reshape(dim1, -1).detach().cpu().numpy()
            y_target = target_mask[mask == 1].reshape(-1, 1).reshape(dim1, -1).detach().cpu().numpy()

            error_nmse += np.sum(
                np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
            num += y_pred.shape[0]
        nmse = error_nmse / num
        return nmse

    def model_forward(self, batch, mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):
        # batch = [i.to(self.device) for i in batch]
        if (type(batch).__name__ == 'dict'):
            for value, key in batch.items():
                batch[value] = key.to(self.device)
            if self.args.rgb_aided:
                loss, loss2, pred, target, mask = self.model(
                    batch['h'], batch['img'],
                    mask_ratio=mask_ratio,
                    mask_strategy=mask_strategy,
                    seed=seed,
                    data=data,
                )
            else:
                loss, loss2, pred, target, mask = self.model(
                    batch['h'], rgb=None,
                    mask_ratio=mask_ratio,
                    mask_strategy=mask_strategy,
                    seed=seed,
                    data=data,
                )
        else:
            batch = batch.to(self.device).squeeze(1)
            loss, loss2, pred, target, mask = self.model(
                batch, rgb=None,
                mask_ratio=mask_ratio,
                mask_strategy=mask_strategy,
                seed=seed,
                data=data,
            )
        return loss, loss2, pred, target, mask
