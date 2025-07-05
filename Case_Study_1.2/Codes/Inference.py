import logging
import math
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import Dataset
from metrics import NMSELoss
from model.cross_stitch import NetworkCrossStitch


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################


def Inference(validate_data_loader):
    epoch_val_loss = []
    # ============Epoch Validate=============== #
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(validate_data_loader, 1):
            for key, value in batch.items():
                batch[key] = batch[key].to(device)
            outputs = model(batch['T1i'], batch['T2i'], batch['T3i'], batch['img'],)
            loss_batch = [0, 0, 0]
            for i in task_range:
                if i == 1:
                    _, loss_batch[0] = criterion_nmse(outputs[0], batch['T1o'])
                elif i == 2:
                    _, loss_batch[1] = criterion_nmse(outputs[1], batch['T2o'])
                elif i == 3:
                    _, loss_batch[2] = criterion_nmse(outputs[2], batch['T3o'])
            for i in task_range:
                loss_batch[i - 1] = loss_batch[i - 1].item()
            epoch_val_loss.append(loss_batch)
        print(f'Val Results: [{model_name}]', end=' ')
        loss_mean = np.nanmean(np.array(epoch_val_loss), axis=0)
        for loss, i in zip(loss_mean, range(len(loss_mean))):
            print(f'Task{i + 1} Loss: {loss}', end=' ')
        print()
    return loss_mean

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    batch_size = 64
    gpu_id = 3
    device = torch.device(f'cuda:{gpu_id}')
    Result = []
    task_range = [1, 2, 3]
    # load dataset
    path1 = './data/M3C'
    validate_set = Dataset(path1, is_train=0, SNR=12,
                           prev_len=32, pred_len=32, num_pilot=8)  # creat data for validation
    Model_Weights = {
        'LLM4WM-RGB': 'LLM4WM-RGB.pth',
        'LLM4WM': 'LLM4WM.pth',
        'SM-MTL-RGB': 'SM-MTL-RGB.pth',
        'SM-MTL': 'SM-MTL.pth',
        'SM-STL-RGB': 'SM-STL-RGB.pth',
        'SM-STL': 'SM-STL.pth',
    }
    for model_name, weight_path in Model_Weights.items():
        save_root = './Weights/'
        model = torch.load(save_root + weight_path, map_location=device, )
        model.device = device

        validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          drop_last=False)  # put training data to DataLoader for batches
        criterion_nmse = NMSELoss().to(device)

        result = Inference(validate_data_loader)  # call train function (
        result_db = 10 * np.log10(result)

        Result.append({'Methods': model_name,
                       'result': result, 'result(db)': result_db})

    import pandas as pd
    pd.DataFrame(Result).to_csv(f'Result.csv', index=False)
    print(f"Saved Result.csv")

