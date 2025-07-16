import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider import Dataset_Pro
from models.GPT2point_open import GPT4SCAgrid
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
WEIGHT = ROOT / "Weights/best_ckpt_sd.pth"

PATH_FREQ = "./data/frequency_crossroad_28_test.mat"
PATH_LIDAR= "./data/LiDAR_point_feature_crossroad_28_test.mat"
PATH_SCATT= "./data/scatterer_gridmap_crossroad_28_test.mat"

# ------------------------ 参数 ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--batch",  type=int, default=1)
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--gpt_layers", type=int, default=6)
parser.add_argument("--seq_len",  type=int, default=1024)
parser.add_argument("--patch_size", type=int, default=64)
parser.add_argument("--stride",     type=int, default=32)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--pred_len', type=int, default=100)

args = parser.parse_args()
device = torch.device(args.device)

# ------------------------ 数据 ------------------------
test_set   = Dataset_Pro(PATH_FREQ, PATH_LIDAR, PATH_SCATT)
loader     = DataLoader(test_set, batch_size=args.batch, shuffle=True, pin_memory=True, drop_last=True)

# ------------------------ 模型 ------------------------
model = GPT4SCAgrid(args, device).to(device)
if WEIGHT.exists():
    state = torch.load(WEIGHT, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state.state_dict(), strict=False)
    print(f"✓ loaded: {WEIGHT.name}")

# ------------------ 工具函数（向量化） -----------------
round50 = lambda x: torch.round(x * 50.)

def masks(pred, gt):
    """返回四类布尔掩码"""
    p_nz, p_z, g_nz, g_z = pred != 0, pred == 0, gt != 0, gt == 0
    return p_z & g_z, p_nz & g_nz, p_nz & g_z, p_z & g_nz

def within_tol(pred, gt, tol=0.03):
    return torch.abs(pred - gt) <= torch.abs(gt) * tol

# ------------------------ 测试 ------------------------
@torch.no_grad()
def test():
    model.eval()
    zero_zero = nz_nz = nz_0 = 0
    tol_cnt   = 0
    total_pix = len(loader) * 100  # H×W=100

    for batch in tqdm(loader, desc="Testing"):
        gt, prev, prev_f = (t.to(device).float() for t in batch)
        pred = model(prev_f, prev)
        pred.clamp_(0, 2.0)

        pred = round50(pred)
        gt   = round50(gt)

        m00, m11, m10, _ = masks(pred, gt)
        zero_zero += m00.sum().item()
        nz_nz     += m11.sum().item()
        nz_0      += m10.sum().item()

        tol_cnt   += within_tol(pred, gt, 0.03).sum().item()

    # ------ 指标 ------
    classification = (zero_zero + nz_nz) / total_pix
    tol_acc= tol_cnt / total_pix
    print(f"classification    : {classification:.4f}")
    print(f"within 3% tol acc : {tol_acc:.4f}")

# ------------------------ main ------------------------
if __name__ == "__main__":

    test()
