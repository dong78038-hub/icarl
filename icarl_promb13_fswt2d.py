# -*- coding: utf-8 -*-
"""
iCaRL reproduction for ProMB_13_FSWT2D.

This script keeps the user's task split, sampling protocol, Acc/IAcc/IF metrics,
MATLAB .mat outputs, feature exports and benchmark-style reporting, while replacing
only the continual-learning method with iCaRL:

1. cross-entropy training on the initial task;
2. class-incremental expansion;
3. exemplar rehearsal;
4. old-model distillation on old-class logits;
5. herding-style exemplar selection;
6. nearest-mean-of-exemplars prediction.

Expected dataset layout:

    ProMB_13_FSWT2D/
        Class1/train/*.png
        Class1/test/*.png
        ...
        Class13/train/*.png
        Class13/test/*.png

Folder names may also be `Class_1`, `class-1`, or any name containing a class id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import math
import random
import re
import time
import warnings

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

METHOD_NAME = "iCaRL"


@dataclass
class Config:
    seed: int = 42
    use_cuda: bool = True

    dataset_dir: str = r"F:\东东毕业\在投\LCEN\代码\数据\1D→2D\ProMB_13_FSWT2D"
    out_root: str = "./runs_promb13_fswt2d_user_protocol"
    n_labels: int = 13

    # 1-based raw class ids. Kept from the user's protocol.
    phase_classes: Tuple[Tuple[int, ...], ...] = (
        tuple(range(1, 7)),
        tuple(range(7, 9)),
        tuple(range(9, 11)),
        tuple(range(11, 14)),
    )

    phase0_train_samples_per_class: int = 100
    incremental_new_train_samples_per_class: int = 5
    test_samples_per_class: int = 100
    sampling_seed_offset: int = 2025

    image_size: int = 224
    image_channels: int = 3
    num_workers: int = 0

    base_ch: int = 64
    dropout: float = 0.20
    embed_dim: int = 128
    head_hidden: int = 256
    cls_label_smoothing: float = 0.05

    epochs_0: int = 30
    enhance_ep: int = 30
    batch_size: int = 16
    eval_bs: int = 256
    lr_phase0: float = 1e-3
    lr_enhance: float = 1e-3
    weight_decay: float = 1e-4
    kd_T: float = 2.0
    kd_lambda: float = 1.0
    exemplars_per_class: int = 5
    log_every: int = 1

    bench_iters: int = 200
    bench_warmup_iters: int = 50
    bench_batch_size: int = 1


CFG = Config()


def parse_args() -> Config:
    import argparse

    p = argparse.ArgumentParser("iCaRL on ProMB_13_FSWT2D")
    p.add_argument("--dataset-dir", type=str, default=CFG.dataset_dir)
    p.add_argument("--out-root", type=str, default=CFG.out_root)
    p.add_argument("--image-size", type=int, default=CFG.image_size)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--epochs-0", type=int, default=CFG.epochs_0)
    p.add_argument("--enhance-ep", type=int, default=CFG.enhance_ep)
    p.add_argument("--batch-size", type=int, default=CFG.batch_size)
    p.add_argument("--eval-bs", type=int, default=CFG.eval_bs)
    p.add_argument("--lr-phase0", type=float, default=CFG.lr_phase0)
    p.add_argument("--lr-enhance", type=float, default=CFG.lr_enhance)
    p.add_argument("--kd-lambda", type=float, default=CFG.kd_lambda)
    p.add_argument("--kd-T", type=float, default=CFG.kd_T)
    p.add_argument("--exemplars-per-class", type=int, default=CFG.exemplars_per_class)
    p.add_argument("--log-every", type=int, default=CFG.log_every)
    args = p.parse_args()

    cfg = copy.deepcopy(CFG)
    cfg.dataset_dir = args.dataset_dir
    cfg.out_root = args.out_root
    cfg.image_size = args.image_size
    cfg.seed = args.seed
    cfg.use_cuda = not args.cpu
    cfg.epochs_0 = args.epochs_0
    cfg.enhance_ep = args.enhance_ep
    cfg.batch_size = args.batch_size
    cfg.eval_bs = args.eval_bs
    cfg.lr_phase0 = args.lr_phase0
    cfg.lr_enhance = args.lr_enhance
    cfg.kd_lambda = args.kd_lambda
    cfg.kd_T = args.kd_T
    cfg.exemplars_per_class = args.exemplars_per_class
    cfg.log_every = args.log_every
    return cfg


CFG = parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
device = torch.device("cuda" if (CFG.use_cuda and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

PHASE_CLASSES_RAW_1 = [list(ph) for ph in CFG.phase_classes]
PHASE_CLASSES_RAW_0 = [[c - 1 for c in ph] for ph in PHASE_CLASSES_RAW_1]
ORDERED_CLASS_IDS_0 = [c for ph in PHASE_CLASSES_RAW_0 for c in ph]
CLASS_ID_MAP = {old_id: new_id for new_id, old_id in enumerate(ORDERED_CLASS_IDS_0)}
INV_CLASS_ID_MAP = {v: k for k, v in CLASS_ID_MAP.items()}
PHASE_CLASSES = [[CLASS_ID_MAP[c] for c in ph] for ph in PHASE_CLASSES_RAW_0]
N_PHASES = len(PHASE_CLASSES)


def _parse_class_id(folder_name: str) -> Optional[int]:
    m = re.search(r"Class[_\- ]?(\d+)", folder_name, flags=re.I)
    if m:
        return int(m.group(1)) - 1
    m = re.search(r"(\d+)", folder_name)
    if m:
        return int(m.group(1)) - 1
    return None


def scan_fswt2d_dataset(root: str):
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"Dataset directory not found: {rootp}")

    img_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    train_by_class: Dict[int, List[Path]] = {}
    test_by_class: Dict[int, List[Path]] = {}
    class_names: Dict[int, str] = {}

    dirs = [p for p in sorted(rootp.iterdir()) if p.is_dir()]
    for enum_idx, d in enumerate(dirs):
        cid_raw = _parse_class_id(d.name)
        if cid_raw is None:
            cid_raw = enum_idx
        if cid_raw not in CLASS_ID_MAP:
            continue
        cid = CLASS_ID_MAP[cid_raw]
        class_names[cid] = d.name
        tr_dir = d / "train"
        te_dir = d / "test"
        tr = sorted([p for p in tr_dir.rglob("*") if p.suffix.lower() in img_ext]) if tr_dir.exists() else []
        te = sorted([p for p in te_dir.rglob("*") if p.suffix.lower() in img_ext]) if te_dir.exists() else []
        train_by_class[cid] = tr
        test_by_class[cid] = te
    return train_by_class, test_by_class, class_names


class ImageRecordsDataset(Dataset):
    def __init__(self, records: List[Tuple[Path, int]], image_size: int):
        self.records = list(records)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return x, int(label)


class TensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def sample_records(by_class: Dict[int, List[Path]], cls_list: List[int], per_class_num: int, seed_base: int):
    recs = []
    for cls in cls_list:
        files = list(by_class.get(cls, []))
        if len(files) == 0:
            print(f"[WARN] No files for class {cls + 1}")
            continue
        rng = random.Random(int(seed_base + cls))
        rng.shuffle(files)
        chosen = files[:min(per_class_num, len(files))]
        recs.extend([(p, cls) for p in chosen])
    return recs


def records_to_tensor(records: List[Tuple[Path, int]], image_size: int, dev: torch.device, bs: int = 256):
    ds = ImageRecordsDataset(records, image_size)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=CFG.num_workers)
    xs, ys = [], []
    for xb, yb in loader:
        xs.append(xb)
        ys.append(yb.long())
    if not xs:
        return torch.empty(0, 3, image_size, image_size, device=dev), torch.empty(0, dtype=torch.long, device=dev)
    return torch.cat(xs, dim=0).to(dev), torch.cat(ys, dim=0).to(dev)


def build_phases(train_by_class, test_by_class):
    phases = []
    for phase_idx, raw_cls_list in enumerate(PHASE_CLASSES):
        train_num = CFG.phase0_train_samples_per_class if phase_idx == 0 else CFG.incremental_new_train_samples_per_class
        tr_records = sample_records(train_by_class, raw_cls_list, train_num,
                                    CFG.seed + CFG.sampling_seed_offset + phase_idx * 1000)
        te_records = sample_records(test_by_class, raw_cls_list, CFG.test_samples_per_class,
                                    CFG.seed + CFG.sampling_seed_offset + phase_idx * 2000 + 999)
        x_tr, y_tr = records_to_tensor(tr_records, CFG.image_size, device, bs=CFG.eval_bs)
        x_te, y_te = records_to_tensor(te_records, CFG.image_size, device, bs=CFG.eval_bs)
        phases.append({
            "classes": [CLASS_ID_MAP[INV_CLASS_ID_MAP[c]] for c in raw_cls_list],
            "classes_raw": [INV_CLASS_ID_MAP[c] for c in raw_cls_list],
            "train_x": x_tr,
            "train_y": y_tr,
            "test_x": x_te,
            "test_y": y_te,
        })
    return phases


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, stride, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + identity)


class ResNet18Small2D(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.stem = ConvBNAct(in_ch, base_ch, 3, 1, 1)
        self.in_ch = base_ch
        self.layer1 = self._make_layer(base_ch, 2, stride=1)
        self.layer2 = self._make_layer(base_ch * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_ch * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_ch * 8, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = base_ch * 8

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock2D(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock2D(self.in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.gap(x).flatten(1)


def _expand_linear(old_fc: nn.Linear, n_new: int):
    new_fc = nn.Linear(old_fc.in_features, old_fc.out_features + n_new).to(old_fc.weight.device)
    nn.init.kaiming_uniform_(new_fc.weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_fc.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(new_fc.bias, -bound, bound)
    with torch.no_grad():
        new_fc.weight[:old_fc.out_features].copy_(old_fc.weight)
        new_fc.bias[:old_fc.out_features].copy_(old_fc.bias)
    return new_fc


class ICaRL2DModel(nn.Module):
    def __init__(self, n_classes: int, cfg: Config):
        super().__init__()
        self.backbone = ResNet18Small2D(3, base_ch=cfg.base_ch)
        feat_dim = self.backbone.out_dim
        self.embed = nn.Sequential(
            nn.Linear(feat_dim, cfg.head_hidden),
            nn.BatchNorm1d(cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.embed_dim),
        )
        self.classifier = nn.Linear(cfg.embed_dim, n_classes)
        self.n_classes = int(n_classes)

    def expand(self, n_new: int):
        self.classifier = _expand_linear(self.classifier, n_new)
        self.n_classes += int(n_new)

    def extract(self, x):
        z = self.backbone(x)
        return F.normalize(self.embed(z), dim=1)

    def forward(self, x):
        emb = self.extract(x)
        logits = self.classifier(emb)
        return {"features": emb, "joint_embed": emb, "base_logits": logits}

    @torch.no_grad()
    def get_joint_embedding(self, x: torch.Tensor, bs: int = 256):
        self.eval()
        outs = []
        for i in range(0, len(x), bs):
            outs.append(self.forward(x[i:i + bs])["joint_embed"].detach())
        if not outs:
            return torch.empty(0, CFG.embed_dim, device=x.device)
        return torch.cat(outs, dim=0)


def make_balanced_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, global_n_classes: int, drop_last: bool = False):
    x_cpu = x.detach().cpu() if x.device.type != "cpu" else x.detach()
    y_cpu = y.detach().cpu() if y.device.type != "cpu" else y.detach()
    counts = torch.bincount(y_cpu, minlength=global_n_classes).float().clamp(min=1)
    weights = 1.0 / counts[y_cpu]
    sampler = WeightedRandomSampler(weights, num_samples=len(y_cpu), replacement=True)
    return DataLoader(TensorDataset(x_cpu, y_cpu), batch_size=batch_size, sampler=sampler, drop_last=drop_last)


def soft_kd(student_logits, teacher_logits, T: float):
    if student_logits.numel() == 0 or teacher_logits.numel() == 0:
        return student_logits.new_tensor(0.0)
    p = F.softmax(teacher_logits / T, dim=1)
    q = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(q, p, reduction="batchmean") * (T ** 2)


class ExemplarMemory:
    def __init__(self, n_per_class: int = 5):
        self.n = int(n_per_class)
        self.ex_x: Dict[int, torch.Tensor] = {}
        self.ex_y: Dict[int, torch.Tensor] = {}
        self.prototypes: Optional[torch.Tensor] = None
        self.prototype_classes: Optional[torch.Tensor] = None

    def get(self):
        if not self.ex_x:
            return None, None
        xs = torch.cat([v for _, v in sorted(self.ex_x.items())], dim=0).to(device)
        ys = torch.cat([v for _, v in sorted(self.ex_y.items())], dim=0).to(device)
        return xs, ys

    @torch.no_grad()
    def update(self, model: ICaRL2DModel, x: torch.Tensor, y: torch.Tensor, class_ids: List[int], bs: int = 256):
        model.eval()
        feats = model.get_joint_embedding(x, bs=bs)
        for c in class_ids:
            mask = (y == int(c))
            if mask.sum() == 0:
                continue
            f_c = F.normalize(feats[mask], dim=1)
            x_c = x[mask]
            y_c = y[mask]
            class_mean = F.normalize(f_c.mean(0), dim=0)

            selected = []
            selected_sum = torch.zeros_like(class_mean)
            candidate = torch.arange(f_c.size(0), device=f_c.device)
            for k in range(min(self.n, f_c.size(0))):
                mu_p = F.normalize((selected_sum.unsqueeze(0) + f_c[candidate]) / float(k + 1), dim=1)
                dist = ((mu_p - class_mean.unsqueeze(0)) ** 2).sum(1)
                best_pos = int(torch.argmin(dist).item())
                best_idx = int(candidate[best_pos].item())
                selected.append(best_idx)
                selected_sum = selected_sum + f_c[best_idx]
                candidate = candidate[candidate != best_idx]
                if candidate.numel() == 0:
                    break

            idx = torch.tensor(selected, dtype=torch.long, device=x.device)
            self.ex_x[int(c)] = x_c[idx].detach().cpu()
            self.ex_y[int(c)] = y_c[idx].detach().cpu()
        self.refresh_prototypes(model, bs=bs)

    @torch.no_grad()
    def refresh_prototypes(self, model: ICaRL2DModel, bs: int = 256):
        if not self.ex_x:
            self.prototypes = None
            self.prototype_classes = None
            return
        protos, cls_ids = [], []
        for c, x_c_cpu in sorted(self.ex_x.items()):
            x_c = x_c_cpu.to(device)
            f = model.get_joint_embedding(x_c, bs=bs)
            protos.append(F.normalize(f.mean(0), dim=0))
            cls_ids.append(int(c))
        self.prototypes = torch.stack(protos, dim=0).to(device)
        self.prototype_classes = torch.tensor(cls_ids, dtype=torch.long, device=device)


def combine_new_memory(new_x, new_y, memory: Optional[ExemplarMemory]):
    if memory is None:
        return new_x, new_y
    ex_x, ex_y = memory.get()
    if ex_x is None:
        return new_x, new_y
    return torch.cat([new_x, ex_x], dim=0), torch.cat([new_y, ex_y], dim=0)


def _nme_logits(model: ICaRL2DModel, x: torch.Tensor, memory: ExemplarMemory):
    if memory.prototypes is None or memory.prototype_classes is None:
        return model(x)["base_logits"]
    feats = F.normalize(model.forward(x)["joint_embed"], dim=1)
    protos = F.normalize(memory.prototypes, dim=1)
    d = torch.cdist(feats, protos, p=2)
    logits_small = -d
    logits = torch.full((x.size(0), model.n_classes), -1e9, device=x.device)
    logits[:, memory.prototype_classes] = logits_small
    return logits


def predict_logits(model: ICaRL2DModel, x: torch.Tensor, memory: Optional[ExemplarMemory] = None):
    if memory is not None and memory.prototypes is not None:
        return _nme_logits(model, x, memory)
    return model(x)["base_logits"]


@torch.no_grad()
def evaluate(model: ICaRL2DModel, x: torch.Tensor, y: torch.Tensor, bs: int = 256, memory: Optional[ExemplarMemory] = None) -> float:
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(x), bs):
        logits = predict_logits(model, x[i:i + bs], memory=memory)
        pred = logits.argmax(dim=1)
        yb = y[i:i + bs]
        correct += (pred == yb).sum().item()
        total += len(yb)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_embedding_retrieval(model: ICaRL2DModel, x_gallery, y_gallery, x_query=None, y_query=None, topk: int = 5):
    model.eval()
    if x_query is None:
        x_query, y_query = x_gallery, y_gallery
    e_gallery = model.get_joint_embedding(x_gallery, bs=CFG.eval_bs).float()
    e_query = model.get_joint_embedding(x_query, bs=CFG.eval_bs).float()
    dist = torch.cdist(e_query, e_gallery, p=2)
    _sync_device(dist.device)
    t0 = time.perf_counter()
    topk_idx = torch.topk(-dist, k=min(topk, dist.size(1)), dim=1).indices
    _sync_device(dist.device)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(e_query))
    precisions, recalls = [], []
    for i in range(len(e_query)):
        idx = topk_idx[i]
        hits = (y_gallery[idx] == y_query[i]).float()
        precisions.append(hits.mean().item())
        total_pos = (y_gallery == y_query[i]).sum().item()
        recalls.append(hits.sum().item() / max(total_pos, 1))
    return {f"Precision@{topk}": float(np.mean(precisions)), f"Recall@{topk}": float(np.mean(recalls)), "RetrievalTime_ms": retrieval_ms}


def compute_IAcc(M, k):
    vals = [M[i][k] for i in range(k + 1) if M[i][k] is not None]
    return float(np.mean(vals)) if vals else 0.0


def compute_IF(M, k):
    if k == 0:
        return 0.0
    vals = [M[i][i] - M[i][k] for i in range(k) if M[i][i] is not None and M[i][k] is not None]
    return float(np.mean(vals)) if vals else 0.0


def _sync_device(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def print_banner(ph: int, M):
    iacc = compute_IAcc(M, ph)
    ifr = compute_IF(M, ph)
    print(f"\n  ╔═ Phase {ph} test results ══════════════════════════════╗")
    for i in range(ph + 1):
        v = M[i][ph]
        tag = "  ← NEW" if i == ph else ""
        if v is not None:
            print(f"  ║  Task {i} (classes {[c + 1 for c in PHASE_CLASSES_RAW_0[i]]})  Acc = {v:6.2f}%{tag}")
    print("  ╠═══════════════════════════════════════════════════════╣")
    print(f"  ║  IAcc = {iacc:6.2f}%      IF = {ifr:6.2f}%")
    print("  ╚═══════════════════════════════════════════════════════╝")


def _labels_to_raw_1based_np(labels: np.ndarray) -> np.ndarray:
    labels = labels.reshape(-1).astype(np.int64)
    raw = np.array([INV_CLASS_ID_MAP.get(int(v), int(v)) + 1 for v in labels], dtype=np.int64)
    return raw.reshape(-1, 1)


def _build_class_to_phase_map() -> Dict[int, int]:
    class_to_phase = {}
    for phase_id, cls_list in enumerate(PHASE_CLASSES):
        for cls_id in cls_list:
            class_to_phase[int(cls_id)] = int(phase_id)
    return class_to_phase


CLASS_TO_PHASE_MAP = _build_class_to_phase_map()


@torch.no_grad()
def _predict_class_labels_np(model: ICaRL2DModel, x: torch.Tensor, bs: int = 256, memory: Optional[ExemplarMemory] = None):
    model.eval()
    preds = []
    for i in range(0, len(x), bs):
        logits = predict_logits(model, x[i:i + bs], memory=memory)
        preds.append(logits.argmax(dim=1).detach().cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.int64)


@torch.no_grad()
def save_phase_feature_mat(model: ICaRL2DModel, phases, upto_phase: int, save_path: str, bs: int = 256, memory: Optional[ExemplarMemory] = None):
    x_list, y_list = [], []
    for i in range(int(upto_phase) + 1):
        x_list.append(phases[i]["test_x"])
        y_list.append(phases[i]["test_y"])
    all_x = torch.cat(x_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    features = model.get_joint_embedding(all_x, bs=bs).detach().cpu().numpy().astype(np.float32)
    true_cls = all_y.detach().cpu().numpy().astype(np.int64).reshape(-1)
    pred_cls = _predict_class_labels_np(model, all_x, bs=bs, memory=memory).reshape(-1)
    sio.savemat(save_path, {
        "features": features,
        "labels": true_cls.reshape(-1, 1),
        "true_labels": true_cls.reshape(-1, 1),
        "pred_labels": pred_cls.reshape(-1, 1),
        "true_labels_raw": _labels_to_raw_1based_np(true_cls),
        "pred_labels_raw": _labels_to_raw_1based_np(pred_cls),
        "upto_phase": np.array([[int(upto_phase)]], dtype=np.int64),
        "num_model_classes": np.array([[int(model.n_classes)]], dtype=np.int64),
    })


def _build_confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    y_true = y_true.reshape(-1).astype(np.int64)
    y_pred = y_pred.reshape(-1).astype(np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


@torch.no_grad()
def save_class_confusion_mat_for_phase(model: ICaRL2DModel, phases, upto_phase: int, save_path: str, bs: int = 256, all_test_classes: bool = True, memory: Optional[ExemplarMemory] = None):
    if all_test_classes:
        eval_phase_ids = list(range(N_PHASES))
        num_matrix_classes = int(CFG.n_labels)
    else:
        eval_phase_ids = list(range(int(upto_phase) + 1))
        num_matrix_classes = int(model.n_classes)
    all_x = torch.cat([phases[i]["test_x"] for i in eval_phase_ids], dim=0)
    all_y = torch.cat([phases[i]["test_y"] for i in eval_phase_ids], dim=0)
    true_cls = all_y.detach().cpu().numpy().astype(np.int64).reshape(-1)
    pred_cls = _predict_class_labels_np(model, all_x, bs=bs, memory=memory).reshape(-1)
    cm = _build_confusion_matrix_np(true_cls, pred_cls, num_classes=num_matrix_classes)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(np.float32) / np.maximum(row_sum, 1)
    class_ids_0based = np.arange(num_matrix_classes, dtype=np.int64).reshape(-1, 1)
    sio.savemat(save_path, {
        "cm": cm,
        "cm_norm": cm_norm,
        "y_true": true_cls.reshape(-1, 1),
        "y_pred": pred_cls.reshape(-1, 1),
        "y_true_raw": _labels_to_raw_1based_np(true_cls),
        "y_pred_raw": _labels_to_raw_1based_np(pred_cls),
        "class_ids_0based": class_ids_0based,
        "class_ids_1based": _labels_to_raw_1based_np(class_ids_0based),
        "phase_idx": np.array([[int(upto_phase)]], dtype=np.int64),
        "num_model_classes": np.array([[int(model.n_classes)]], dtype=np.int64),
        "num_matrix_classes": np.array([[int(num_matrix_classes)]], dtype=np.int64),
        "all_test_classes": np.array([[int(all_test_classes)]], dtype=np.int64),
    })


@torch.no_grad()
def save_final_phase_confusion_mat(model: ICaRL2DModel, phases, save_path: str, bs: int = 256, memory: Optional[ExemplarMemory] = None):
    all_x = torch.cat([ph["test_x"] for ph in phases], dim=0)
    all_y = torch.cat([ph["test_y"] for ph in phases], dim=0)
    pred_cls = _predict_class_labels_np(model, all_x, bs=bs, memory=memory)
    true_cls = all_y.detach().cpu().numpy().astype(np.int64)
    true_phase = np.array([CLASS_TO_PHASE_MAP[int(v)] for v in true_cls], dtype=np.int64).reshape(-1, 1)
    pred_phase = np.array([CLASS_TO_PHASE_MAP[int(v)] for v in pred_cls], dtype=np.int64).reshape(-1, 1)
    cm = _build_confusion_matrix_np(true_phase.reshape(-1), pred_phase.reshape(-1), num_classes=N_PHASES)
    sio.savemat(save_path, {"cm": cm, "y_true": true_phase, "y_pred": pred_phase})


def make_optimizer(params, lr: float, weight_decay: float = 1e-4):
    return optim.AdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=weight_decay)


def train_phase0(model: ICaRL2DModel, trx, try_, epochs, batch_size, lr, log_every, test_x_ph=None, test_y_ph=None, memory: Optional[ExemplarMemory] = None):
    opt = make_optimizer(model.parameters(), lr=lr, weight_decay=CFG.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        loader = make_balanced_loader(trx, try_, batch_size, model.n_classes, drop_last=True)
        total_loss, total_ce = 0.0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            ce = F.cross_entropy(out["base_logits"], yb, label_smoothing=CFG.cls_label_smoothing)
            ce.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += ce.item()
            total_ce += ce.item()
        sch.step()
        if (ep + 1) % log_every == 0:
            tr_acc = evaluate(model, trx, try_, bs=CFG.eval_bs, memory=memory)
            line = f"  [Ph0] Ep {ep + 1:03d}/{epochs} | loss={total_loss / max(1, len(loader)):.4f} | LCE={total_ce / max(1, len(loader)):.4f} | train={tr_acc:.2f}%"
            if test_x_ph is not None:
                te_acc = evaluate(model, test_x_ph, test_y_ph, bs=CFG.eval_bs, memory=memory)
                line += f" | test={te_acc:.2f}%"
            print(line)


def train_phase_k(model: ICaRL2DModel, old_model: ICaRL2DModel, n_old: int, new_x, new_y, memory: ExemplarMemory, phases, phase_idx: int, acc_matrix):
    k = phase_idx
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    print(f"  ┌ [iCaRL] {CFG.enhance_ep} epochs — LCE + exemplar rehearsal + distillation")
    opt = make_optimizer(model.parameters(), lr=CFG.lr_enhance, weight_decay=CFG.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.enhance_ep)
    comb_x, comb_y = combine_new_memory(new_x, new_y, memory)

    for ep in range(CFG.enhance_ep):
        model.train()
        loader = make_balanced_loader(comb_x, comb_y, CFG.batch_size, model.n_classes, drop_last=True)
        tl, tce, tkd = 0.0, 0.0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            with torch.no_grad():
                old_out = old_model(xb)
            ce = F.cross_entropy(out["base_logits"], yb, label_smoothing=CFG.cls_label_smoothing)
            kd = soft_kd(out["base_logits"][:, :n_old], old_out["base_logits"][:, :n_old], CFG.kd_T)
            loss = ce + CFG.kd_lambda * kd
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item()
            tce += ce.item()
            tkd += kd.item()
        sch.step()
        memory.refresh_prototypes(model, CFG.eval_bs)
        if (ep + 1) % CFG.log_every == 0:
            task_accs = [evaluate(model, phases[i]["test_x"], phases[i]["test_y"], bs=CFG.eval_bs, memory=memory) for i in range(k + 1)]
            iacc_now = float(np.mean(task_accs))
            fgt = [acc_matrix[i][i] - task_accs[i] for i in range(k) if acc_matrix[i][i] is not None]
            ifr_now = float(np.mean(fgt)) if fgt else 0.0
            t_str = "  ".join([f"T{i}={task_accs[i]:.1f}%" for i in range(k + 1)])
            print(f"  │  [iCaRL] Ep {ep + 1:03d}/{CFG.enhance_ep} | loss={tl / max(1, len(loader)):.4f} | LCE={tce / max(1, len(loader)):.4f} | LDI={tkd / max(1, len(loader)):.4f} | {t_str} | IAcc={iacc_now:.2f}% | IF={ifr_now:.2f}%")
    print("  └ iCaRL done.")


def count_flops(model: nn.Module, image_size: int, memory: Optional[ExemplarMemory] = None) -> float:
    flops = []

    def add(v):
        if v > 0:
            flops.append(int(v))

    def conv_hook(m, inp, out):
        x = inp[0]
        b = x.size(0)
        cout = m.out_channels
        h, w = out.shape[-2:]
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        cin_eff = m.in_channels // m.groups
        add(2 * b * cout * h * w * cin_eff * kh * kw)

    def linear_hook(m, inp, out):
        x = inp[0]
        b = max(1, x.numel() // m.in_features)
        add(2 * b * m.in_features * m.out_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device)
    with torch.inference_mode():
        predict_logits(model, dummy, memory=memory)
    for h in hooks:
        h.remove()
    return sum(flops) / 1e9


def benchmark_inference(model: nn.Module, image_size: int, bench_iters: int, warmup_iters: int = 50, batch_size: int = 1, memory: Optional[ExemplarMemory] = None):
    model.eval()
    dev = next(model.parameters()).device
    dummy = torch.zeros(batch_size, 3, image_size, image_size, device=dev)
    with torch.inference_mode():
        for _ in range(max(0, warmup_iters)):
            predict_logits(model, dummy, memory=memory)
        _sync_device(dev)
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            predict_logits(model, dummy, memory=memory)
        _sync_device(dev)
        elapsed = time.perf_counter() - t0
    return elapsed * 1000.0 / max(1, bench_iters) / max(1, batch_size)


def main():
    out_dir = Path(CFG.out_root) / METHOD_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print(f" {METHOD_NAME} + ProMB_13 FSWT2D (User protocol)")
    print("=" * 76)
    print(" Front-end replacement: FSWT 2D pseudo-color images")
    print(f" Sampling: phase0/class={CFG.phase0_train_samples_per_class}, incremental new/class={CFG.incremental_new_train_samples_per_class}, test/class={CFG.test_samples_per_class}")
    print(f" Image size: {CFG.image_size} x {CFG.image_size} x 3")
    print(f" Output dir: {out_dir}")

    train_by_class, test_by_class, _class_names = scan_fswt2d_dataset(CFG.dataset_dir)
    phases = build_phases(train_by_class, test_by_class)
    print("\nPhase layout:")
    for i, ph in enumerate(phases):
        raw_disp = [c + 1 for c in ph["classes_raw"]]
        print(f"  Ph {i}: classes={raw_disp} | train={ph['train_x'].size(0)} | test={ph['test_x'].size(0)}")

    n_init = len(phases[0]["classes"])
    model = ICaRL2DModel(n_classes=n_init, cfg=CFG).to(device)
    n_param_init = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nInitial params: {n_param_init:.3f} M")

    memory = ExemplarMemory(n_per_class=CFG.exemplars_per_class)
    acc_matrix = [[None] * N_PHASES for _ in range(N_PHASES)]

    print("\n" + "─" * 76)
    print(f"[Phase 0] Initial training | classes: {[c + 1 for c in phases[0]['classes_raw']]}")
    print("─" * 76)
    train_phase0(model, phases[0]["train_x"], phases[0]["train_y"], epochs=CFG.epochs_0,
                 batch_size=CFG.batch_size, lr=CFG.lr_phase0, log_every=CFG.log_every,
                 test_x_ph=phases[0]["test_x"], test_y_ph=phases[0]["test_y"], memory=memory)

    memory.update(model, phases[0]["train_x"], phases[0]["train_y"], phases[0]["classes"], bs=CFG.eval_bs)
    acc_matrix[0][0] = evaluate(model, phases[0]["test_x"], phases[0]["test_y"], bs=CFG.eval_bs, memory=memory)
    print_banner(0, acc_matrix)
    save_phase_feature_mat(model, phases, upto_phase=0, save_path=str(out_dir / "phase0.mat"), bs=CFG.eval_bs, memory=memory)
    save_class_confusion_mat_for_phase(model, phases, upto_phase=0, save_path=str(out_dir / "cm_phase0_class.mat"), bs=CFG.eval_bs, all_test_classes=True, memory=memory)

    for ph in range(1, N_PHASES):
        print("\n" + "─" * 76)
        print(f"[Phase {ph}] Incremental | new classes: {[c + 1 for c in phases[ph]['classes_raw']]}")
        print("─" * 76)
        old_model = copy.deepcopy(model).to(device)
        n_old = model.n_classes
        model.expand(len(phases[ph]["classes"]))
        print(f"  → Expanded to {model.n_classes} classes; old classes = {n_old}")

        train_phase_k(model=model, old_model=old_model, n_old=n_old,
                      new_x=phases[ph]["train_x"], new_y=phases[ph]["train_y"],
                      memory=memory, phases=phases, phase_idx=ph, acc_matrix=acc_matrix)

        memory.update(model, phases[ph]["train_x"], phases[ph]["train_y"], phases[ph]["classes"], bs=CFG.eval_bs)

        for prev in range(ph + 1):
            acc_matrix[prev][ph] = evaluate(model, phases[prev]["test_x"], phases[prev]["test_y"], bs=CFG.eval_bs, memory=memory)
        print_banner(ph, acc_matrix)
        save_phase_feature_mat(model, phases, upto_phase=ph, save_path=str(out_dir / f"phase{ph}.mat"), bs=CFG.eval_bs, memory=memory)
        save_class_confusion_mat_for_phase(model, phases, upto_phase=ph, save_path=str(out_dir / f"cm_phase{ph}_class.mat"), bs=CFG.eval_bs, all_test_classes=True, memory=memory)

    print("\n" + "═" * 76)
    print(" FINAL CONTINUAL LEARNING SUMMARY")
    print("═" * 76)
    hdr = "       | " + "  ".join([f"Task{i}" for i in range(N_PHASES)]) + " |  IAcc  |   IF"
    print(hdr)
    print("-" * len(hdr))
    for ph in range(N_PHASES):
        row = f"Ph {ph}  | "
        for task in range(N_PHASES):
            v = acc_matrix[task][ph]
            row += f" {v:6.2f}%" if v is not None else "   ---  "
        row += f" | {compute_IAcc(acc_matrix, ph):6.2f}% | {compute_IF(acc_matrix, ph):5.2f}%"
        print(row)

    all_test_x = torch.cat([ph["test_x"] for ph in phases], dim=0)
    all_test_y = torch.cat([ph["test_y"] for ph in phases], dim=0)
    emb_metrics = evaluate_embedding_retrieval(model, all_test_x, all_test_y, topk=5)
    save_final_phase_confusion_mat(model, phases, save_path=str(out_dir / "cm.mat"), bs=CFG.eval_bs, memory=memory)

    n_final = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = count_flops(model, CFG.image_size, memory=memory)
    infer_ms = benchmark_inference(model, CFG.image_size, CFG.bench_iters, CFG.bench_warmup_iters, CFG.bench_batch_size, memory=memory)

    print("\n" + "═" * 76)
    print(f"Final parameters : {n_final:.3f} M")
    print(f"FLOPs (single)   : {gflops:.4f} G")
    print(f"Avg inference    : {infer_ms:.3f} ms")
    print("Embedding retrieval metrics:")
    for k, v in emb_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("═" * 76)

    sio.savemat(str(out_dir / "summary.mat"), {
        "acc_matrix": np.array([[np.nan if v is None else v for v in row] for row in acc_matrix], dtype=np.float32),
        "final_IAcc": np.array([[compute_IAcc(acc_matrix, N_PHASES - 1)]], dtype=np.float32),
        "final_IF": np.array([[compute_IF(acc_matrix, N_PHASES - 1)]], dtype=np.float32),
        "params_M": np.array([[n_final]], dtype=np.float32),
        "gflops": np.array([[gflops]], dtype=np.float32),
        "infer_ms": np.array([[infer_ms]], dtype=np.float32),
    })

    return model, acc_matrix, emb_metrics


if __name__ == "__main__":
    model, acc_matrix, emb_metrics = main()
