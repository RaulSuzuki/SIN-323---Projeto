"""Treina o modelo multitarefa e grava os pesos em models/melhor_modelo.pt.

Uso: python -P src/treinar.py [--epocas 25]
Escolhe a epoca pelo mAP de validacao, com parada antecipada.
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from comum import CACHE, DOENCAS, MODELOS, OUT, SEED, dispositivo
from modelo import Folhas, ModeloFolha


def avaliar(modelo, dl, disp):
    modelo.eval()
    logits, sevs, ys, sy = [], [], [], []
    with torch.no_grad():
        for x, y, s, _ in dl:
            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                a, b = modelo(x.to(disp))
            logits.append(a.float().cpu()); sevs.append(b.float().cpu()); ys.append(y); sy.append(s)
    logits, sevs, ys, sy = torch.cat(logits), torch.cat(sevs), torch.cat(ys), torch.cat(sy)
    p = torch.sigmoid(logits).numpy()
    ap = [average_precision_score(ys[:, k], p[:, k]) for k in range(len(DOENCAS))]
    acc_sev = (sevs.argmax(1) == sy).float().mean().item()
    return float(np.mean(ap)), ap, acc_sev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--paciencia", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    disp = dispositivo()
    assert CACHE.exists(), f"cache {CACHE} nao existe: rode preparar_dados.py"
    MODELOS.mkdir(exist_ok=True)
    arq_modelo = MODELOS / "melhor_modelo.pt"
    print("dispositivo:", disp, torch.cuda.get_device_name(0) if disp.type == "cuda" else "(CPU - sera lento)", flush=True)

    ds_tr, ds_va = Folhas("train", True), Folhas("val")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, num_workers=2)
    meta = ds_tr.meta
    pos = meta[DOENCAS].sum().values
    pos_weight = torch.tensor(np.clip((len(meta) - pos) / pos, 1, 5), dtype=torch.float32).to(disp)
    cont = np.bincount(meta["severity"], minlength=5)
    peso_sev = torch.tensor(np.clip(cont.sum() / (5 * cont), 0.5, 4), dtype=torch.float32).to(disp)

    modelo = ModeloFolha().to(disp)
    perda_d = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    perda_s = nn.CrossEntropyLoss(weight=peso_sev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epocas * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=disp.type == "cuda")

    print("ambiente:", json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                                   "torchvision": torchvision.__version__, "cuda": torch.version.cuda}), flush=True)

    melhor, sem_melhora, hist = -1, 0, []
    for ep in range(1, args.epocas + 1):
        modelo.train(); t0 = time.time(); soma = 0
        for x, y, s, _ in dl_tr:
            x, y, s = x.to(disp), y.to(disp), s.to(disp)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                a, b = modelo(x)
                perda = perda_d(a.float(), y) + 0.5 * perda_s(b.float(), s)
            scaler.scale(perda).backward()
            scaler.step(opt); scaler.update(); sched.step()
            soma += perda.item()
        m_ap, aps, acc_sev = avaliar(modelo, dl_va, disp)
        hist.append({"epoca": ep, "perda_treino": soma / len(dl_tr), "val_mAP": m_ap,
                     **{f"val_AP_{d}": a for d, a in zip(DOENCAS, aps)}, "val_acc_severidade": acc_sev})
        print(f"ep {ep:02d} perda {soma/len(dl_tr):.3f} | val mAP {m_ap:.3f} sev {acc_sev:.3f} | {time.time()-t0:.0f}s", flush=True)
        if m_ap > melhor:
            melhor, sem_melhora = m_ap, 0
            torch.save(modelo.state_dict(), arq_modelo)
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                print("parada antecipada"); break
    pd.DataFrame(hist).to_csv(OUT / "historico_treino.csv", index=False)
    print("melhor val mAP:", round(melhor, 4), "| modelo:", arq_modelo)


if __name__ == "__main__":
    main()
