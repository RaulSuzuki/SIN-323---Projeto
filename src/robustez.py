"""Robustez: degrada as fotos de teste (luz, desfoque, compressao, resolucao, ruido, zoom, fundo, rotacao)."""
import io, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from torchvision import transforms as T

from comum import CACHE, DOENCAS, MEDIA, DESVIO, MODELOS, OUT, SEED, LARGURA, ALTURA, dispositivo
from gradcam import mascara_folha
from modelo import carregar

disp = dispositivo(); modelo = carregar(MODELOS / "melhor_modelo.pt", disp)
cal = json.load(open(MODELOS / "calibracao.json")); ths = np.array(cal["limiares"])
meta = pd.read_csv(OUT / "metadados.csv"); meta = meta[meta.split == "test"].reset_index(drop=True)
Y = meta[DOENCAS].values.astype(int); SY = meta["severity"].values
tf = T.Compose([T.ToTensor(), T.Normalize(MEDIA, DESVIO)])
rng = np.random.default_rng(SEED)


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def ruido(im, s):
    a = np.array(im).astype(np.float32) + rng.normal(0, s, (ALTURA, LARGURA, 3)); return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def zoom(im, f=0.6):
    w, h = int(LARGURA * f), int(ALTURA * f); x, y = (LARGURA - w) // 2, (ALTURA - h) // 2
    return im.crop((x, y, x + w, y + h)).resize((LARGURA, ALTURA), Image.LANCZOS)


def fundo_campo(im):
    """Troca o fundo (nao-folha, por limiar de cor) por textura verde-escura com ruido: clutter de campo."""
    a = np.array(im); m = mascara_folha(a)
    tex = gaussian_filter(rng.normal(0, 1, (ALTURA, LARGURA, 3)), sigma=(3, 3, 0)); tex = (tex - tex.min()) / (np.ptp(tex) + 1e-6)
    fundo = (np.array([40, 70, 35]) + tex * np.array([60, 80, 50])).astype(np.uint8)
    a[~m] = fundo[~m]; return Image.fromarray(a)


PERTURB = {
    "original": lambda im: im,
    "brilho x0.5 (escuro)": lambda im: ImageEnhance.Brightness(im).enhance(0.5),
    "brilho x1.6 (claro)": lambda im: ImageEnhance.Brightness(im).enhance(1.6),
    "contraste x0.5": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
    "desfoque r=2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "desfoque r=4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "JPEG qualidade 15": lambda im: jpeg(im, 15),
    "baixa resolucao (1/4)": lambda im: im.resize((LARGURA // 4, ALTURA // 4), Image.BILINEAR).resize((LARGURA, ALTURA), Image.BILINEAR),
    "ruido gaussiano s=20": lambda im: ruido(im, 20),
    "zoom 60% (recorte central)": zoom,
    "fundo de campo sintetico": fundo_campo,
    "rotacao 90 graus": lambda im: im.rotate(90, expand=True).resize((LARGURA, ALTURA), Image.LANCZOS),
}
imgs = [Image.open(CACHE / f"{i}.jpg").convert("RGB") for i in meta["id"]]
linhas = []
for nome, fn in PERTURB.items():
    L, S = [], []
    for s in range(0, len(imgs), 16):
        x = torch.stack([tf(fn(im)) for im in imgs[s:s + 16]]).to(disp)
        with torch.no_grad():
            a, b = modelo(x)
        L.append(a.cpu()); S.append(b.cpu())
    P = torch.sigmoid(torch.cat(L) / cal["Td"]).numpy(); PS = F.softmax(torch.cat(S) / cal["Ts"], 1).numpy()
    pred = (P >= ths).astype(int); sp = PS.argmax(1)
    amb = 1 - np.abs(2 * P - 1).min(1); ent = -(PS * np.log(PS + 1e-9)).sum(1) / np.log(5)
    tp = (pred & Y).sum(0); fp = (pred & (1 - Y)).sum(0); fn_ = ((1 - pred) & Y).sum(0)
    linhas.append({"perturbacao": nome, "acerto_exato": (pred == Y).all(1).mean(),
                   "F1_macro": float(np.mean(2 * tp / np.maximum(2 * tp + fp + fn_, 1))),
                   "severidade_acuracia": (sp == SY).mean(), "taxa_de_recusa": (np.maximum(amb, ent) > cal["tau"]).mean(),
                   "pct_previsto_saudavel": (pred.sum(1) == 0).mean()})
    print(linhas[-1], flush=True)
res = pd.DataFrame(linhas); res.to_csv(OUT / "teste_robustez.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 5)); r = res.iloc[::-1]
ax.barh(r["perturbacao"], r["acerto_exato"], color="#2e7d32"); ax.set_xlabel("acerto exato do conjunto de doencas (teste)")
for y_, v in enumerate(r["acerto_exato"]): ax.text(v + 0.01, y_, f"{v:.0%}", va="center", fontsize=8)
ax.set_xlim(0, 1); plt.tight_layout(); plt.savefig(OUT / "teste_robustez.png", dpi=120)
