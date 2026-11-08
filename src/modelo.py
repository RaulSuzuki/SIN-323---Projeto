"""Dataset, transformacoes e modelo multitarefa (doencas multilabel + severidade)."""
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from comum import CACHE, DOENCAS, MEDIA, DESVIO, OUT

N_SEV = 5


def transformacao(treino):
    base = [T.ToTensor(), T.Normalize(MEDIA, DESVIO)]
    if not treino:
        return T.Compose(base)
    # augmentation: iluminacao, orientacao, pequenas oclusoes
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        *base,
        T.RandomErasing(p=0.25, scale=(0.01, 0.04), value=0),
    ])


class Folhas(Dataset):
    """Folhas de um split (train, val ou test) de outputs/metadados.csv, lidas do cache 512x256."""

    def __init__(self, split, treino=False, cache=CACHE):
        meta = pd.read_csv(OUT / "metadados.csv")
        self.meta = meta[meta["split"] == split].reset_index(drop=True)
        self.cache = cache
        self.tf = transformacao(treino)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        img = Image.open(self.cache / f"{int(r['id'])}.jpg").convert("RGB")
        y = torch.tensor(r[DOENCAS].astype(float).values, dtype=torch.float32)
        return self.tf(img), y, int(r["severity"]), int(r["id"])


class ModeloFolha(nn.Module):
    """ConvNeXt-Tiny pre-treinado no ImageNet + cabeca multilabel (4 doencas) + cabeca de severidade (5 niveis).
    Expoe features, pool, norm e drop para o Grad-CAM de gradcam.py."""

    def __init__(self, pretreinado=True):
        super().__init__()
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretreinado else None)
        self.features = net.features
        self.pool = net.avgpool
        self.norm = nn.Sequential(net.classifier[0], net.classifier[1])  # LayerNorm2d + Flatten
        self.drop = nn.Dropout(0.3)
        self.cab_doenca = nn.Linear(768, len(DOENCAS))
        self.cab_sev = nn.Linear(768, N_SEV)

    def forward(self, x, retornar_mapa=False):
        mapa = self.features(x)
        v = self.norm(self.pool(mapa))
        v = self.drop(v)
        saida = self.cab_doenca(v), self.cab_sev(v)
        return (*saida, mapa) if retornar_mapa else saida


def carregar(caminho, disp):
    m = ModeloFolha(pretreinado=False)
    m.load_state_dict(torch.load(caminho, map_location=disp))
    return m.to(disp).eval()
