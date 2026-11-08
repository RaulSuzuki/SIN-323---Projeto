"""Constantes e utilitarios compartilhados."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
BRACOL = RAIZ / "data" / "bracol_raw" / "coffee-datasets" / "coffee-datasets" / "leaf"
OUT = RAIZ / "outputs"
MODELOS = RAIZ / "models"

CACHE = RAIZ / "data" / "cache_512x256"

DOENCAS = ["miner", "rust", "phoma", "cercospora"]
NOMES_PT = {
    "miner": "Bicho-mineiro",
    "rust": "Ferrugem",
    "phoma": "Mancha de Phoma",
    "cercospora": "Cercosporiose",
}
SEVERIDADE_PT = ["Saudavel", "Muito baixa", "Baixa", "Alta", "Muito alta"]

LARGURA, ALTURA = 512, 256  # as fotos originais sao 2048x1024 (proporcao 2:1)
SEED = 42
MEDIA = (0.485, 0.456, 0.406)  # normalizacao do ImageNet
DESVIO = (0.229, 0.224, 0.225)


def dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
