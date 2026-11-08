"""Grad-CAM sobre a ultima etapa do ConvNeXt e uma metrica proxy contra o atalho do fundo."""
import numpy as np
import torch
import torch.nn.functional as F


def gradcam(modelo, x, classe):
    """x: tensor (1,3,H,W). Devolve mapa normalizado [0,1] (H,W) para a saida `classe` (doenca)."""
    modelo.eval()
    x = x.requires_grad_(False)
    with torch.enable_grad():
        mapa = modelo.features(x)
        mapa.retain_grad()
        v = modelo.drop(modelo.norm(modelo.pool(mapa)))
        logit = modelo.cab_doenca(v)[0, classe]
        grad = torch.autograd.grad(logit, mapa)[0]
    pesos = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((pesos * mapa).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def mascara_folha(img_rgb):
    """Aproximacao da folha: o fundo do BRACOL e cinza claro (saturacao baixa), a folha e saturada.
    Aproximada (inclui parte da sombra nas bordas); serve so como proxy. img_rgb: uint8 (H,W,3)."""
    a = img_rgb.astype(np.int16)
    sat = a.max(2) - a.min(2)
    return (sat > 40) | (a.max(2) < 70)


def massa_na_folha(cam, folha):
    """Fracao da massa do Grad-CAM que cai sobre a folha e fracao de area que a folha ocupa."""
    return float(cam[folha].sum() / (cam.sum() + 1e-8)), float(folha.mean())
