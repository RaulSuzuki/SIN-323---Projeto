"""Analise de erros: lista as folhas de teste com conjunto de doencas errado e mostra o Grad-CAM."""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from comum import CACHE, DOENCAS, NOMES_PT, MODELOS, OUT, dispositivo
from gradcam import gradcam
from modelo import Folhas, carregar

df = pd.read_csv(OUT / "predicoes_teste.csv")
cal = json.load(open(MODELOS / "calibracao.json")); ths = np.array(cal["limiares"])
Y = df[[f"y_{d}" for d in DOENCAS]].values.astype(int); P = df[[f"p_{d}" for d in DOENCAS]].values
pred = (P >= ths).astype(int)
err = np.where((pred != Y).any(1))[0]
nome = lambda v: "+".join(NOMES_PT[d] for d, b in zip(DOENCAS, v) if b) or "Saudavel"
linhas = []
for i in err:
    falt, extra = [DOENCAS[k] for k in range(4) if Y[i, k] and not pred[i, k]], [DOENCAS[k] for k in range(4) if pred[i, k] and not Y[i, k]]
    linhas.append({"id": int(df.id[i]), "real": nome(Y[i]), "previsto": nome(pred[i]),
                   "tipo": ("faltou " + ",".join(falt) if falt else "") + (" | " if falt and extra else "") + ("a mais " + ",".join(extra) if extra else ""),
                   "incerteza": round(float(df.incerteza[i]), 3), "recusada": int(df.incerteza[i] > cal["tau"]),
                   "sev_real": int(df.sev_real[i])})
L = pd.DataFrame(linhas); L.to_csv(OUT / "erros_teste.csv", index=False)
print(len(L), "folhas com erro de", len(df))
so_extra = L.tipo.str.startswith("a mais"); so_falta = L.tipo.str.startswith("faltou") & ~L.tipo.str.contains("a mais")
print("so doenca a mais:", int(so_extra.sum()), "| so doenca faltando:", int(so_falta.sum()), "| ambos:", int(len(L) - so_extra.sum() - so_falta.sum()))
print("recusadas entre os erros:", int(L.recusada.sum()), "de", len(L), "| recusadas entre os acertos:", int(((df.incerteza > cal['tau']) & ~np.isin(np.arange(len(df)), err)).sum()), "de", len(df) - len(L))
print("erros por tipo de doenca real:"); print(L.real.value_counts().head(8).to_string())
print("severidade real nos erros:", L.sev_real.value_counts().sort_index().to_dict(), "| nos acertos:", df.sev_real.drop(df.index[err]).value_counts().sort_index().to_dict())
print("\nprimeiros erros:"); print(L.head(14).to_string(index=False))

disp = dispositivo(); modelo = carregar(MODELOS / "melhor_modelo.pt", disp); tf = Folhas("test").tf
sel = L.sort_values("incerteza", ascending=False).head(16)
fig, ax = plt.subplots(4, 4, figsize=(18, 10))
for a in ax.ravel(): a.axis("off")
for a, (_, r) in zip(ax.ravel(), sel.iterrows()):
    img = Image.open(CACHE / f"{r.id}.jpg").convert("RGB"); idx = int(df.index[df.id == r.id][0])
    k = int(np.argmax(np.where(pred[idx] != Y[idx], P[idx], -1)))  # doenca em que errou (maior prob)
    cam = gradcam(modelo, tf(img).unsqueeze(0).to(disp), k)
    a.imshow(img); a.imshow(cam, cmap="jet", alpha=0.4)
    a.set_title(f"id {r.id}: real {r.real}\nprev {r.previsto} (CAM: {NOMES_PT[DOENCAS[k]]})", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "erros_gradcam.png", dpi=75)
