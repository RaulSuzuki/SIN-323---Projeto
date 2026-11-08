"""Intervalos de confianca (bootstrap sobre as folhas de teste) para as metricas do teste."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score

from comum import DOENCAS, MODELOS, NOMES_PT, OUT, SEED

B = 2000
df = pd.read_csv(OUT / "predicoes_teste.csv")
ths = np.array(json.load(open(MODELOS / "calibracao.json"))["limiares"])
Y = df[[f"y_{d}" for d in DOENCAS]].values.astype(int)
P = df[[f"p_{d}" for d in DOENCAS]].values
PS = df[[f"ps_{j}" for j in range(5)]].values
SY = df["sev_real"].values
pred = (P >= ths).astype(int); sp = PS.argmax(1)


def f1(y, p):
    tp = (y & p).sum(); return 2 * tp / max(2 * tp + (~y.astype(bool) & p.astype(bool)).sum() + (y.astype(bool) & ~p.astype(bool)).sum(), 1)


def metricas(i):
    y, p, pr, s, sy = Y[i], P[i], pred[i], sp[i], SY[i]
    out = {"acerto_exato": (pr == y).all(1).mean(), "acuracia_por_rotulo": (pr == y).mean(),
           "severidade_acuracia": (s == sy).mean(), "severidade_MAE": np.abs(s - sy).mean()}
    f1s, aps = [], []
    for k, d in enumerate(DOENCAS):
        f = f1(y[:, k], pr[:, k]); f1s.append(f); out[f"F1_{NOMES_PT[d]}"] = f
        if 0 < y[:, k].sum() < len(y):
            aps.append(average_precision_score(y[:, k], p[:, k]))
    out["F1_macro"] = np.mean(f1s); out["mAP"] = np.mean(aps)
    return out


rng = np.random.default_rng(SEED)
base = metricas(np.arange(len(df)))
boot = pd.DataFrame([metricas(rng.integers(0, len(df), len(df))) for _ in range(B)])
tab = pd.DataFrame({"estimativa": pd.Series(base), "IC95_inf": boot.quantile(0.025), "IC95_sup": boot.quantile(0.975)})
tab.to_csv(OUT / "teste_intervalos_confianca.csv")
print(f"bootstrap com {B} reamostragens de {len(df)} folhas de teste")
print(tab.round(3).to_string())
