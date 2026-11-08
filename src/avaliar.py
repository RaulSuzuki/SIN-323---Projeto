"""Calibra no conjunto de validacao e avalia no teste: multilabel, severidade, recusa e Grad-CAM."""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_fscore_support,
                             cohen_kappa_score, confusion_matrix)
from torch.utils.data import DataLoader

from comum import CACHE, DOENCAS, NOMES_PT, MODELOS, OUT, SEVERIDADE_PT, MEDIA, DESVIO, dispositivo
from gradcam import gradcam, mascara_folha, massa_na_folha
from modelo import Folhas, carregar


def prever(modelo, split, disp):
    dl = DataLoader(Folhas(split), batch_size=16, num_workers=0)
    L, S, Y, SY, ID = [], [], [], [], []
    with torch.no_grad():
        for x, y, s, i in dl:
            x = x.to(disp)
            a, b = modelo(x)
            a2, b2 = modelo(torch.flip(x, dims=[3]))  # TTA: espelhamento horizontal
            L.append(((a + a2) / 2).cpu()); S.append(((b + b2) / 2).cpu()); Y.append(y); SY.append(s); ID.append(i)
    return torch.cat(L), torch.cat(S), torch.cat(Y).numpy(), torch.cat(SY).numpy(), torch.cat(ID).numpy()


def ajustar_temperatura(logits, alvo, multilabel):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def f():
        opt.zero_grad()
        z = logits / T
        perda = F.binary_cross_entropy_with_logits(z, alvo) if multilabel else F.cross_entropy(z, alvo)
        perda.backward(); return perda
    opt.step(f)
    return float(T.clamp(min=0.05).item())


def ece(prob, y, bins=10):
    prob, y = prob.ravel(), y.ravel()
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (prob > lo) & (prob <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(e)


def limiares_f1(p, y):
    ths = []
    for k in range(p.shape[1]):
        grade = np.linspace(0.1, 0.9, 33)
        f1s = [precision_recall_fscore_support(y[:, k], p[:, k] >= t, average="binary", zero_division=0)[2] for t in grade]
        ths.append(float(grade[int(np.argmax(f1s))]))
    return ths


def incerteza(p, ps):
    """Maior ambiguidade entre as doencas (prob perto do limiar) combinada com a entropia da severidade."""
    amb = 1 - np.abs(2 * p - 1).min(axis=1)  # 1 = alguma doenca com prob ~0.5
    ent = -(ps * np.log(ps + 1e-9)).sum(1) / np.log(ps.shape[1])
    return np.maximum(amb, ent)


def main():
    disp = dispositivo()
    modelo = carregar(MODELOS / "melhor_modelo.pt", disp)
    Lv, Sv, Yv, SYv, _ = prever(modelo, "val", disp)
    Lt, St, Yt, SYt, IDt = prever(modelo, "test", disp)

    # --- calibracao (so na validacao) ---
    Td = ajustar_temperatura(Lv, torch.tensor(Yv, dtype=torch.float32), True)
    Ts = ajustar_temperatura(Sv, torch.tensor(SYv), False)
    pv, pt = torch.sigmoid(Lv / Td).numpy(), torch.sigmoid(Lt / Td).numpy()
    pt_raw = torch.sigmoid(Lt).numpy()
    psv, pst = F.softmax(Sv / Ts, 1).numpy(), F.softmax(St / Ts, 1).numpy()
    ths = limiares_f1(pv, Yv)
    res = {"temperatura_doenca": Td, "temperatura_severidade": Ts, "limiares": dict(zip(DOENCAS, ths)),
           "ECE_teste_sem_calibrar": ece(pt_raw, Yt), "ECE_teste_calibrado": ece(pt, Yt)}

    # --- multilabel no teste ---
    pred = (pt >= np.array(ths)).astype(int)
    linhas = []
    for k, d in enumerate(DOENCAS):
        p, r, f, _ = precision_recall_fscore_support(Yt[:, k], pred[:, k], average="binary", zero_division=0)
        linhas.append({"doenca": NOMES_PT[d], "n_positivos_teste": int(Yt[:, k].sum()), "precisao": p,
                       "recall": r, "F1": f, "AP": average_precision_score(Yt[:, k], pt[:, k]),
                       "AUC": roc_auc_score(Yt[:, k], pt[:, k])})
    tab = pd.DataFrame(linhas)
    tab.to_csv(OUT / "teste_por_doenca.csv", index=False)
    print(tab.round(3).to_string(index=False))
    saud_real = Yt.sum(1) == 0
    saud_pred = pred.sum(1) == 0
    res.update({
        "mAP": float(tab["AP"].mean()), "F1_macro": float(tab["F1"].mean()),
        "exact_match": float((pred == Yt).all(1).mean()),
        "hamming_acc": float((pred == Yt).mean()),
        "saudavel_recall": float((saud_pred & saud_real).sum() / max(saud_real.sum(), 1)),
        "saudavel_precisao": float((saud_pred & saud_real).sum() / max(saud_pred.sum(), 1)),
        "n_teste": int(len(Yt)),
        "n_teste_multidoenca": int((Yt.sum(1) > 1).sum()),
        "exact_match_so_multidoenca": float((pred[Yt.sum(1) > 1] == Yt[Yt.sum(1) > 1]).all(1).mean()),
    })

    # confusoes: doenca real perdida -> o que o modelo previu no lugar (em folhas de rotulo unico)
    pares = {}
    unica = Yt.sum(1) == 1
    for i in np.where(unica)[0]:
        real = DOENCAS[int(Yt[i].argmax())]
        for k, d in enumerate(DOENCAS):
            if pred[i, k] and d != real:
                pares[(NOMES_PT[real], NOMES_PT[d])] = pares.get((NOMES_PT[real], NOMES_PT[d]), 0) + 1
    top = sorted(pares.items(), key=lambda kv: -kv[1])
    res["confusoes_mais_comuns(real->previsto a mais)"] = [{"real": a, "previsto": b, "n": n} for (a, b), n in top[:6]]

    # --- severidade ---
    sp = pst.argmax(1)
    res["severidade"] = {"acuracia": float((sp == SYt).mean()), "MAE_niveis": float(np.abs(sp - SYt).mean()),
                         "dentro_de_1_nivel": float((np.abs(sp - SYt) <= 1).mean()),
                         "kappa_quadratico": float(cohen_kappa_score(SYt, sp, weights="quadratic")),
                         "baseline_classe_majoritaria": float((SYt == np.bincount(SYt).argmax()).mean())}
    cm = confusion_matrix(SYt, sp, labels=range(5))
    pd.DataFrame(cm, index=[f"real_{s}" for s in SEVERIDADE_PT], columns=[f"prev_{s}" for s in SEVERIDADE_PT]
                 ).to_csv(OUT / "teste_severidade_confusao.csv")

    # --- recusa por baixa confianca (limiar escolhido na validacao para ~80% de cobertura) ---
    uv, ut = incerteza(pv, psv), incerteza(pt, pst)
    tau = float(np.quantile(uv, 0.80))
    res["recusa_tau"] = tau
    curva = []
    ok = (pred == Yt).all(1)
    for cob in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        lim = np.quantile(ut, cob) if cob < 1 else ut.max() + 1
        m = ut <= lim
        curva.append({"cobertura_alvo": cob, "cobertura_real": float(m.mean()),
                      "exact_match_aceitas": float(ok[m].mean()), "n_aceitas": int(m.sum())})
    pd.DataFrame(curva).to_csv(OUT / "teste_recusa.csv", index=False)
    m80 = ut <= tau
    res["recusa_no_teste"] = {"cobertura": float(m80.mean()), "exact_match_aceitas": float(ok[m80].mean()),
                              "exact_match_recusadas": float(ok[~m80].mean()) if (~m80).any() else None}

    # --- Grad-CAM: fracao da atencao sobre a folha (proxy contra o atalho do fundo) ---
    fr_cam, fr_area = [], []
    amostra = []
    tf = Folhas("test").tf
    for n, i in enumerate(IDt):
        for k in range(len(DOENCAS)):
            if pred[n, k] and Yt[n, k]:
                img = Image.open(CACHE / f"{int(i)}.jpg").convert("RGB")
                x = tf(img).unsqueeze(0).to(disp)
                cam = gradcam(modelo, x, k)
                folha = mascara_folha(np.array(img))
                a, b = massa_na_folha(cam, folha)
                fr_cam.append(a); fr_area.append(b)
                if len(amostra) < 8:
                    amostra.append((img, cam, DOENCAS[k], int(i)))
    res["gradcam_proxy_fundo"] = {"n": len(fr_cam), "massa_media_na_folha": float(np.mean(fr_cam)),
                                  "area_media_da_folha": float(np.mean(fr_area)),
                                  "razao_massa_sobre_area": float(np.mean(np.array(fr_cam) / np.maximum(fr_area, 1e-6)))}
    fig, ax = plt.subplots(2, 4, figsize=(16, 5))
    for a, (img, cam, d, i) in zip(ax.ravel(), amostra):
        a.imshow(img); a.imshow(cam, cmap="jet", alpha=0.45); a.set_title(f"id {i}: {NOMES_PT[d]}", fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "gradcam_exemplos.png", dpi=110); plt.close()

    # salva predicoes do teste e a calibracao (temperaturas, limiares, recusa)
    pd.DataFrame({"id": IDt, **{f"p_{d}": pt[:, k] for k, d in enumerate(DOENCAS)},
                  **{f"y_{d}": Yt[:, k] for k, d in enumerate(DOENCAS)},
                  **{f"ps_{j}": pst[:, j] for j in range(5)}, "sev_real": SYt, "incerteza": ut}
                 ).to_csv(OUT / "predicoes_teste.csv", index=False)
    json.dump({"Td": Td, "Ts": Ts, "limiares": ths, "tau": tau}, open(MODELOS / "calibracao.json", "w"), indent=2)
    json.dump(res, open(OUT / "resultados_teste.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
