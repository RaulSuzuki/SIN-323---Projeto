import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from config import FALHAS, FIGURAS, LIMIAR_CARGA, LIMIAR_SEM_CARGA, MANUTENCOES
from src.data import carregar_dados
from src.features import features_horarias


def marcar(ax):
    for ini, fim in FALHAS:
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fim), color="red", alpha=0.25)
    for m in MANUTENCOES:
        ax.axvline(pd.Timestamp(m), color="green", ls="--", lw=1)


def salvar(fig, nome):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / nome, dpi=120, bbox_inches="tight")


def hist_corrente(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["Motor_current"].dropna(), bins=200)
    ax.set_yscale("log")
    ax.axvline(LIMIAR_SEM_CARGA, color="orange", ls="--", label=f"limiar sem carga ({LIMIAR_SEM_CARGA})")
    ax.axvline(LIMIAR_CARGA, color="red", ls="--", label=f"limiar carga ({LIMIAR_CARGA})")
    ax.set_title("Distribuição da corrente do motor (A)")
    ax.legend()
    salvar(fig, "01_hist_corrente.png")


def visao_geral(feats):
    cols = [("duty_cycle", "Duty cycle (fração sob carga)"),
            ("transicoes", "Transições liga/carga por hora"),
            ("tp3_media", "TP3 média (bar)"),
            ("oleo_media", "Temperatura do óleo média (°C)")]
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 9), sharex=True)
    for ax, (c, titulo) in zip(axes, cols):
        ax.plot(feats.index, feats[c], lw=0.6)
        ax.set_title(titulo, fontsize=10)
        marcar(ax)
    fig.suptitle("Visão geral (vermelho = falha, verde tracejado = manutenção)")
    fig.tight_layout()
    salvar(fig, "02_visao_geral.png")


def zooms(feats):
    for i, (ini, fim) in enumerate(FALHAS, start=1):
        a = pd.Timestamp(ini) - pd.Timedelta(days=3)
        b = pd.Timestamp(fim) + pd.Timedelta(days=1)
        sub = feats.loc[a:b]
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for ax, c in zip(axes, ["duty_cycle", "transicoes", "tp3_media"]):
            ax.plot(sub.index, sub[c], marker=".", lw=0.8)
            ax.set_title(c, fontsize=10)
            marcar(ax)
        fig.suptitle(f"Falha {i}: 3 dias antes até 1 dia depois")
        fig.tight_layout()
        salvar(fig, f"03_zoom_falha_{i}.png")


def main():
    df = carregar_dados()
    feats = features_horarias(df)
    hist_corrente(df)
    visao_geral(feats)
    zooms(feats)
    print(f"Figuras salvas em: {FIGURAS}")
    plt.show()


if __name__ == "__main__":
    main()