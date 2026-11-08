"""Valida as imagens do BRACOL, gera cache 512x256, agrupa quase-duplicatas e divide 70/15/15.

A divisao e estratificada pela combinacao de rotulos (multilabel) e agrupada por
quase-duplicata (pHash), para que copias da mesma foto nunca fiquem em splits diferentes.
"""
import warnings
import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from comum import BRACOL, CACHE, OUT, DOENCAS, LARGURA, ALTURA, SEED

LIMIAR_PHASH = 4


def achar(grupos, i):
    while grupos[i] != i:
        grupos[i] = grupos[grupos[i]]
        i = grupos[i]
    return i


def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    csv = pd.read_csv(BRACOL / "leaf" / "dataset.csv") if (BRACOL / "leaf").exists() else pd.read_csv(BRACOL / "dataset.csv")
    pasta = BRACOL / "images"
    linhas, hashes, descartadas = [], {}, []
    for _, r in csv.iterrows():
        caminho = pasta / f"{int(r['id'])}.jpg"
        if not caminho.exists():
            descartadas.append((int(r["id"]), "arquivo ausente (zip do Mendeley truncado)"))
            continue
        try:
            with Image.open(caminho) as im:
                im.load()
                im = im.convert("RGB")
                hashes[int(r["id"])] = imagehash.phash(im)
                im.resize((LARGURA, ALTURA), Image.LANCZOS).save(CACHE / f"{int(r['id'])}.jpg", quality=95)
        except Exception as e:  # imagem corrompida
            descartadas.append((int(r["id"]), f"corrompida: {str(e)[:50]}"))
            continue
        linhas.append(r)
    df = pd.DataFrame(linhas).reset_index(drop=True)
    df["id"] = df["id"].astype(int)
    print(f"imagens validas: {len(df)} | descartadas: {len(descartadas)} "
          f"({sum('ausente' in d[1] for d in descartadas)} ausentes, "
          f"{sum('corrompida' in d[1] for d in descartadas)} corrompidas)")

    # grupos de quase-duplicatas (componentes conexas, distancia pHash <= limiar)
    ids = df["id"].tolist()
    pai = {i: i for i in ids}
    pares = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if hashes[ids[a]] - hashes[ids[b]] <= LIMIAR_PHASH:
                pares += 1
                pai[achar(pai, ids[a])] = achar(pai, ids[b])
    df["grupo"] = [achar(pai, i) for i in ids]
    print(f"pares quase-duplicados: {pares} | grupos com mais de uma foto: "
          f"{(df.groupby('grupo').size() > 1).sum()}")

    # estratificacao pela combinacao de rotulos
    combo = df[DOENCAS].astype(int).astype(str).agg("".join, axis=1)
    df["split"] = ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
        resto_idx, teste_idx = next(sgkf.split(df, combo, df["grupo"]))
        df.loc[teste_idx, "split"] = "test"
        resto = df.iloc[resto_idx]
        sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
        tr, va = next(sgkf2.split(resto, combo.iloc[resto_idx], resto["grupo"]))
        df.loc[resto.index[tr], "split"] = "train"
        df.loc[resto.index[va], "split"] = "val"

    # verificacoes
    g = df.groupby("grupo")["split"].nunique()
    assert (g == 1).all(), "grupo de quase-duplicatas dividido entre splits!"
    print("\nquase-duplicatas atravessando splits: 0 (verificado)")
    print("\nimagens por split:", df["split"].value_counts().to_dict())
    df["saudavel"] = (df[DOENCAS].sum(axis=1) == 0).astype(int)
    print(df.groupby("split")[DOENCAS + ["saudavel"]].sum())
    print("\nfolhas com 2+ doencas por split:")
    print(df.assign(multi=(df[DOENCAS].sum(axis=1) > 1)).groupby("split")["multi"].sum())
    print("\nseveridade por split:")
    print(pd.crosstab(df["split"], df["severity"]))

    df[["id", *DOENCAS, "severity", "grupo", "split"]].to_csv(OUT / "metadados.csv", index=False)
    pd.DataFrame(descartadas, columns=["id", "motivo"]).to_csv(OUT / "descartadas.csv", index=False)


if __name__ == "__main__":
    main()
