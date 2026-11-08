"""Auditoria do BRACOL: rotulos, imagens presentes, multilabel, severidade, duplicatas."""
import hashlib, sys
from pathlib import Path
import pandas as pd
from PIL import Image
import imagehash

raiz = Path(sys.argv[1])
csv = pd.read_csv(raiz / "leaf" / "dataset.csv")
imgs = {int(p.stem): p for p in (raiz / "leaf" / "images").glob("*.jpg")}
print("linhas csv:", len(csv), "| imagens no disco:", len(imgs))
csv["existe"] = csv["id"].isin(imgs)
print("ids sem imagem:", (~csv["existe"]).sum())
ok = csv[csv["existe"]].copy()

corrompidas, tamanhos, sha, ph = [], {}, {}, {}
for i in ok["id"]:
    p = imgs[i]
    try:
        with Image.open(p) as im:
            im.load()
            tamanhos[i] = im.size
            ph[i] = imagehash.phash(im)
        sha[i] = hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception as e:
        corrompidas.append((i, str(e)[:60]))
print("corrompidas:", corrompidas)
ok = ok[~ok["id"].isin([c[0] for c in corrompidas])]
print("imagens validas:", len(ok))
print("tamanhos mais comuns:", pd.Series(tamanhos).value_counts().head(5).to_dict())

cols = ["miner", "rust", "phoma", "cercospora"]
print("\ncontagem por rotulo (imagens validas):")
print(ok[cols].sum().to_dict())
print("saudaveis (nenhum rotulo):", int((ok[cols].sum(axis=1) == 0).sum()))
print("n de doencas por folha:", ok[cols].sum(axis=1).value_counts().sort_index().to_dict())
print("combinacoes multilabel:")
print(ok[ok[cols].sum(axis=1) > 1].groupby(cols).size().to_dict())
print("severity:", ok["severity"].value_counts().sort_index().to_dict())
print("predominant_stress:", ok["predominant_stress"].value_counts().sort_index().to_dict())
print("severity por predominant_stress:")
print(pd.crosstab(ok["predominant_stress"], ok["severity"]))

# distribuicao ao longo dos ids (a ordem do zip truncado e lexicografica)
print("\nids ausentes: min/max", csv.loc[~csv['existe'],'id'].min(), csv.loc[~csv['existe'],'id'].max())
print("rotulos nos ids ausentes (csv):", csv.loc[~csv['existe'], cols].sum().to_dict())

# duplicatas
from collections import defaultdict
g = defaultdict(list)
for i, h in sha.items(): g[h].append(i)
dup_exatas = [v for v in g.values() if len(v) > 1]
print("\ngrupos identicos (sha256):", len(dup_exatas), dup_exatas[:5])
ids = sorted(ph)
quase = []
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        if ph[ids[a]] - ph[ids[b]] <= 4:
            quase.append((ids[a], ids[b]))
print("pares quase-duplicados (pHash<=4):", len(quase), quase[:10])
