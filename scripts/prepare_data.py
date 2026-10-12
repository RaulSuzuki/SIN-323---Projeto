import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # permite importar config e src

from config import CSV, PARQUET
from src.data import carregar_csv, lacunas, salvar_parquet


def main():
    if not CSV.exists():
        raise SystemExit(f"CSV não encontrado em: {CSV}\nMova o arquivo para a pasta data/.")

    df = carregar_csv()
    print(f"Linhas: {len(df):,}")
    print(f"Período: {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"Tipo do timestamp: {df['timestamp'].dtype}")
    print(f"Passo mediano entre amostras: {df['timestamp'].diff().median()}")
    print(f"Valores nulos: {int(df.isna().sum().sum())}")
    print(f"Colunas: {list(df.columns)}")

    salvar_parquet(df)
    print(f"\nParquet salvo em: {PARQUET}")

    g = lacunas(df)
    print(f"\nLacunas (> 1 min): {len(g)}")
    if len(g):
        print(f"Tempo total sem dados em lacunas: {g['duracao'].sum()}")
        print(g.sort_values("duracao", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()