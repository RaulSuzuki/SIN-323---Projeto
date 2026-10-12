import pandas as pd

from config import CENARIOS, CSV, LACUNA_MAX, PARQUET


def carregar_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV, index_col=0, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def salvar_parquet(df: pd.DataFrame) -> None:
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET, index=False)


def carregar_dados() -> pd.DataFrame:
    """Lê do parquet (rápido); se não existir, gera a partir do CSV."""
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)
    df = carregar_csv()
    salvar_parquet(df)
    return df


def recortar(df: pd.DataFrame, ini: str, fim: str) -> pd.DataFrame:
    m = (df["timestamp"] >= ini) & (df["timestamp"] < fim)
    return df.loc[m].reset_index(drop=True)


def cenario(nome: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = carregar_dados() if df is None else df
    ini, fim = CENARIOS[nome]
    return recortar(df, ini, fim)


def lacunas(df: pd.DataFrame, limite: str = LACUNA_MAX) -> pd.DataFrame:
    """Lista os trechos sem dados (diferença entre timestamps acima do limite)."""
    d = df["timestamp"].diff()
    m = d > pd.Timedelta(limite)
    return pd.DataFrame({
        "inicio": df["timestamp"].shift()[m],
        "fim": df["timestamp"][m],
        "duracao": d[m],
    }).reset_index(drop=True)