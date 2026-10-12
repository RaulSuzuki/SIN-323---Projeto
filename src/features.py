import numpy as np
import pandas as pd

from config import LIMIAR_CARGA, MIN_COBERTURA, SEG_POR_LINHA


def features_horarias(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Atributos agregados por janela. Janelas com poucos dados viram NaN
    (senão uma lacuna pareceria 'zero transições', igual a uma falha)."""
    d = df.set_index("timestamp")
    carga = (d["Motor_current"] > LIMIAR_CARGA).astype(int)
    mudou = carga.diff().abs().fillna(0)
    r = d.resample(freq)

    out = pd.DataFrame({
        "duty_cycle": carga.resample(freq).mean(),
        "transicoes": mudou.resample(freq).sum(),
        "tp3_media": r["TP3"].mean(),
        "tp3_std": r["TP3"].std(),
        "corrente_media": r["Motor_current"].mean(),
        "oleo_media": r["Oil_temperature"].mean(),
        "n_amostras": r["Motor_current"].count(),
    })
    esperado = pd.Timedelta(freq).total_seconds() / SEG_POR_LINHA
    out["cobertura"] = out["n_amostras"] / esperado

    valores = out.columns.difference(["n_amostras", "cobertura"])
    out.loc[out["cobertura"] < MIN_COBERTURA, valores] = np.nan
    return out