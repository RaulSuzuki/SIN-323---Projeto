from pathlib import Path

# ---------- caminhos ----------
RAIZ = Path(__file__).resolve().parent
CSV = RAIZ / "data" / "MetroPT3AirCompressor.csv"
PARQUET = RAIZ / "data" / "processed" / "metropt3.parquet"
MODELOS = RAIZ / "models"
DB = RAIZ / "db" / "monitoramento.sqlite"
FIGURAS = RAIZ / "reports" / "figuras"

# ---------- falhas reais (relatórios da empresa) ----------
FALHAS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),   # confirmado no gráfico: termina ~06:00
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]

# manutenções (a de maio é interpretação: a UCI escreve "30Apr")
MANUTENCOES = ["2020-05-30 12:00", "2020-06-08 16:00", "2020-07-16 00:00"]

# ---------- cenários da demo ----------
CENARIOS = {
    "falha_29_maio": ("2020-05-27", "2020-05-31"),
    "falha_05_junho": ("2020-06-03", "2020-06-09"),
    "normal_controle": ("2020-03-10", "2020-03-14"),
}

# ---------- parâmetros ----------
TREINO_FIM = "2020-03-01"
SEG_POR_LINHA = 10            # o CSV tem uma linha a cada ~10 s
LACUNA_MAX = "1min"           # acima disso, considera lacuna nos dados
MIN_COBERTURA = 0.8           # janela com menos de 80% das amostras vira NaN

# PROVISÓRIOS: ajustar olhando o histograma de Motor_current (figura 01)
LIMIAR_SEM_CARGA = 1.0
LIMIAR_CARGA = 5.5