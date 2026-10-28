"""Laboratório local do PPO (ver LAB.md na raiz da cópia)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_LAB = RAIZ / "lab"
PARAR = PASTA_LAB / "PARAR"          # o arquivo de parada do lab (nunca o rl/PARAR oficial)
LOGS = RAIZ / "rl" / "logs"
