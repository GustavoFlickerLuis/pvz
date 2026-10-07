"""Todos os números do jogo, num lugar só.

São os VALORES INICIAIS da Parte 2 do CONCEITO. A calibração é feita aqui:
o agente aleatório deve perder quase sempre e o guloso deve vencer as fases médias.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------- gramado e tempo
FAIXAS = 5
COLUNAS = 9
K_TICKS = 10          # o agente decide a cada k ticks (seção 2.3)
MAX_TICKS = 20_000    # trava de segurança: partida que passa disso conta como derrota

# ---------------------------------------------------------------- sol
SOL_INICIAL = 100
SOL_CEU = 25
INTERVALO_CEU = 240
SOL_GIRASSOL = 25
INTERVALO_GIRASSOL = 120


# ---------------------------------------------------------------- plantas
@dataclass(frozen=True)
class TipoPlanta:
    nome: str
    simbolo: str   # usado na visualização e no modo manual
    custo: int
    recarga: int
    vida: int


GIRASSOL, ERVILHA, REPETIDORA, BATATA, MINA = range(5)

PLANTAS = (
    TipoPlanta("girassol",   "G", custo=50,  recarga=50,  vida=300),
    TipoPlanta("ervilha",    "E", custo=100, recarga=50,  vida=300),
    TipoPlanta("repetidora", "R", custo=200, recarga=50,  vida=300),
    TipoPlanta("batata",     "B", custo=50,  recarga=200, vida=4000),
    TipoPlanta("mina",       "M", custo=25,  recarga=200, vida=300),
)

DANO_TIRO = 20
INTERVALO_TIRO = 15
TIROS = {ERVILHA: 1, REPETIDORA: 2}
MINA_ARMAR = 150
MINA_DANO = 1800


# ---------------------------------------------------------------- zumbis
@dataclass(frozen=True)
class TipoZumbi:
    nome: str
    simbolo: str
    vida: int
    ticks_por_coluna: int
    mordida: int


COMUM, BLINDADO, RAPIDO = range(3)

ZUMBIS = (
    TipoZumbi("comum",    "z", vida=200,  ticks_por_coluna=50, mordida=5),
    TipoZumbi("blindado", "b", vida=1100, ticks_por_coluna=70, mordida=5),
    TipoZumbi("rapido",   "r", vida=150,  ticks_por_coluna=25, mordida=5),
)
