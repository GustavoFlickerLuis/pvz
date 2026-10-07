"""Hordas e fases (seção 2.6).

Uma fase é uma sequência de segmentos (início, meio, fim). Cada classe de segmento
é uma DISTRIBUIÇÃO com "personalidade" própria (o que costuma entrar, em que ritmo,
onde); a semente sorteia a partida concreta: quantos zumbis, quando o segmento
começa, os intervalos entre zumbis, a composição e as faixas.

As classes abaixo ainda são um rascunho, para ter o que testar.
"""
import itertools
import random
from dataclasses import dataclass
from functools import cached_property

from . import config as C
from .config import BLINDADO, COMUM, RAPIDO

# Intensidade de cada segmento (início, meio, fim), multiplicada pela `dificuldade`:
# multiplica o número de zumbis e comprime o intervalo (o segmento mantém a duração).
PESO_SEGMENTO = (1.0, 2.0, 2.25)

# Variação sorteada pela semente em cada fila de zumbis
VAR_QUANTIDADE = 0.25   # ±25% no número de zumbis
VAR_INICIO = 100        # ±100 ticks no começo da fila
VAR_INTERVALO = 0.30    # ±30% em cada intervalo, individualmente


@dataclass(frozen=True)
class Spawn:
    tick: int
    tipo: int
    faixa: int
    segmento: int


@dataclass
class Fase:
    nome: str
    spawns: tuple   # de Spawn, ordenados por tick
    duracao: int    # soma das durações dos segmentos (usado no canal de progresso)

    @cached_property
    def vida_total(self):
        return sum(C.ZUMBIS[s.tipo].vida for s in self.spawns)


# ---------------------------------------------------------------- segmentos
@dataclass(frozen=True)
class Fila:
    """Valores médios de uma fila de zumbis, antes do sorteio e da escala.

    `padrao` dá as proporções da composição (cada zumbi é sorteado dele).
    `faixa_unica`: todos na mesma faixa, sorteada uma vez por segmento.
    """
    n: int
    inicio: int
    intervalo: int
    padrao: tuple
    faixa_unica: bool = False


@dataclass(frozen=True)
class Classe:
    duracao: int
    filas: tuple


def _sortear_fila(rng, fila, escala, duracao, faixa_unica):
    """Lista de (tick relativo ao segmento, tipo, faixa), toda dentro de [0, duracao)."""
    n = max(1, round(fila.n * escala * rng.uniform(1 - VAR_QUANTIDADE, 1 + VAR_QUANTIDADE)))
    intervalo = fila.intervalo / escala
    offsets, t = [], 0.0
    for _ in range(n):
        offsets.append(t)
        t += intervalo * rng.uniform(1 - VAR_INTERVALO, 1 + VAR_INTERVALO)
    ultimo = duracao - 1
    if offsets[-1] > ultimo:                       # fila longa demais: comprime
        offsets = [o * ultimo / offsets[-1] for o in offsets]
    inicio = fila.inicio + rng.randint(-VAR_INICIO, VAR_INICIO)
    inicio = min(max(0, inicio), ultimo - offsets[-1])   # não sai do segmento
    return [
        (int(inicio + o),
         rng.choice(fila.padrao),
         faixa_unica if fila.faixa_unica else rng.randrange(C.FAIXAS))
        for o in offsets
    ]


def sortear_segmento(rng, classe, escala):
    faixa_unica = rng.randrange(C.FAIXAS)
    lista = []
    for fila in classe.filas:
        lista += _sortear_fila(rng, fila, escala, classe.duracao, faixa_unica)
    return lista


INICIO = {
    "calmo":          Classe(900, (Fila(4, 400, 120, (COMUM,)),)),
    "cedo_apressado": Classe(900, (Fila(5, 200, 100, (COMUM,)),)),
    "so_rapidos":     Classe(900, (Fila(4, 350, 130, (RAPIDO,)),)),
}

MEIO = {
    "misto":            Classe(1200, (Fila(10, 0, 110, (COMUM, COMUM, RAPIDO, COMUM, BLINDADO)),)),
    "uma_faixa_pesada": Classe(1200, (Fila(8, 0, 120, (COMUM,), faixa_unica=True),
                                      Fila(2, 300, 400, (COMUM,)))),
    "rapidos_em_massa": Classe(1200, (Fila(10, 100, 100, (RAPIDO,)),)),
}

FIM = {
    "blindados":  Classe(900, (Fila(12, 0, 60, (BLINDADO, COMUM, COMUM)),)),
    "tudo_junto": Classe(900, (Fila(15, 100, 30, (COMUM, RAPIDO, COMUM, BLINDADO, COMUM)),)),
    "duas_ondas": Classe(900, (Fila(8, 0, 35, (COMUM, COMUM, RAPIDO, BLINDADO)),
                               Fila(8, 500, 35, (COMUM, COMUM, RAPIDO, BLINDADO)))),
}

SEGMENTOS = (INICIO, MEIO, FIM)


# ---------------------------------------------------------------- fases
def montar_fase(classes, semente=0, dificuldade=1.0, pesos=PESO_SEGMENTO):
    """Monta uma fase a partir de 1 a 3 classes, na ordem início, meio, fim.

    `classes` pode ser uma tupla ("calmo", "misto", "blindados") ou uma string
    "calmo,misto,blindados". Fases com menos segmentos servem para o currículo.
    Determinística: mesmas classes + semente + dificuldade + pesos = mesma fase.
    """
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.replace("+", ",").split(",") if c.strip()]
    if not 1 <= len(classes) <= 3:
        raise ValueError("uma fase tem de 1 a 3 segmentos")
    # O rng depende também das classes: sem isso, duas fases cujos segmentos anteriores
    # consomem o mesmo número de sorteios teriam o resto da horda idêntico.
    # (Semente em string é determinística, não depende de PYTHONHASHSEED.)
    rng = random.Random(f"{semente}|{'+'.join(classes)}")
    spawns, inicio = [], 0
    for i, (nome, tabela) in enumerate(zip(classes, SEGMENTOS)):
        if nome not in tabela:
            raise ValueError(f"segmento {i} não tem a classe {nome!r}; opções: {sorted(tabela)}")
        classe = tabela[nome]
        lista = sortear_segmento(rng, classe, dificuldade * pesos[i])
        spawns += [Spawn(inicio + t, tipo, faixa, i) for t, tipo, faixa in lista]
        inicio += classe.duracao
    spawns.sort(key=lambda s: s.tick)
    return Fase("+".join(classes), tuple(spawns), inicio)


def fase_manual(spawns, nome="manual"):
    """Fase escrita à mão: lista de (tick, tipo, faixa). Útil para testes."""
    lista = tuple(sorted((Spawn(t, tipo, faixa, 0) for t, tipo, faixa in spawns), key=lambda s: s.tick))
    return Fase(nome, lista, max((s.tick for s in lista), default=0) + 1)


def todas_as_fases():
    """As 27 combinações de classes (início × meio × fim)."""
    return list(itertools.product(INICIO, MEIO, FIM))
