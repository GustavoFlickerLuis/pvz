"""Hordas e fases (seção 2.6).

Uma fase é uma sequência de segmentos (início, meio, fim). Cada classe de segmento
define O QUE entra e QUANDO; a semente sorteia só EM QUAL FAIXA.

As classes abaixo são um primeiro rascunho, para ter o que testar.
"""
import itertools
import random
from dataclasses import dataclass
from functools import cached_property

from . import config as C
from .config import BLINDADO, COMUM, RAPIDO


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
def _fila(rng, escala, n, inicio, intervalo, padrao, faixa=None):
    """n zumbis a partir do tick `inicio`, um a cada `intervalo` ticks.

    `padrao` é repetido em ciclo (a composição não depende da semente).
    `escala` (dificuldade) multiplica o número de zumbis e comprime o intervalo,
    para o segmento manter a mesma duração.
    """
    n = max(1, round(n * escala))
    intervalo = max(1, round(intervalo / escala))
    return [
        (inicio + i * intervalo,
         padrao[i % len(padrao)],
         faixa if faixa is not None else rng.randrange(C.FAIXAS))
        for i in range(n)
    ]


# Cada classe: (rng, escala) -> (lista de (tick relativo, tipo, faixa), duração do segmento)
INICIO = {
    "calmo":          lambda rng, e: (_fila(rng, e, 4, 400, 120, [COMUM]), 900),
    "cedo_apressado": lambda rng, e: (_fila(rng, e, 5, 200, 100, [COMUM]), 900),
    "so_rapidos":     lambda rng, e: (_fila(rng, e, 4, 350, 130, [RAPIDO]), 900),
}


def _uma_faixa_pesada(rng, e):
    faixa = rng.randrange(C.FAIXAS)
    return (_fila(rng, e, 8, 0, 120, [COMUM], faixa=faixa)
            + _fila(rng, e, 2, 300, 400, [COMUM])), 1200


MEIO = {
    "misto":            lambda rng, e: (_fila(rng, e, 10, 0, 110, [COMUM, COMUM, RAPIDO, COMUM, BLINDADO]), 1200),
    "uma_faixa_pesada": _uma_faixa_pesada,
    "rapidos_em_massa": lambda rng, e: (_fila(rng, e, 10, 100, 100, [RAPIDO]), 1200),
}

FIM = {
    "blindados":  lambda rng, e: (_fila(rng, e, 12, 0, 60, [BLINDADO, COMUM, COMUM]), 900),
    "tudo_junto": lambda rng, e: (_fila(rng, e, 15, 100, 30, [COMUM, RAPIDO, COMUM, BLINDADO, COMUM]), 900),
    "duas_ondas": lambda rng, e: (_fila(rng, e, 8, 0, 35, [COMUM, COMUM, RAPIDO, BLINDADO])
                                  + _fila(rng, e, 8, 500, 35, [COMUM, COMUM, RAPIDO, BLINDADO]), 900),
}

SEGMENTOS = (INICIO, MEIO, FIM)


# ---------------------------------------------------------------- fases
def montar_fase(classes, semente=0, dificuldade=1.0):
    """Monta uma fase a partir de 1 a 3 classes, na ordem início, meio, fim.

    `classes` pode ser uma tupla ("calmo", "misto", "blindados") ou uma string
    "calmo,misto,blindados". Fases com menos segmentos servem para o currículo.
    """
    if isinstance(classes, str):
        classes = [c.strip() for c in classes.replace("+", ",").split(",") if c.strip()]
    if not 1 <= len(classes) <= 3:
        raise ValueError("uma fase tem de 1 a 3 segmentos")
    rng = random.Random(semente)
    spawns, inicio = [], 0
    for i, (nome, tabela) in enumerate(zip(classes, SEGMENTOS)):
        if nome not in tabela:
            raise ValueError(f"segmento {i} não tem a classe {nome!r}; opções: {sorted(tabela)}")
        lista, duracao = tabela[nome](rng, dificuldade)
        spawns += [Spawn(inicio + t, tipo, faixa, i) for t, tipo, faixa in lista]
        inicio += duracao
    spawns.sort(key=lambda s: s.tick)
    return Fase("+".join(classes), tuple(spawns), inicio)


def fase_manual(spawns, nome="manual"):
    """Fase escrita à mão: lista de (tick, tipo, faixa). Útil para testes."""
    lista = tuple(sorted((Spawn(t, tipo, faixa, 0) for t, tipo, faixa in spawns), key=lambda s: s.tick))
    return Fase(nome, lista, max((s.tick for s in lista), default=0) + 1)


def todas_as_fases():
    """As 27 combinações de classes (início × meio × fim)."""
    return list(itertools.product(INICIO, MEIO, FIM))
