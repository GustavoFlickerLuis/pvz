"""Baselines sem aprendizado (Etapa 1). Todos têm o método agir(amb) -> ação."""
import random
from functools import partial

import numpy as np

from . import config as C
from .ambiente import ESPERAR, codificar_acao


class Aleatorio:
    """Sorteia entre as ações válidas (esperar incluído)."""

    def __init__(self, semente=0):
        self.rng = random.Random(semente)

    def agir(self, amb):
        return int(self.rng.choice(np.flatnonzero(amb.mascara())))


def _primeira(*regras):
    """Ação da primeira regra que não devolve None (a ação 0 é válida, então nada de `or`)."""
    for regra in regras:
        a = regra()
        if a is not None:
            return a
    return ESPERAR


class Guloso:
    """Gasta assim que puder, na faixa mais ameaçada.

    - Com zumbi no gramado: repetidora ou ervilha na coluna livre mais à esquerda
      da faixa mais ameaçada.
    - Senão (ou se não der): girassóis nas colunas 0 e 1; depois ervilhas nas
      faixas com menos atiradores.
    """

    def agir(self, amb):
        m = amb.mascara()
        return _primeira(lambda: self._defender(amb, m),
                         lambda: self._girassol(m, (0, 1)),
                         lambda: self._ervilha(amb, m))

    def _defender(self, amb, m):
        ameaca = [
            sum(z.vida * (1 + (C.COLUNAS - z.x) / C.COLUNAS) for z in amb.zumbis[f])
            for f in range(C.FAIXAS)
        ]
        if max(ameaca) > 0:
            faixa = int(np.argmax(ameaca))
            for tipo in (C.REPETIDORA, C.ERVILHA):
                for col in range(C.COLUNAS):
                    a = codificar_acao(tipo, faixa, col)
                    if m[a]:
                        return a
        return None

    @staticmethod
    def _girassol(m, colunas):
        for col in colunas:
            for faixa in range(C.FAIXAS):
                a = codificar_acao(C.GIRASSOL, faixa, col)
                if m[a]:
                    return a
        return None

    @staticmethod
    def _ervilha(amb, m):
        atiradores = [
            sum(1 for p in amb.plantas if p.faixa == f and p.tipo in C.TIROS)
            for f in range(C.FAIXAS)
        ]
        for faixa in sorted(range(C.FAIXAS), key=lambda f: atiradores[f]):
            for col in range(C.COLUNAS):
                a = codificar_acao(C.ERVILHA, faixa, col)
                if m[a]:
                    return a
        return None


class Econ(Guloso):
    """Econ(N): primeiro a economia, depois a defesa.

    Enquanto houver menos de N girassóis vivos, girassol tem prioridade total
    (colunas da esquerda primeiro; se não der para plantar agora, espera).
    Com N girassóis, age como o Guloso, mas sem plantar girassóis.
    """

    def __init__(self, n):
        self.n = n

    def agir(self, amb):
        m = amb.mascara()
        if sum(p.tipo == C.GIRASSOL for p in amb.plantas) < self.n:
            return _primeira(lambda: self._girassol(m, range(C.COLUNAS)), lambda: ESPERAR)
        return _primeira(lambda: self._defender(amb, m), lambda: self._ervilha(amb, m))


AGENTES = {
    "aleatorio": Aleatorio,
    "guloso": Guloso,
    **{f"econ{n}": partial(Econ, n) for n in (3, 4, 5, 6, 8)},
}
