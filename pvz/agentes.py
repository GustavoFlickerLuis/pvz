"""Baselines sem aprendizado (Etapa 1). Todos têm o método agir(amb) -> ação."""
import random

import numpy as np

from . import config as C
from .ambiente import ESPERAR, codificar_acao


class Aleatorio:
    """Sorteia entre as ações válidas (esperar incluído)."""

    def __init__(self, semente=0):
        self.rng = random.Random(semente)

    def agir(self, amb):
        return int(self.rng.choice(np.flatnonzero(amb.mascara())))


class Guloso:
    """Gasta assim que puder, na faixa mais ameaçada.

    - Com zumbi no gramado: repetidora ou ervilha na coluna livre mais à esquerda
      da faixa mais ameaçada.
    - Senão (ou se não der): girassóis nas colunas 0 e 1; depois ervilhas nas
      faixas com menos atiradores.
    """

    def agir(self, amb):
        m = amb.mascara()

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

        for col in (0, 1):
            for faixa in range(C.FAIXAS):
                a = codificar_acao(C.GIRASSOL, faixa, col)
                if m[a]:
                    return a

        atiradores = [
            sum(1 for p in amb.plantas if p.faixa == f and p.tipo in C.TIROS)
            for f in range(C.FAIXAS)
        ]
        for faixa in sorted(range(C.FAIXAS), key=lambda f: atiradores[f]):
            for col in range(C.COLUNAS):
                a = codificar_acao(C.ERVILHA, faixa, col)
                if m[a]:
                    return a
        return ESPERAR


AGENTES = {
    "aleatorio": Aleatorio,
    "guloso": Guloso,
}
