"""Avaliação da política gulosa (ε = 0, com máscara) e referências (Econ5, Guloso).

    python -m rl.avaliar referencias     # Econ5 e Guloso em FASES_TREINO × 300–319 → rl/referencias.csv

As partidas de uma avaliação andam juntas (uma inferência em lote por decisão). O código
da partida só usa reset/step/observar/mascara; vitória e dano saem das recompensas.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

from desafio3.modelo import formato_do_modelo
from pvz import config as C
from pvz.agentes import Econ, Guloso
from pvz.ambiente import Ambiente
from pvz.divisao import FASES_TESTE, FASES_TREINO

RAIZ = Path(__file__).parent
REFERENCIAS = RAIZ / "referencias.csv"


def partidas(sementes, fases=FASES_TREINO):
    assert not set(map(tuple, fases)) & set(FASES_TESTE), "nunca avaliar nas fases de teste"
    return [(f, s) for f in fases for s in sementes]


def _resumo(lista):
    """lista de (fase, semente, venceu, dano, decisões) → dicionário."""
    return {"partidas": len(lista), "vitorias": float(np.mean([l[2] for l in lista])),
            "dano": float(np.mean([l[3] for l in lista])),
            "decisoes": float(np.mean([l[4] for l in lista])), "detalhe": lista}


@torch.no_grad()
def avaliar_rede(modelo, lista, dev, dificuldade=1.0, k=C.K_TICKS, formato=None):
    """Política gulosa em todas as partidas de `lista`, em paralelo, com k ticks por decisão."""
    modelo.eval()
    formato = formato or formato_do_modelo(modelo)   # a observação no formato que a rede espera
    ambs, obs = [], []
    for fase, semente in lista:
        a = Ambiente(k=k, formato=formato)
        obs.append(a.reset(fase, semente, dificuldade))
        ambs.append(a)
    vivos = list(range(len(ambs)))
    dano = [0.0] * len(ambs)
    decisoes = [0] * len(ambs)
    venceu = [False] * len(ambs)
    while vivos:
        x = torch.from_numpy(np.stack([obs[i] for i in vivos])).to(dev)
        m = torch.from_numpy(np.stack([ambs[i].mascara() for i in vivos])).to(dev)
        acoes = modelo(x).masked_fill(~m, -math.inf).argmax(1).tolist()
        seguem = []
        for i, a in zip(vivos, acoes):
            obs[i], r, fim = ambs[i].step(a)
            decisoes[i] += 1
            dano[i] += r
            if fim:
                venceu[i] = r > 0
                dano[i] -= 1.0 if venceu[i] else -1.0   # tira o ±1 do fim
            else:
                seguem.append(i)
        vivos = seguem
    modelo.train()
    return _resumo([(" ".join(f) if not isinstance(f, str) else f, s, venceu[i], dano[i], decisoes[i])
                    for i, (f, s) in enumerate(lista)])


def avaliar_agente(agente, lista, dificuldade=1.0, k=C.K_TICKS):
    res = []
    for fase, semente in lista:
        amb = Ambiente(k=k)
        amb.reset(fase, semente, dificuldade)
        dano, n, fim = 0.0, 0, False
        while not fim:
            _, r, fim = amb.step(agente.agir(amb))
            dano += r
            n += 1
        v = r > 0
        res.append((" ".join(fase), semente, v, dano - (1.0 if v else -1.0), n))
    return _resumo(res)


def referencias():
    lista = partidas(range(300, 320))
    linhas = []
    for nome, ag in (("Econ5", Econ(5)), ("Guloso", Guloso())):
        r = avaliar_agente(ag, lista)
        print(f"{nome}: vitórias {100 * r['vitorias']:.1f}% em {r['partidas']} partidas, "
              f"dano {r['dano']:.3f}, decisões {r['decisoes']:.0f}")
        linhas.append({"agente": nome, "partidas": r["partidas"], "vitorias": r["vitorias"],
                       "dano": r["dano"], "decisoes": r["decisoes"]})
    with open(REFERENCIAS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(REFERENCIAS)


if __name__ == "__main__":
    if sys.argv[1:] == ["referencias"]:
        referencias()
    else:
        raise SystemExit(__doc__)
