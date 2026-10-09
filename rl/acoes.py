"""Distribuição de ações da política gulosa de cada execução terminada.

    python -m rl.acoes                  # as execuções do lote 1 (status.FILA)
    python -m rl.acoes cnn2-s0 cnn3-cur-s0

Joga FASES_TREINO × sementes 300–302 (63 partidas) com o modelo.pt de cada execução, na CPU,
e conta as ações (esperar e cada planta). Só usa reset/step/observar/mascara.
"""
import json
import math
import sys
from collections import Counter

import torch

from desafio3.modelo import criar, formato
from pvz import config as C
from pvz.ambiente import ESPERAR, Ambiente, decodificar_acao

from .avaliar import partidas
from .status import FILA, PASTA_LOGS

NOMES = ["esperar"] + [p.nome for p in C.PLANTAS]


@torch.no_grad()
def distribuicao(modelo, fmt="v1"):
    cont, vit, dec = Counter(), 0, 0
    for fase, semente in partidas(range(300, 303)):
        amb = Ambiente(formato=fmt)
        obs, fim = amb.reset(fase, semente), False
        while not fim:
            q = modelo(torch.from_numpy(obs)[None])[0]
            a = int(q.masked_fill(~torch.from_numpy(amb.mascara()), -math.inf).argmax())
            cont["esperar" if a == ESPERAR else C.PLANTAS[decodificar_acao(a)[0]].nome] += 1
            obs, r, fim = amb.step(a)
            dec += 1
        vit += r > 0
    n = sum(cont.values())
    return {k: cont[k] / n for k in NOMES}, vit / 63, dec / 63


def main(ids):
    torch.set_num_threads(1)
    print("| execução | vitórias (63) | decisões/partida | " + " | ".join(NOMES) + " |")
    print("|---|---|---|" + "---|" * len(NOMES))
    for id_ in ids:
        pasta = PASTA_LOGS / id_
        if not (pasta / "modelo.pt").exists():
            continue
        cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))["config"]
        arq = cfg.get("arquitetura", "cnn1")   # o lote 1 não gravava a arquitetura
        m = criar(arq)
        m.load_state_dict(torch.load(pasta / "modelo.pt", map_location="cpu"))
        d, v, dec = distribuicao(m.eval(), formato(arq))
        print(f"| {id_} | {100 * v:.1f}% | {dec:.0f} | "
              + " | ".join(f"{100 * d[k]:.1f}%" for k in NOMES) + " |", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or [f"{r}-s{s}" for r, s in FILA])
