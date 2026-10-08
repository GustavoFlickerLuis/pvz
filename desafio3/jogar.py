"""Extra exploratório (fora do Desafio 3): a rede do proxy joga?

    python -m desafio3.jogar

RedeGulosa: a cada decisão, argmax_a Q(s, a) entre as ações válidas, usando só
amb.observar() e amb.mascara(). Pesos: a vencedora (sgd 3e-2 cosseno) e a baseline
(adam 1e-4 constante), semente 0, gravados por `treino --salvar-pesos` em desafio3/pesos/.
Joga nas 21 FASES_TREINO com as sementes 200–219 (faixa de treino, fora dos dados), nunca
nas fases de teste, e compara com Econ5 (sem aleatoriedade) e Guloso nas mesmas partidas.
O código da partida só usa reset/step; vitória e dano saem das recompensas.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pvz import config as C
from pvz.agentes import Econ, Guloso
from pvz.ambiente import ESPERAR, Ambiente, decodificar_acao
from pvz.divisao import FASES_TESTE, FASES_TREINO

from .modelo import CNN
from .treino import PASTA_LOGS, PASTA_PESOS

SEMENTES = range(200, 220)
SAIDA = Path(__file__).parent / "rede_joga.md"
REDES = {   # rótulo: (otimizador, lr, agenda)
    "RedeGulosa (vencedora sgd 3e-2 cosseno)": ("sgd", 3e-2, "cosseno"),
    "RedeGulosa (baseline adam 1e-4 constante)": ("adam", 1e-4, "constante"),
}


class RedeGulosa:
    def __init__(self, modelo):
        self.modelo = modelo.eval()

    @torch.no_grad()
    def agir(self, amb):
        q = self.modelo(torch.from_numpy(amb.observar())[None])[0]
        q[~torch.from_numpy(amb.mascara())] = -math.inf
        return int(q.argmax())


def carregar_rede(otimizador, lr, agenda, semente=0):
    for peso in sorted(PASTA_PESOS.glob("*.pt")):
        with open(PASTA_LOGS / "rascunho" / peso.stem / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)["config"]
        if (cfg["otimizador"], cfg["agenda"], cfg["semente"]) == (otimizador, agenda, semente) \
                and math.isclose(cfg["lr"], lr):
            modelo = CNN()
            modelo.load_state_dict(torch.load(peso, map_location="cpu"))
            return modelo
    raise FileNotFoundError(f"sem pesos para {otimizador} {lr} {agenda} s{semente}: "
                            "rode o treino com --salvar-pesos")


def jogar(agente, fase, semente):
    """(venceu, dano, ações)."""
    amb = Ambiente()
    amb.reset(fase, semente)
    acoes, dano, fim = [], 0.0, False
    while not fim:
        a = agente.agir(amb)
        acoes.append(a)
        _, r, fim = amb.step(a)
        dano += r
    venceu = r > 0
    dano -= 1.0 if venceu else -1.0   # tira o ±1 do fim: sobra a fração de dano
    return venceu, dano, acoes


def distribuicao(acoes):
    tipos = Counter("esperar" if a == ESPERAR else C.PLANTAS[decodificar_acao(a)[0]].nome
                    for a in acoes)
    n = len(acoes)
    return {k: tipos.get(k, 0) / n for k in ["esperar"] + [p.nome for p in C.PLANTAS]}


def main():
    assert not set(FASES_TREINO) & set(FASES_TESTE)
    torch.set_num_threads(1)
    agentes = {r: RedeGulosa(carregar_rede(*cfg)) for r, cfg in REDES.items()}
    agentes["Econ5 (sem aleatoriedade)"] = Econ(5)
    agentes["Guloso"] = Guloso()
    out = ["# A rede do proxy joga? (extra exploratório, fora do Desafio 3)", "",
           f"21 FASES_TREINO × sementes {SEMENTES.start}–{SEMENTES.stop - 1} = "
           f"{len(FASES_TREINO) * len(SEMENTES)} partidas por agente (as mesmas para todos).",
           "Dano: fração da vida total da fase causada como dano (média por partida).", "",
           "| agente | vitórias | dano médio | decisões/partida | esperar | girassol | ervilha | "
           "repetidora | batata | mina |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for rotulo, agente in agentes.items():
        res = [jogar(agente, f, s) for f in FASES_TREINO for s in SEMENTES]
        vit = np.mean([v for v, _, _ in res])
        dano = np.mean([d for _, d, _ in res])
        todas = [a for _, _, acoes in res for a in acoes]
        dist = distribuicao(todas)
        linha = (f"| {rotulo} | {100 * vit:.1f}% | {dano:.3f} | {len(todas) / len(res):.0f} | "
                 + " | ".join(f"{100 * v:.1f}%" for v in dist.values()) + " |")
        out.append(linha)
        print(linha, flush=True)
    SAIDA.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(SAIDA)


if __name__ == "__main__":
    main()
