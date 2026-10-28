"""Calibração dos baselines: todos os agentes x 27 fases x N sementes, em paralelo.

Metas (27 fases x 5 sementes):
  - aleatório ~ 0% de vitórias;
  - guloso entre 20% e 45%;
  - melhor Econ(N) entre 60% e 85%, com a curva em N com pico no MEIO
    (N pequeno e N grande piores: é o dilema economia x defesa);
  - nenhum Econ(N) acima de 90% (se uma regra fixa vence tudo, não sobra o que aprender);
  - sol médio sobrando no fim < 200.
Ajuste só INTERVALO_GIRASSOL (config.py) e PESO_SEGMENTO (hordas.py), um de cada vez,
e anote cada tentativa.

Como rodar (da raiz do repo):
    python tools/calibrar.py --girassol 120 --pesos 1 2 2.25 --saida resultado.json
    python tools/analisar.py resultado.json

Os valores entram por argumento, sem editar os arquivos. Cada processo do pool recebe
INTERVALO_GIRASSOL uma vez, no initializer (atribuição absoluta, nada se acumula entre
jobs); os pesos vão como argumento de cada job, sem tocar em estado global.

Resultado de referência (120; 1, 2, 2.25): aleatório 0%, guloso 37,8%,
econ3/4/5/6/8 = 25/52/67/54/36%.
"""
import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvz import config as C  # noqa: E402
from pvz.agentes import AGENTES  # noqa: E402
from pvz.ambiente import Ambiente  # noqa: E402
from pvz.hordas import montar_fase, todas_as_fases  # noqa: E402

NOMES = ["aleatorio", "guloso", "econ3", "econ4", "econ5", "econ6", "econ8"]


def _init(girassol):
    C.INTERVALO_GIRASSOL = girassol


def _job(args):
    nome, fase, semente, pesos = args
    amb = Ambiente()
    amb.reset(montar_fase(fase, semente, 1.0, pesos))
    agente = AGENTES[nome]()
    segmento_de, guardados = {}, []   # id(zumbi) -> segmento de onde ele veio
    fim = False
    while not fim:
        p0 = amb.prox_spawn
        _, _, fim = amb.step(agente.agir(amb))
        novos = list(amb.fase.spawns[p0:amb.prox_spawn])
        for faixa, lista in enumerate(amb.zumbis):
            segs = [s.segmento for s in novos if s.faixa == faixa]
            for z in lista:
                if id(z) not in segmento_de and segs:
                    segmento_de[id(z)] = segs.pop(0)
                    guardados.append(z)   # mantém vivo: id() não pode ser reaproveitado
    culpado = None
    if not amb.venceu:
        cruzou = [z for l in amb.zumbis for z in l if z.x < 0]
        culpado = segmento_de.get(id(cruzou[0])) if cruzou else None   # None = estourou MAX_TICKS
    return dict(agente=nome, fase=list(fase), semente=semente, venceu=bool(amb.venceu),
                sol=amb.sol, dano=amb.info()["dano_frac"], tick=amb.tick, culpado=culpado,
                girassol=C.INTERVALO_GIRASSOL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girassol", type=int, required=True)
    ap.add_argument("--pesos", type=float, nargs=3, required=True)
    ap.add_argument("--sementes", type=int, default=5)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    jobs = [(n, f, s, tuple(a.pesos)) for n in NOMES for f in todas_as_fases() for s in range(a.sementes)]
    t0 = time.perf_counter()
    with mp.Pool(initializer=_init, initargs=(a.girassol,)) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    assert all(r["girassol"] == a.girassol for r in res)
    with open(a.saida, "w") as f:
        json.dump(dict(girassol=a.girassol, pesos=a.pesos, res=res), f)
    print(f"girassol={a.girassol} pesos={a.pesos}  ({time.perf_counter() - t0:.0f}s)")
    for n in NOMES:
        rs = [r for r in res if r["agente"] == n]
        print(f"  {n:10s} vitórias {statistics.mean(r['venceu'] for r in rs):6.1%}  "
              f"sol final {statistics.mean(r['sol'] for r in rs):6.0f}  "
              f"dano {statistics.mean(r['dano'] for r in rs):6.1%}")


if __name__ == "__main__":
    main()
