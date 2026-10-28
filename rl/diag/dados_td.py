"""C3: dados de TD com a política β (Econ5 + 15% aleatórias), k = 20, γ = 0,994.

    python -m rl.diag.dados_td [--processos 16]

Treino: FASES_TREINO × sementes 0–59; validação: × 100–109 (as do desafio3/dados.py).
Por transição: observação v1 (float16; a v2 se reconstrói com o sol e a máscara), sol,
máscara, ação, recompensa, fim, episódio, posição da decisão e G de Monte Carlo.
Arrays em rl/diag/dados/td_k20_<conjunto>.npz (no .gitignore); meta em td_k20.json.
"""
import argparse
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from desafio3.dados import commit_atual, retornos
from pvz.ambiente import Ambiente
from pvz.divisao import FASES_TREINO

from .comum import GAMMA, K, acao_beta

PASTA = Path(__file__).parent / "dados"
CONJUNTOS = {"treino": range(0, 60), "val": range(100, 110)}
CAMPOS = ("obs", "sol", "mascara", "acao", "r", "fim", "ep", "pos", "G")


def partida(args):
    fase, semente = args
    amb = Ambiente(k=K)
    rng = random.Random(f"{','.join(fase)}|{semente}")   # a semente do desafio3/dados.py
    obs, fim = amb.reset(fase, semente), False
    O, S, M, A, R = [], [], [], [], []
    while not fim:
        m = amb.mascara()
        a = acao_beta(amb, rng, m)
        O.append(obs)
        S.append(round(float(obs[11, 0, 0]) * 1000))   # sol, lido da observação v1
        M.append(m)
        A.append(a)
        obs, r, fim = amb.step(a)
        R.append(r)
    n = len(A)
    return {"obs": np.stack(O).astype(np.float16), "sol": np.array(S, np.int16),
            "mascara": np.stack(M), "acao": np.array(A, np.int16),
            "r": np.array(R, np.float32), "fim": np.arange(n) == n - 1,
            "pos": np.arange(n, dtype=np.int16), "G": retornos(R, GAMMA)}


def gerar(sementes, processos):
    with Pool(processos) as p:
        partes = p.map(partida, [(f, s) for f in FASES_TREINO for s in sementes])
    for i, x in enumerate(partes):
        x["ep"] = np.full(len(x["acao"]), i, np.int32)
    return {c: np.concatenate([x[c] for x in partes]) for c in CAMPOS}


def carregar(conjunto):
    with np.load(PASTA / f"td_k{K}_{conjunto}.npz") as z:
        return {c: z[c] for c in CAMPOS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processos", type=int, default=16)
    a = p.parse_args()
    PASTA.mkdir(exist_ok=True)
    t0 = time.time()
    meta = {"k": K, "gamma": GAMMA, "beta": "econ5 + 15% aleatórias", "commit": commit_atual()}
    for nome, sementes in CONJUNTOS.items():
        d = gerar(sementes, a.processos)
        np.savez(PASTA / f"td_k{K}_{nome}.npz", **d)
        h = hashlib.sha256(b"".join(np.ascontiguousarray(d[c]).tobytes() for c in CAMPOS))
        meta[nome] = {"sementes": [sementes.start, sementes.stop], "transicoes": len(d["acao"]),
                      "episodios": int(d["ep"].max()) + 1,
                      "vitorias": float(np.mean(d["r"][d["fim"]] > 0)),
                      "esperar": float(np.mean(d["acao"] == 225)), "sha256": h.hexdigest()}
        print(nome, meta[nome], flush=True)
    meta["segundos"] = round(time.time() - t0, 1)
    (PASTA / f"td_k{K}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
