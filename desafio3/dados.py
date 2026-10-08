"""Dataset do proxy supervisionado (Desafio 3, etapa 1).

Política de coleta: Econ5 com ε = 15% de ações aleatórias (uniforme entre as válidas,
"esperar" incluída), rng semeado por fase + semente. Para cada decisão grava a
observação, a ação tomada e o retorno de Monte Carlo G_t = r_t + γ·G_{t+1}.

    python -m desafio3.dados [--saida desafio3/dados]

Conjuntos:
- treino: FASES_TREINO × sementes 0–59
- val:    FASES_TREINO × sementes 100–109
- teste:  FASES_TESTE × sementes 10000–10019. Gerado junto, mas `carregar()` se
  recusa a abri-lo: só é lido uma vez, no fim do Desafio 3.

Este código só usa reset, step, observar e mascara do Ambiente. O Econ5 lê o estado
interno, mas ele é a política de coleta (uma regra escrita à mão), não o aprendiz.
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pvz.agentes import AGENTES
from pvz.ambiente import Ambiente
from pvz.divisao import FASES_TESTE, FASES_TREINO
from pvz.hordas import todas_as_fases

PASTA = Path(__file__).parent / "dados"
CAMPOS = ("obs", "acao", "G", "fase", "semente", "passo", "venceu")
FASES = todas_as_fases()   # o campo `fase` guarda o índice nesta lista


@dataclass(frozen=True)
class ConfigDados:
    agente: str = "econ5"
    epsilon: float = 0.15
    gamma: float = 0.997
    sementes_treino: tuple = (0, 60)       # intervalos [início, fim)
    sementes_val: tuple = (100, 110)
    sementes_teste: tuple = (10_000, 10_020)

    def conjuntos(self):
        return {
            "treino": (FASES_TREINO, range(*self.sementes_treino)),
            "val": (FASES_TREINO, range(*self.sementes_val)),
            "teste": (FASES_TESTE, range(*self.sementes_teste)),
        }


def commit_atual():
    """(hash do HEAD, se há alterações não commitadas)."""
    raiz = Path(__file__).resolve().parent.parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True,
                          text=True, check=True).stdout.strip()
    sujo = subprocess.run(["git", "status", "--porcelain"], cwd=raiz, capture_output=True,
                          text=True, check=True).stdout.strip() != ""
    return head, sujo


def retornos(recompensas, gamma):
    g = np.zeros(len(recompensas), dtype=np.float64)
    acum = 0.0
    for t in range(len(recompensas) - 1, -1, -1):
        acum = recompensas[t] + gamma * acum
        g[t] = acum
    return g.astype(np.float32)


def partida(cfg, fase, semente):
    """Joga uma partida com a política de coleta e devolve os arrays dela."""
    amb = Ambiente()
    agente = AGENTES[cfg.agente]()
    rng = random.Random(f"{','.join(fase)}|{semente}")
    obs = amb.reset(fase, semente)
    lista_obs, acoes, recompensas = [], [], []
    fim = False
    while not fim:
        m = amb.mascara()
        if rng.random() < cfg.epsilon:
            a = int(rng.choice(np.flatnonzero(m)))
        else:
            a = int(agente.agir(amb))
        lista_obs.append(obs)
        acoes.append(a)
        obs, r, fim = amb.step(a)
        recompensas.append(r)
    venceu = recompensas[-1] > 0   # a última recompensa leva o ±1 (o dano nunca passa de 1)
    n = len(acoes)
    return {
        "obs": np.stack(lista_obs).astype(np.float16),
        "acao": np.array(acoes, dtype=np.int16),
        "G": retornos(recompensas, cfg.gamma),
        "fase": np.full(n, FASES.index(fase), dtype=np.int16),
        "semente": np.full(n, semente, dtype=np.int32),
        "passo": np.arange(n, dtype=np.int16),
        "venceu": np.full(n, venceu, dtype=bool),
    }


def gerar_conjunto(cfg, fases, sementes):
    partes = [partida(cfg, f, s) for f in fases for s in sementes]
    return {c: np.concatenate([p[c] for p in partes]) for c in CAMPOS}


def hash_arrays(arrays):
    h = hashlib.sha256()
    for c in CAMPOS:
        a = np.ascontiguousarray(arrays[c])
        h.update(c.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def pares(arrays):
    return set(zip(arrays["fase"].tolist(), arrays["semente"].tolist()))


def resumo(arrays):
    g = arrays["G"]
    inicio = arrays["passo"] == 0
    return {
        "amostras": int(len(g)),
        "partidas": int(inicio.sum()),
        "vitorias": float(arrays["venceu"][inicio].mean()),
        "G": {"media": float(g.mean()), "desvio": float(g.std()),
              "min": float(g.min()), "max": float(g.max())},
    }


def gerar(cfg, pasta=PASTA):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    head, sujo = commit_atual()
    meta = {"config": asdict(cfg), "commit": head, "commit_sujo": sujo,
            "fases": [",".join(f) for f in FASES], "conjuntos": {}}
    dados = {}
    for nome, (fases, sementes) in cfg.conjuntos().items():
        t0 = time.time()
        dados[nome] = gerar_conjunto(cfg, fases, sementes)
        np.savez(pasta / f"{nome}.npz", **dados[nome])
        meta["conjuntos"][nome] = {**resumo(dados[nome]), "hash": hash_arrays(dados[nome]),
                                   "tempo_s": round(time.time() - t0, 1)}
        print(nome, meta["conjuntos"][nome])

    # nenhuma partida do teste pode estar no treino ou na validação
    assert not pares(dados["teste"]) & (pares(dados["treino"]) | pares(dados["val"]))
    assert not pares(dados["treino"]) & pares(dados["val"])
    # limites teóricos do retorno (seção 2.4): dano soma no máximo 1, mais ±1 no fim
    for nome, d in dados.items():
        assert -1.0 - 1e-4 <= d["G"].min() and d["G"].max() <= 2.0 + 1e-4, nome

    with open(pasta / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def carregar(conjunto, pasta=PASTA, permitir_teste=False):
    """Lê um conjunto e confere o hash contra o meta.json. O teste exige permitir_teste=True."""
    if conjunto == "teste" and not permitir_teste:
        raise PermissionError("o conjunto de teste só é lido uma vez, no fim do Desafio 3")
    pasta = Path(pasta)
    with np.load(pasta / f"{conjunto}.npz") as z:
        arrays = {c: z[c] for c in CAMPOS}
    with open(pasta / "meta.json", encoding="utf-8") as f:
        esperado = json.load(f)["conjuntos"][conjunto]["hash"]
    if hash_arrays(arrays) != esperado:
        raise ValueError(f"{conjunto}.npz não bate com o hash do meta.json; gere os dados de novo")
    return arrays


def hash_dataset(pasta=PASTA):
    """Hash de treino + validação (o que o treino usa), para registrar nos logs."""
    with open(Path(pasta) / "meta.json", encoding="utf-8") as f:
        c = json.load(f)["conjuntos"]
    return hashlib.sha256((c["treino"]["hash"] + c["val"]["hash"]).encode()).hexdigest()[:16]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--saida", default=str(PASTA))
    args = p.parse_args()
    t0 = time.time()
    gerar(ConfigDados(), args.saida)
    print(f"pronto em {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
