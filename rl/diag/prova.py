"""C4: provas (estados com o valor de cada candidata estimado por rollouts) e C5: nota().

    python -m rl.diag.prova [--processos 16]     # gera as duas provas de k = 20

Sementes (nenhuma usada no rl/: treino 1000–9999, avaliação 300–319, currículo 400–420,
sanidade 300–499; desafio3 e dados_td: 0–59 e 100–109):
- prova de AVALIAÇÃO: FASES_TREINO × 600–629, 600 estados, 8 rollouts por candidata;
- prova de TREINO (do teto): FASES_TREINO × 700–799, 2.000 estados, 4 rollouts.

Estados: partidas com β; só decisões com ≥ 2 tipos de planta disponíveis; estratificados
pela posição da decisão (< 30, 30–99, ≥ 100), um terço em cada faixa. Candidatas: esperar,
a ação do Econ5 puro, uma ação por tipo disponível (célula válida sorteada) e 2 plantios
extras sorteados (sem repetir). Para cada candidata: amb.clonar(), executa a candidata e
continua com β até o fim; Q̂ = média do retorno descontado (γ = 0,994) e o erro-padrão.
O rollout r usa o mesmo rng de β em todas as candidatas do estado (números aleatórios
comuns: a diferença entre candidatas fica menos ruidosa); muda de um rollout para outro.
"""
import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from pvz.ambiente import ESPERAR, Ambiente
from pvz.divisao import FASES_TREINO

from .comum import ECON5, GAMMA, K, acao_beta, entrada, tipos_disponiveis

PASTA = Path(__file__).parent / "dados"
PROVAS = {"avaliacao": dict(sementes=range(600, 630), estados=600, rollouts=8),
          "treino": dict(sementes=range(700, 800), estados=2000, rollouts=4)}
FAIXAS_POS = ((0, 30), (30, 100), (100, 10 ** 6))
MAX_CAND = 9
CAMPOS = ("obs", "sol", "mascara", "cand", "q", "ep", "n_cand", "econ5", "pos", "fase", "semente")


def _rng_beta(fase, semente):
    return random.Random(f"prova|{','.join(fase)}|{semente}")


def _faixa(pos):
    return next(i for i, (a, b) in enumerate(FAIXAS_POS) if a <= pos < b)


def elegiveis(args):
    """Passada 1: as posições elegíveis (≥ 2 tipos disponíveis) de uma partida."""
    fase, semente = args
    amb, rng = Ambiente(k=K), _rng_beta(fase, semente)
    amb.reset(fase, semente)
    pos, fim, lista = 0, False, []
    while not fim:
        m = amb.mascara()
        if len(tipos_disponiveis(m)) >= 2:
            lista.append(pos)
        _, _, fim = amb.step(acao_beta(amb, rng, m))
        pos += 1
    return lista


def _rollout(amb, acao, seed):
    amb = amb.clonar()
    rng = random.Random(seed)
    _, r, fim = amb.step(acao)
    g, desc = r, GAMMA
    while not fim:
        _, r, fim = amb.step(acao_beta(amb, rng))
        g += desc * r
        desc *= GAMMA
    return g


def avaliar_partida(args):
    """Passada 2: rejoga a partida e, nas posições escolhidas, mede as candidatas."""
    fase, semente, posicoes, rollouts = args
    amb, rng = Ambiente(k=K), _rng_beta(fase, semente)
    obs = amb.reset(fase, semente)
    escolha = random.Random(f"cand|{','.join(fase)}|{semente}")
    saida, pos, fim = [], 0, False
    alvo = set(posicoes)
    while not fim and pos <= max(posicoes):
        m = amb.mascara()
        if pos in alvo:
            econ = int(ECON5.agir(amb))
            cand = [ESPERAR, econ]
            for t in tipos_disponiveis(m):
                validas = np.flatnonzero(m[t * 45:(t + 1) * 45]) + t * 45
                cand.append(int(escolha.choice(list(validas))))
            plantios = [a for a in np.flatnonzero(m[:ESPERAR]) if a not in cand]
            cand += [int(a) for a in escolha.sample(plantios, min(2, len(plantios)))]
            cand = list(dict.fromkeys(cand))
            G = np.array([[_rollout(amb, a, f"rollout|{fase}|{semente}|{pos}|{r}")
                           for r in range(rollouts)] for a in cand])
            q = np.full((MAX_CAND, 2), np.nan, np.float32)
            q[:len(cand), 0] = G.mean(1)
            q[:len(cand), 1] = G.std(1, ddof=1) / np.sqrt(rollouts)
            c = np.full(MAX_CAND, -1, np.int16)
            c[:len(cand)] = cand
            saida.append({"obs": obs.astype(np.float16), "sol": round(float(obs[11, 0, 0]) * 1000),
                          "mascara": m, "cand": c, "q": q, "n_cand": len(cand),
                          "econ5": cand.index(econ), "pos": pos, "semente": semente,
                          "fase": FASES_TREINO.index(fase)})
        obs, _, fim = amb.step(acao_beta(amb, rng, m))
        pos += 1
    return saida


def gerar(nome, processos):
    cfg = PROVAS[nome]
    partidas = [(f, s) for f in FASES_TREINO for s in cfg["sementes"]]
    with Pool(processos) as p:
        eleg = p.map(elegiveis, partidas)
    rng = np.random.default_rng(12345 if nome == "avaliacao" else 54321)
    por_faixa = [[(i, pos) for i, l in enumerate(eleg) for pos in l if _faixa(pos) == f]
                 for f in range(3)]
    n = cfg["estados"] // 3
    escolhidos = {}
    for lista in por_faixa:
        for j in rng.choice(len(lista), size=min(n, len(lista)), replace=False):
            i, pos = lista[j]
            escolhidos.setdefault(i, []).append(pos)
    tarefas = [(*partidas[i], sorted(ps), cfg["rollouts"]) for i, ps in escolhidos.items()]
    with Pool(processos) as p:
        res = [x for parte in p.map(avaliar_partida, tarefas, chunksize=1) for x in parte]
    d = {c: np.array([x[c] for x in res]) for c in CAMPOS if c != "ep"}
    d["ep"] = np.array([0] * len(res))
    return d, [len(l) for l in por_faixa]


def carregar(nome):
    with np.load(PASTA / f"prova_{nome}_k{K}.npz") as z:
        return {c: z[c] for c in z.files}


# ---------------------------------------------------------------- C5: nota
@torch.no_grad()
def q_candidatas(rede, prova, formato, dev="cpu", bloco=1024):
    """Q da rede nas candidatas de cada estado: (estados, MAX_CAND), nan fora."""
    rede.eval()
    out = []
    for i in range(0, len(prova["obs"]), bloco):
        sl = slice(i, i + bloco)
        x = entrada(formato, torch.from_numpy(prova["obs"][sl]), torch.from_numpy(prova["sol"][sl]),
                    torch.from_numpy(prova["mascara"][sl])).to(dev)
        q = rede(x).cpu().numpy()
        c = prova["cand"][sl]
        out.append(np.where(c >= 0, np.take_along_axis(q, np.maximum(c, 0), 1), np.nan))
    rede.train()
    return np.concatenate(out)


def nota(q_rede, prova, limiar=0.02):
    """% de pares claros na ordem certa (empate = meio), por grupo, e arrependimento.

    Par claro: candidatas do mesmo estado com |Q̂a − Q̂b| > limiar e > 2·EP combinado.
    Grupo i: pares que envolvem a ação do Econ5; grupo ii: só entre as outras.
    Arrependimento: Q̂ da melhor candidata − Q̂ da escolhida pela rede (empates na rede:
    média das empatadas)."""
    acertos = {"i": [], "ii": []}
    arrep, arrep_econ = [], []
    for s in range(len(q_rede)):
        n = int(prova["n_cand"][s])
        qh, ep, qr = prova["q"][s, :n, 0], prova["q"][s, :n, 1], q_rede[s, :n]
        e5 = int(prova["econ5"][s])
        for a in range(n):
            for b in range(a + 1, n):
                d = qh[a] - qh[b]
                if abs(d) > limiar and abs(d) > 2 * np.sqrt(ep[a] ** 2 + ep[b] ** 2):
                    dr = qr[a] - qr[b]
                    ok = 0.5 if dr == 0 else float(np.sign(dr) == np.sign(d))
                    acertos["i" if e5 in (a, b) else "ii"].append(ok)
        melhores = np.flatnonzero(qr == qr.max())
        arrep.append(qh.max() - qh[melhores].mean())
        arrep_econ.append(qh.max() - qh[e5])
    todos = acertos["i"] + acertos["ii"]
    pct = lambda l: 100 * float(np.mean(l)) if l else float("nan")
    return {"nota": pct(todos), "pares": len(todos), "nota_i": pct(acertos["i"]),
            "pares_i": len(acertos["i"]), "nota_ii": pct(acertos["ii"]),
            "pares_ii": len(acertos["ii"]), "arrependimento": float(np.mean(arrep)),
            "arrependimento_econ5": float(np.mean(arrep_econ))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processos", type=int, default=16)
    p.add_argument("--so", choices=list(PROVAS))
    a = p.parse_args()
    PASTA.mkdir(exist_ok=True)
    for nome in ([a.so] if a.so else PROVAS):
        t0 = time.time()
        d, faixas = gerar(nome, a.processos)
        np.savez(PASTA / f"prova_{nome}_k{K}.npz", **d)
        r = nota(d["q"][:, :, 0], d)   # a própria resposta: 100%
        meta = {"k": K, "gamma": GAMMA, **{k: (list(v) if isinstance(v, range) else v)
                                           for k, v in PROVAS[nome].items()},
                "estados": len(d["pos"]), "elegiveis_por_faixa": faixas,
                "estados_por_faixa": [int(np.sum((d["pos"] >= x) & (d["pos"] < y)))
                                      for x, y in FAIXAS_POS],
                "candidatas_media": float(d["n_cand"].mean()), "pares_claros": r["pares"],
                "pares_i": r["pares_i"], "pares_ii": r["pares_ii"],
                "arrependimento_econ5": r["arrependimento_econ5"],
                "ep_mediano": float(np.nanmedian(d["q"][:, :, 1])),
                "segundos": round(time.time() - t0, 1)}
        (PASTA / f"prova_{nome}_k{K}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(nome, meta, flush=True)


if __name__ == "__main__":
    main()
