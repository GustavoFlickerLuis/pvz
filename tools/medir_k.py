"""Baselines por k (ticks entre decisões): o jogo continua com o mesmo dilema com k maior?

    python tools/medir_k.py [--processos 8]     # saída: tools/medir_k.csv e tools/medir_k.md

Agentes aleatorio, guloso, econ3, econ4, econ5, econ6, econ8; k ∈ {10, 20, 30};
FASES_TREINO × sementes 300–319 (420 partidas). Por (agente, k): vitórias com IC 95%
(Wilson), decisões por partida, decisões não forçadas (máscara com algum plantio válido),
sol final médio e o segmento (pelo tick) em que acontecem as derrotas.

Escolha do K (regra combinada antes de medir): K = 20 com γ = 0,994, a não ser que com
k = 20 a curva Econ(N) perca o pico no meio (o máximo tem de estar em N = 4, 5 ou 6 e
acima das duas pontas, N = 3 e 8) ou o Econ5 caia mais de 6 pontos em relação a k = 10.
Os baselines leem o estado interno (são regras escritas à mão), como em tools/calibrar.py.
"""
import argparse
import csv
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvz.agentes import AGENTES  # noqa: E402
from pvz.ambiente import ESPERAR, Ambiente  # noqa: E402
from pvz.divisao import FASES_TREINO  # noqa: E402
from pvz.hordas import SEGMENTOS  # noqa: E402

AGENTES_K = ("aleatorio", "guloso", "econ3", "econ4", "econ5", "econ6", "econ8")
KS = (10, 20, 30)
SEMENTES = range(300, 320)
SAIDA = Path(__file__).parent


def wilson(v, n, z=1.96):
    p = v / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - m, c + m


def segmento_do_tick(fase, tick):
    fim = 0
    for i, nome in enumerate(fase):
        fim += SEGMENTOS[i][nome].duracao
        if tick < fim:
            return i
    return 3   # depois do último segmento (zumbis que sobraram)


def partida(args):
    agente, k, fase, semente = args
    amb = Ambiente(k=k)
    amb.reset(fase, semente)
    ag = AGENTES[agente]()
    if agente == "aleatorio":
        ag.rng.seed(semente)
    decisoes = nao_forcadas = 0
    fim = False
    while not fim:
        m = amb.mascara()
        nao_forcadas += bool(m[:ESPERAR].any())
        _, r, fim = amb.step(ag.agir(amb))
        decisoes += 1
    venceu = r > 0
    return {"agente": agente, "k": k, "fase": "+".join(fase), "semente": semente,
            "venceu": int(venceu), "decisoes": decisoes, "nao_forcadas": nao_forcadas,
            "sol_final": amb.sol,
            "segmento_derrota": "" if venceu else segmento_do_tick(fase, amb.tick)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processos", type=int, default=8)
    a = p.parse_args()
    tarefas = [(ag, k, f, s) for ag in AGENTES_K for k in KS for f in FASES_TREINO for s in SEMENTES]
    with Pool(a.processos) as pool:
        res = pool.map(partida, tarefas, chunksize=8)
    linhas = []
    for ag in AGENTES_K:
        for k in KS:
            r = [x for x in res if x["agente"] == ag and x["k"] == k]
            n, v = len(r), sum(x["venceu"] for x in r)
            lo, hi = wilson(v, n)
            der = [x["segmento_derrota"] for x in r if not x["venceu"]]
            linhas.append({
                "agente": ag, "k": k, "partidas": n, "vitorias": v / n, "ic_lo": lo, "ic_hi": hi,
                "decisoes": np.mean([x["decisoes"] for x in r]),
                "nao_forcadas": np.mean([x["nao_forcadas"] for x in r]),
                "sol_final": np.mean([x["sol_final"] for x in r]),
                **{f"derrotas_seg{i}": der.count(i) for i in range(4)}})
    with open(SAIDA / "medir_k.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    vit = {(l["agente"], l["k"]): l["vitorias"] for l in linhas}
    md = ["# Baselines por k (FASES_TREINO × sementes 300–319, 420 partidas)", "",
          "| agente | k | vitórias (IC 95%) | decisões/partida | não forçadas | sol final | "
          "derrotas por segmento (início/meio/fim/depois) |", "|---|---|---|---|---|---|---|"]
    for l in linhas:
        md.append(f"| {l['agente']} | {l['k']} | {100 * l['vitorias']:.1f}% "
                  f"({100 * l['ic_lo']:.1f}–{100 * l['ic_hi']:.1f}) | {l['decisoes']:.0f} | "
                  f"{l['nao_forcadas']:.0f} | {l['sol_final']:.0f} | "
                  + "/".join(str(l[f"derrotas_seg{i}"]) for i in range(4)) + " |")
    md += ["", "## Curva Econ(N)", "", "| k | " + " | ".join(f"N = {n}" for n in (3, 4, 5, 6, 8)) + " |",
           "|---|---|---|---|---|---|"]
    picos = {}
    for k in KS:
        curva = [vit[(f"econ{n}", k)] for n in (3, 4, 5, 6, 8)]
        md.append(f"| {k} | " + " | ".join(f"{100 * c:.1f}%" for c in curva) + " |")
        i = int(np.argmax(curva))
        picos[k] = i in (1, 2, 3) and curva[i] > curva[0] and curva[i] > curva[4]
    queda = 100 * (vit[("econ5", 10)] - vit[("econ5", 20)])
    escolhe20 = picos[20] and queda <= 6
    md += ["", "## Escolha do K", "",
           f"- Com k = 20, a curva Econ(N) {'mantém' if picos[20] else 'PERDE'} o pico no meio.",
           f"- Econ5: {100 * vit[('econ5', 10)]:.1f}% (k = 10) → {100 * vit[('econ5', 20)]:.1f}% "
           f"(k = 20): queda de {queda:.1f} pontos (limite: 6).",
           f"- **K = {20 if escolhe20 else 10}**"
           + (" com γ = 0,994." if escolhe20 else ": fica só k = 10 (γ = 0,997) daqui em diante.")]
    (SAIDA / "medir_k.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
