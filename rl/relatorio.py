"""Tabelas e figuras do treino de RL: python -m rl.relatorio

- rl/figuras/curvas_vitoria.png: vitórias nas avaliações periódicas (63 partidas), as 3
  receitas no mesmo eixo (média das sementes em linha grossa, cada semente em linha fina).
- rl/figuras/sinais_<receita>.png: perda TD, ‖g‖, ‖θ‖, ρ por camada, η, Q médio e ε de
  uma execução por receita (a de menor semente com registro).
- rl/tabelas.md: por execução e por receita (média ± desvio), ao lado de Econ5 e Guloso.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desafio3.modelo import CAMADAS

from .avaliar import REFERENCIAS
from .status import FILA, PASTA_LOGS, RECEITAS, ler

RAIZ = Path(__file__).parent
FIGURAS = RAIZ / "figuras"
TABELAS = RAIZ / "tabelas.md"
CORES = {"baseline": "tab:blue", "vencedora": "tab:green", "muon": "tab:red"}
ROTULOS = {"baseline": "baseline: adam 1e-4 constante", "vencedora": "vencedora: sgd 3e-2 cosseno",
           "muon": "muon 1e-3 (AdamW 3e-4) warmup_cosseno"}


def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return {c: np.array([float(l[c]) if l[c] != "" else np.nan for l in linhas])
            for c in (linhas[0] if linhas else [])}


def curvas():
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in RECEITAS:
        series = []
        for s in range(3):
            arq = PASTA_LOGS / f"{r}-s{s}" / "avaliacoes.csv"
            if not arq.exists():
                continue
            a = ler_csv(arq)
            if not a:
                continue
            ax.plot(a["passo"], 100 * a["vitorias"], color=CORES[r], lw=0.8, alpha=0.45)
            series.append((a["passo"], a["vitorias"]))
        if series:
            n = min(len(p) for p, _ in series)
            media = np.mean([v[:n] for _, v in series], axis=0)
            ax.plot(series[0][0][:n], 100 * media, color=CORES[r], lw=2.5,
                    label=f"{ROTULOS[r]} (média de {len(series)} semente(s))")
    if REFERENCIAS.exists():
        with open(REFERENCIAS, encoding="utf-8") as f:
            for l, ls in zip(csv.DictReader(f), ("--", ":")):
                ax.axhline(100 * float(l["vitorias"]), color="k", ls=ls, lw=1,
                           label=f"{l['agente']} (420 partidas)")
    ax.set_xlabel("passos de jogo")
    ax.set_ylabel("vitórias (%), política gulosa, FASES_TREINO × sementes 300–302")
    ax.set_title("Double DQN: vitórias nas avaliações periódicas (linhas finas: cada semente)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "curvas_vitoria.png", dpi=120)
    plt.close(fig)


def sinais(r):
    for s in range(3):
        pasta = PASTA_LOGS / f"{r}-s{s}"
        if (pasta / "registro.csv").exists() and ler_csv(pasta / "registro.csv"):
            break
    else:
        return None
    d = ler_csv(pasta / "registro.csv")
    x = d["passo_jogo"]
    fig, eixos = plt.subplots(7, 1, figsize=(10, 17), sharex=True)
    eixos[0].plot(x, d["perda_lote"], lw=0.4)
    eixos[0].set_yscale("log")
    eixos[0].set_ylabel("perda TD (Huber, lote)")
    eixos[1].plot(x, d["norma_g"], lw=0.4)
    eixos[1].axhline(10, color="k", ls="--", lw=0.8)
    eixos[1].set_yscale("log")
    eixos[1].set_ylabel("‖g‖ antes do corte\n(-- corte = 10)")
    eixos[2].plot(x, d["norma_theta"])
    eixos[2].set_ylabel("‖θ‖")
    for n in CAMADAS:
        eixos[3].plot(x, d[f"rho_{n}"], lw=0.4, label=n)
    eixos[3].set_yscale("log")
    eixos[3].set_ylabel("ρ por camada")
    eixos[3].legend(fontsize=7, ncol=5)
    for c in sorted(k for k in d if k.startswith("eta_")):
        nome = "η" if "eta_1" not in d else {"eta_0": "η Muon", "eta_1": "η AdamW"}[c]
        eixos[4].plot(x, d[c], label=nome)
    eixos[4].set_yscale("log")
    eixos[4].set_ylabel("η")
    eixos[4].legend(fontsize=7)
    eixos[5].plot(x, d["q_medio"], lw=0.5)
    eixos[5].set_ylabel("Q médio previsto\n(ação tomada)")
    eixos[6].plot(x, d["epsilon"])
    eixos[6].set_ylabel("ε")
    eixos[6].set_xlabel("passos de jogo")
    for ax in eixos:
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Sinais do treino: {ROTULOS[r]}, {pasta.name} (a cada 50 atualizações)")
    fig.tight_layout()
    caminho = FIGURAS / f"sinais_{r}.png"
    fig.savefig(caminho, dpi=110)
    plt.close(fig)
    return caminho


def tabelas():
    out = ["# Double DQN × receitas do proxy: tabelas", "",
           "Vitórias da política gulosa (ε = 0, com máscara). Final: FASES_TREINO × sementes "
           "300–319 = 420 partidas. Periódicas: × 300–302 = 63 partidas, a cada 50.000 passos.", "",
           "## Por execução", "",
           "| execução | status | passos | episódios | vitórias final | dano final | curva periódica |",
           "|---|---|---|---|---|---|---|"]
    finais = {r: [] for r in RECEITAS}
    for r, s in FILA:
        id_ = f"{r}-s{s}"
        prog, res = ler(id_)
        if prog is None:
            out.append(f"| {id_} | não rodou | — | — | — | — | — |")
            continue
        curva = ", ".join(f"{p // 1000}k {100 * v:.0f}%" for p, v in prog["avaliacoes"])
        if res:
            finais[r].append(res["vitorias_final"])
            out.append(f"| {id_} | {res['status']} | {res['passos']} | {res['episodios']} | "
                       f"{100 * res['vitorias_final']:.1f}% | {res['dano_final']:.3f} | {curva} |")
        else:
            out.append(f"| {id_} | {prog['status']} | {prog['passo']} | — | — | — | {curva} |")
    out += ["", "## Por receita (média ± desvio entre sementes)", "",
            "| agente | n | vitórias final (420 partidas cada) |", "|---|---|---|"]
    for r in RECEITAS:
        v = 100 * np.array(finais[r])
        if len(v) == 0:
            cel = "—"
        elif len(v) == 1:
            cel = f"{v[0]:.1f}%"
        else:
            cel = f"{v.mean():.1f}% ± {v.std(ddof=1):.1f}  ({', '.join(f'{x:.1f}' for x in v)})"
        out.append(f"| {ROTULOS[r]} | {len(v)} | {cel} |")
    if REFERENCIAS.exists():
        with open(REFERENCIAS, encoding="utf-8") as f:
            for l in csv.DictReader(f):
                out.append(f"| {l['agente']} (referência) | — | {100 * float(l['vitorias']):.1f}% |")
    TABELAS.write_text("\n".join(out) + "\n", encoding="utf-8")
    return "\n".join(out)


def main():
    FIGURAS.mkdir(exist_ok=True)
    curvas()
    for r in RECEITAS:
        print(sinais(r))
    print(tabelas())
    print(FIGURAS / "curvas_vitoria.png")


if __name__ == "__main__":
    main()
