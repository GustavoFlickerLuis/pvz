"""Gráficos de uma execução: python -m desafio3.graficos <pasta do log>

Painéis empilhados no mesmo eixo de passos: perda (lote, lote fixo do treino e
validação), ‖g‖ antes do corte, ‖θ‖, η de cada otimizador e ρ por camada.
Grava graficos.png na pasta do log.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .modelo import CAMADAS


def ler(pasta):
    with open(pasta / "registro.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    colunas = {c: np.array([float(l[c]) if l[c] != "" else np.nan for l in linhas])
               for c in linhas[0]}
    with open(pasta / "config.json", encoding="utf-8") as f:
        info = json.load(f)
    return colunas, info


def serie(r, coluna):
    ok = ~np.isnan(r[coluna])
    return r["passo"][ok], r[coluna][ok]


def desenhar(pasta):
    pasta = Path(pasta)
    r, info = ler(pasta)
    cfg, res = info["config"], info["resultado"]
    fig, eixos = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    titulo = (f"{cfg['otimizador']}  η={cfg['lr']:g}  {cfg['agenda']}  semente {cfg['semente']}  "
              f"— val final {res['val_final']:.4f}, mín {res['val_min']:.4f} "
              f"(passo {res['passo_val_min']}), {res['status']}")
    fig.suptitle(titulo, fontsize=10)

    ax = eixos[0]
    ax.plot(*serie(r, "perda_lote"), lw=0.6, alpha=0.5, label="lote")
    ax.plot(*serie(r, "huber_treino"), marker=".", label="treino (lote fixo)")
    ax.plot(*serie(r, "huber_val"), marker=".", label="validação")
    ax.set_yscale("log")
    ax.set_ylabel("Huber")
    ax.legend()

    ax = eixos[1]
    ax.plot(*serie(r, "norma_g"), lw=0.6)
    ax.axhline(cfg["clip"], color="k", ls="--", lw=0.8, label=f"corte = {cfg['clip']:g}")
    ax.set_yscale("log")
    ax.set_ylabel("‖g‖ (antes do corte)")
    ax.legend()

    ax = eixos[2]
    ax.plot(*serie(r, "norma_theta"))
    ax.set_ylabel("‖θ‖")

    ax = eixos[3]
    for c in sorted(k for k in r if k.startswith("eta_")):
        ax.plot(*serie(r, c), label=c)
    ax.set_ylabel("η")
    ax.legend()

    ax = eixos[4]
    for n in CAMADAS:
        ax.plot(*serie(r, f"rho_{n}"), lw=0.7, label=n)
    ax.set_yscale("log")
    ax.set_ylabel("ρ = ‖Δθ‖ / ‖θ‖")
    ax.set_xlabel("passo")
    ax.legend(fontsize=8, ncol=5)

    for ax in eixos:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    saida = pasta / "graficos.png"
    fig.savefig(saida, dpi=110)
    plt.close(fig)
    return saida


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pasta")
    print(desenhar(p.parse_args().pasta))


if __name__ == "__main__":
    main()
