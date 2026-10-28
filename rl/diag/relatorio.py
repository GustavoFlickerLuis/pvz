"""Tabelas e figura do rl/diag/RELATORIO.md a partir dos logs.

    python -m rl.diag.relatorio      # imprime as tabelas em Markdown e grava figuras/vies_ini.png
"""
import csv
import json
from pathlib import Path

import numpy as np

from pvz.ambiente import ESPERAR

from . import prova

RAIZ = Path(__file__).parent
LOGS = RAIZ / "logs"
FIG = RAIZ / "figuras"
G1 = ("td2000-cnn1-s0", "td500-cnn1-s0", "mc-cnn1-s0")
G2 = ("td2000-cnn1d-s0", "td500-cnn1d-s0")


def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [{k: float(v) for k, v in l.items()} for l in csv.DictReader(f)]


def tipo_da_acao(c):
    return np.where(c == ESPERAR, 5, np.where(c >= 0, c // 45, -1))


def nota_por_tipo():
    """Referência sem estado: Q = média de Q̂ por tipo de ação (esperar e as 5 plantas),
    ajustada na prova de treino e aplicada na de avaliação."""
    pt, pa = prova.carregar("treino"), prova.carregar("avaliacao")
    tt, qt = tipo_da_acao(pt["cand"]), pt["q"][:, :, 0]
    media = np.array([np.nanmean(qt[tt == k]) for k in range(6)])
    ta = tipo_da_acao(pa["cand"])
    q = np.where(ta >= 0, media[np.maximum(ta, 0)], np.nan)
    return prova.nota(q, pa), media


def propagou(td, mc, tol=0.05):
    """1ª atualização com |viés ini| < 0,05 e Huber ini ≤ 1,1 × a do mc na mesma atualização."""
    hmc = {l["atualizacao"]: l["huber_ini"] for l in mc}
    for l in td:
        t = l["atualizacao"]
        if t > 0 and abs(l["vies_ini"]) < tol and t in hmc and l["huber_ini"] <= 1.1 * hmc[t]:
            return int(t)
    return None


def main():
    linhas = []
    mc = ler_csv(LOGS / "mc-cnn1-s0" / "metricas.csv")
    linhas += ["## G1: propagação (cnn1, k = 20, 100k atualizações)", "",
               "| execução | viés ini (final) | Huber ini (final) | Huber ini do mc (final) | "
               "viés meio / fim | propagou? |", "|---|---|---|---|---|---|"]
    for id_ in G1 + G2:
        if not (LOGS / id_ / "metricas.csv").exists():
            continue
        m = ler_csv(LOGS / id_ / "metricas.csv")
        f = m[-1]
        prop = "—" if id_.startswith("mc") else (
            "sim, em " + f"{propagou(m, mc):,}".replace(",", ".") if propagou(m, mc)
            else "não aconteceu em 100k")
        linhas.append(f"| {id_} | {f['vies_ini']:+.3f} | {f['huber_ini']:.4f} | {mc[-1]['huber_ini']:.4f} | "
                      f"{f['vies_meio']:+.3f} / {f['vies_fim']:+.3f} | {prop} |")

    nt_tipo, media = nota_por_tipo()
    linhas += ["", "## Notas na prova de avaliação (k = 20; 600 estados)", "",
               "| rede | nota (pares claros) | grupo i (com Econ5) | grupo ii (só alternativas) | "
               "arrependimento | nota na atualização 0 |", "|---|---|---|---|---|---|"]
    linhas.append(f"| referência: Q constante | 50,0% | 50,0% | 50,0% | — | — |")
    linhas.append(f"| referência: Q por tipo (sem estado) | {nt_tipo['nota']:.1f}% ({nt_tipo['pares']}) | "
                  f"{nt_tipo['nota_i']:.1f}% ({nt_tipo['pares_i']}) | {nt_tipo['nota_ii']:.1f}% "
                  f"({nt_tipo['pares_ii']}) | {nt_tipo['arrependimento']:.4f} | — |")
    linhas.append(f"| referência: escolha do Econ5 | — | — | — | {nt_tipo['arrependimento_econ5']:.4f} | — |")
    for id_ in G1 + G2:
        if not (LOGS / id_ / "notas.csv").exists():
            continue
        n = ler_csv(LOGS / id_ / "notas.csv")
        f = n[-1]
        curva = ", ".join(f"{int(l['atualizacao']) // 1000}k: {l['nota']:.1f}" for l in n)
        linhas.append(f"| {id_} | {f['nota']:.1f}% ({int(f['pares'])}) | {f['nota_i']:.1f}% | "
                      f"{f['nota_ii']:.1f}% | {f['arrependimento']:.4f} | {n[0]['nota']:.1f}% |")
        linhas.append(f"|   (curva: {curva}) | | | | | |")
    for arq in ("cnn1d", "cnn1"):
        r = LOGS / f"teto-{arq}-s0" / "resultado.json"
        if r.exists():
            j = json.loads(r.read_text(encoding="utf-8"))
            f = j["nota"]
            linhas.append(f"| teto-{arq} (supervisionado em Q̂; parada em {j['parada']}) | "
                          f"{f['nota']:.1f}% ({f['pares']}) | {f['nota_i']:.1f}% | {f['nota_ii']:.1f}% | "
                          f"{f['arrependimento']:.4f} | — |")
    linhas += ["", "Q̂ médio por tipo na prova de treino (esperar, girassol, ervilha, repetidora, "
               "batata, mina): " + ", ".join(f"{v:.3f}" for v in [media[5], *media[:5]])]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        for id_ in G1 + G2:
            if (LOGS / id_ / "metricas.csv").exists():
                m = ler_csv(LOGS / id_ / "metricas.csv")
                ax.plot([l["atualizacao"] for l in m], [l["vies_ini"] for l in m], label=id_)
        ax.axhspan(-0.05, 0.05, color="0.9")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("atualizações")
        ax.set_ylabel("viés médio Q(s, a_β) − G, primeiras 30 decisões")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "vies_ini.png", dpi=120)
    except Exception as e:
        print(f"(sem figura: {e})")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
