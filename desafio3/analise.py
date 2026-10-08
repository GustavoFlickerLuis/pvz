"""Tabelas e gráficos do protocolo, lidos do experimentos.csv e dos logs.

    python -m desafio3.analise varredura      # tabela otimizador × η, melhor η, gráfico val × η
    python -m desafio3.analise agendas        # tabela otimizador × agenda
    python -m desafio3.analise confirmacao    # média ± desvio por combinação e veredito
    python -m desafio3.analise rho <log_muon> <log_adamw>   # ρ por camada lado a lado
    python -m desafio3.analise instavel       # execução com maior pico de ‖g‖ sem divergir

Os gráficos vão para desafio3/figuras/.
"""
import argparse
import csv
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .modelo import CAMADAS
from .treino import EXPERIMENTOS, PASTA_LOGS

FIGURAS = Path(__file__).parent / "figuras"
METRICAS = ("val_final", "val_esperar", "val_plantio")


def ler():
    with open(EXPERIMENTOS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chave(e):
    """(otimizador, η, η AdamW, agenda): identifica uma combinação."""
    return e["otimizador"], float(e["lr"]), e["lr_adamw"], e["agenda"]


def nome(c):
    o, lr, aw, ag = c
    return f"{o} η={lr:g}" + (f" (AdamW {float(aw):g})" if aw else "") + f" {ag}"


def fmt(x):
    return f"{float(x):.5f}" if x not in ("", None) else "—"


def melhores_eta(exps):
    """{otimizador: execução de menor val_final na varredura, entre as que não divergiram}."""
    melhor = {}
    for e in exps:
        if e["etapa"] != "varredura_lr" or e["status"] == "divergiu":
            continue
        o = e["otimizador"]
        if o not in melhor or float(e["val_final"]) < float(melhor[o]["val_final"]):
            melhor[o] = e
    return melhor


def varredura():
    exps = [e for e in ler() if e["etapa"] == "varredura_lr"]
    melhor = melhores_eta(exps)
    print("| otimizador | η | val_final | val_esperar | val_plantio | status |")
    print("|---|---|---|---|---|---|")
    por_otim = {}
    for e in sorted(exps, key=lambda e: (e["otimizador"], float(e["lr"]))):
        o = e["otimizador"]
        por_otim.setdefault(o, []).append(e)
        marca = " **←**" if melhor.get(o) is e else ""
        aw = f" (AdamW {float(e['lr_adamw']):g})" if e["lr_adamw"] else ""
        print(f"| {o} | {float(e['lr']):g}{aw} | {fmt(e['val_final'])}{marca} | "
              f"{fmt(e['val_esperar'])} | {fmt(e['val_plantio'])} | {e['status']} |")
    print()
    for o, e in melhor.items():
        lrs = sorted(float(x["lr"]) for x in por_otim[o])
        borda = " (BORDA)" if float(e["lr"]) in (lrs[0], lrs[-1]) else ""
        print(f"melhor η de {o}: {float(e['lr']):g}{borda}, val_final {fmt(e['val_final'])}")

    FIGURAS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for o, es in por_otim.items():
        lr = [float(e["lr"]) for e in es]
        v = [float(e["val_final"]) if e["status"] != "divergiu" else np.nan for e in es]
        linha, = ax.plot(lr, v, marker="o", label=o)
        for e in es:
            if e["status"] == "divergiu":
                ax.axvline(float(e["lr"]), color=linha.get_color(), ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("η (no Muon, η das matrizes)")
    ax.set_ylabel("Huber de validação (val_final)")
    ax.set_title("Varredura de η — warmup_cosseno, 10.000 passos, semente 0\n"
                 "(linhas pontilhadas: η que divergiram)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "varredura_eta.png", dpi=120)
    print(FIGURAS / "varredura_eta.png")


def agendas():
    exps = ler()
    melhor = melhores_eta(exps)
    linhas = [e for e in exps if e["etapa"] == "agendas"]
    # a execução warmup_cosseno é a do passo 2 (reaproveitada)
    otims = sorted({e["otimizador"] for e in linhas})
    linhas += [melhor[o] for o in otims]
    print("| otimizador | η | agenda | val_final | val_esperar | val_plantio | status | origem |")
    print("|---|---|---|---|---|---|---|---|")
    for e in sorted(linhas, key=lambda e: (e["otimizador"], float(e["val_final"]))):
        origem = "passo 2 (reaproveitada)" if e["etapa"] == "varredura_lr" else "passo 3"
        aw = f" (AdamW {float(e['lr_adamw']):g})" if e["lr_adamw"] else ""
        print(f"| {e['otimizador']} | {float(e['lr']):g}{aw} | {e['agenda']} | {fmt(e['val_final'])} | "
              f"{fmt(e['val_esperar'])} | {fmt(e['val_plantio'])} | {e['status']} | {origem} |")


def estatisticas(exps):
    grupos = {}
    for e in exps:
        grupos.setdefault(chave(e), {})[int(e["semente"])] = e
    est = {}
    for c, por_semente in grupos.items():
        es = list(por_semente.values())
        est[c] = {m: (np.mean([float(e[m]) for e in es]), np.std([float(e[m]) for e in es], ddof=1))
                  for m in METRICAS}
        est[c]["n"] = len(es)
        est[c]["sementes"] = sorted(por_semente)
    return est


def vence(a, b):
    """True se a vence b pelo critério do hipoteses.md (2 desvios-padrão combinados)."""
    (ma, sa), (mb, sb) = a["val_final"], b["val_final"]
    sc = math.sqrt((sa ** 2 + sb ** 2) / 2)
    return mb - ma > 2 * sc, (mb - ma) / sc if sc > 0 else math.inf


def confirmacao():
    exps = ler()
    base = [e for e in exps if e["etapa"] == "baseline" and e["val_esperar"] != ""]
    conf = [e for e in exps if e["etapa"] == "confirmacao"]
    est = estatisticas(base + conf)
    print("| combinação | n | val_final | val_esperar | val_plantio |")
    print("|---|---|---|---|---|")
    ordem = sorted(est, key=lambda c: est[c]["val_final"][0])
    for c in ordem:
        s = est[c]
        cel = " | ".join(f"{s[m][0]:.5f} ± {s[m][1]:.5f}" for m in METRICAS)
        print(f"| {nome(c)} | {s['n']} | {cel} |")
    print()
    for a, b in itertools.permutations(ordem, 2):
        if est[a]["val_final"][0] < est[b]["val_final"][0]:
            v, z = vence(est[a], est[b])
            print(f"{nome(a)}  vs  {nome(b)}: diferença = {z:.2f} s_c → "
                  f"{'VENCE' if v else 'empate (não passa de 2 s_c)'}")


def rho(log_a, log_b):
    from .graficos import ler as ler_log, serie
    FIGURAS.mkdir(exist_ok=True)
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, pasta in zip(eixos, (log_a, log_b)):
        r, info = ler_log(Path(pasta))
        cfg = info["config"]
        for n in CAMADAS:
            ax.plot(*serie(r, f"rho_{n}"), lw=0.7, label=n)
        aw = f", AdamW η={cfg['lr_adamw']:g}" if cfg.get("lr_adamw") else ""
        ax.set_title(f"{cfg['otimizador']} η={cfg['lr']:g}{aw}, {cfg['agenda']}", fontsize=10)
        ax.set_yscale("log")
        ax.set_xlabel("passo")
        ax.grid(alpha=0.3, which="both")
        ax.axhline(1e-3, color="k", ls="--", lw=0.8)
    eixos[0].set_ylabel("ρ = ‖Δθ‖ / ‖θ‖ por camada")
    eixos[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURAS / "rho_muon_adamw.png", dpi=120)
    print(FIGURAS / "rho_muon_adamw.png")


def pasta_do_log(e):
    return PASTA_LOGS / e["id"]


def instavel():
    pior = None
    for e in ler():
        if e["status"] == "divergiu":
            continue
        with open(pasta_do_log(e) / "registro.csv", encoding="utf-8") as f:
            g = max(float(l["norma_g"]) for l in csv.DictReader(f) if l["norma_g"])
        if pior is None or g > pior[0]:
            pior = (g, e)
    g, e = pior
    print(f"maior pico de ‖g‖ sem divergir: {g:.2f} em {e['id']} ({e['etapa']})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("comando", choices=("varredura", "agendas", "confirmacao", "rho", "instavel"))
    p.add_argument("logs", nargs="*")
    a = p.parse_args()
    {"varredura": varredura, "agendas": agendas, "confirmacao": confirmacao,
     "instavel": instavel}.get(a.comando, lambda: rho(*a.logs))()


if __name__ == "__main__":
    main()
