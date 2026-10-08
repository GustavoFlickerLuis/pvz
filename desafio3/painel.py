"""painel.pdf do Desafio 3: python -m desafio3.painel

Páginas:
1. Os quatro sinais alinhados no mesmo eixo de passos (perda de treino e de validação,
   ‖g‖, ‖θ‖, η) para a vencedora, a baseline, o Muon 1e-3 e o SGD 0,3 (instável).
2. val_final × η por otimizador (a varredura).
3. ρ por camada, Muon × AdamW, lado a lado.
4. As tabelas finais (treino × validação, velocidade) e o resultado do teste.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import analise
from .graficos import ler as ler_log
from .graficos import serie
from .treino import PASTA_LOGS

SAIDA = Path(__file__).parent / "painel.pdf"
TESTE = Path(__file__).parent / "teste.csv"


def achar(filtro):
    """Log (pasta) da semente 0 que satisfaz o filtro."""
    es = [e for e in analise.ler() if filtro(e) and e["semente"] == "0"]
    assert len(es) >= 1
    return PASTA_LOGS / es[0]["id"]


def logs_sinais():
    return {
        "vencedora: sgd 3e-2 cosseno": achar(lambda e: e["etapa"] == "confirmacao"
                                             and e["otimizador"] == "sgd" and e["agenda"] == "cosseno"),
        "baseline: adam 1e-4 constante": achar(lambda e: e["etapa"] == "baseline"),
        "muon 1e-3 warmup_cosseno": achar(lambda e: e["etapa"] == "confirmacao"
                                          and e["otimizador"] == "muon"),
        "instável: sgd 0,3 warmup_cosseno": achar(lambda e: e["etapa"] == "extra"
                                                  and e["otimizador"] == "sgd"),
    }


def pagina_sinais(pdf):
    logs = logs_sinais()
    fig, eixos = plt.subplots(4, 4, figsize=(17, 11), sharex=True, sharey="row")
    for j, (rotulo, pasta) in enumerate(logs.items()):
        r, info = ler_log(pasta)
        res = info["resultado"]
        col = eixos[:, j]
        col[0].set_title(f"{rotulo}\nval_final {res['val_final']:.4f}", fontsize=10)
        col[0].plot(*serie(r, "perda_lote"), lw=0.5, alpha=0.5, label="treino (lote)")
        col[0].plot(*serie(r, "huber_treino"), marker=".", ms=3, label="treino (lote fixo)")
        col[0].plot(*serie(r, "huber_val"), marker=".", ms=3, label="validação")
        col[0].set_yscale("log")
        col[1].plot(*serie(r, "norma_g"), lw=0.6)
        col[1].axhline(info["config"]["clip"], color="k", ls="--", lw=0.8)
        col[1].set_yscale("log")
        col[2].plot(*serie(r, "norma_theta"))
        col[2].set_yscale("log")
        for c in sorted(k for k in r if k.startswith("eta_")):
            nome = {"eta_0": "η" if "eta_1" not in r else "η Muon", "eta_1": "η AdamW"}[c]
            col[3].plot(*serie(r, c), label=nome)
        col[3].set_yscale("log")
        col[3].legend(fontsize=7)
        col[3].set_xlabel("passo")
        for ax in col:
            ax.grid(alpha=0.3, which="both")
    eixos[0, 0].legend(fontsize=7)
    for ax, nome in zip(eixos[:, 0], ("Huber", "‖g‖ (antes do corte; -- corte = 10)", "‖θ‖", "η")):
        ax.set_ylabel(nome)
    fig.suptitle("Os quatro sinais alinhados no mesmo eixo de passos (semente 0; 10.000 passos, "
                 "lote 256)", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pagina_varredura(pdf):
    fig, ax = plt.subplots(figsize=(11, 7))
    analise.desenhar_varredura(ax, analise.por_otimizador_varredura())
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pagina_rho(pdf):
    muon = achar(lambda e: e["etapa"] == "confirmacao" and e["otimizador"] == "muon")
    adamw = achar(lambda e: e["etapa"] == "varredura_lr" and e["otimizador"] == "adamw"
                  and math.isclose(float(e["lr"]), 3e-4))
    fig, eixos = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    analise.desenhar_rho(eixos, (muon, adamw))
    fig.suptitle("ρ por camada: Muon (convs no Muon, cabeças e vieses no AdamW) × AdamW "
                 "(-- referência 1e-3)", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _tabela(ax, titulo, cabecalho, linhas, larguras):
    ax.axis("off")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    t = ax.table(cellText=linhas, colLabels=cabecalho, colWidths=larguras, loc="upper left",
                 cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.35)


def pagina_tabelas(pdf):
    dados = analise.tabelas(escrever=False)
    base = dados[analise.CONFIGS[0][0]]
    m = lambda x, k: np.mean([e[k] for e in x])
    ms = lambda x, k: (f"{m(x, k):.4f}" if len(x) == 1 else
                       f"{m(x, k):.4f} ± {np.std([e[k] for e in x], ddof=1):.4f}")
    a, b = [], []
    for rotulo, _ in analise.CONFIGS:
        x = dados[rotulo]
        a.append([rotulo, len(x), ms(x, "huber_treino"), ms(x, "val_final"), ms(x, "val_esperar"),
                  ms(x, "val_plantio"), f"{m(x, 'huber_treino') - m(base, 'huber_treino'):+.4f}",
                  f"{m(x, 'val_final') - m(base, 'val_final'):+.4f}",
                  ", ".join(str(e["passo_val_min"]) for e in x)])
        b.append([rotulo, ", ".join("nunca" if e["passos_alvo"] is None else str(e["passos_alvo"])
                                    for e in x), f"{m(x, 'ms_passo'):.2f}"])
    with open(TESTE, newline="", encoding="utf-8") as f:
        teste = list(csv.DictReader(f))
    c = []
    for rotulo in dict.fromkeys(l["config"] for l in teste):
        x = [{k: float(l[k]) for k in ("teste_huber", "teste_esperar", "teste_plantio")}
             for l in teste if l["config"] == rotulo]
        c.append([rotulo, len(x)] + [ms(x, k) for k in ("teste_huber", "teste_esperar", "teste_plantio")])

    fig = plt.figure(figsize=(17, 11))
    eixos = fig.subplots(3, 1, gridspec_kw={"height_ratios": [1.25, 1.0, 0.6]})
    _tabela(eixos[0], "a) Treino × validação (média ± desvio, 5 sementes; Δ contra a baseline)",
            ["config", "n", "huber_treino", "val_final", "val_esperar", "val_plantio", "Δtreino",
             "Δval", "passo do val_min"], a, [0.2, 0.03, 0.1, 0.1, 0.1, 0.1, 0.06, 0.06, 0.17])
    _tabela(eixos[1], f"b) Velocidade (secundária): passos até val ≤ {analise.ALVO_VELOCIDADE} "
            "(subestima as agendas que decaem: medido dentro de uma execução de 10.000 passos)",
            ["config", "passos até o alvo (por semente)", "ms/passo"], b, [0.25, 0.35, 0.08])
    _tabela(eixos[2], "Teste (avaliado uma vez; Huber, média ± desvio, 5 sementes). "
            "Vencedora × baseline: empate (1,93 s_c); ambas vencem o Muon.",
            ["config", "n", "Huber total", "esperar", "plantio"], c, [0.25, 0.03, 0.13, 0.13, 0.13])
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(SAIDA) as pdf:
        pagina_sinais(pdf)
        pagina_varredura(pdf)
        pagina_rho(pdf)
        pagina_tabelas(pdf)
    print(SAIDA)


if __name__ == "__main__":
    main()
