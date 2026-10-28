"""G1/G2: a rede propaga o valor? Treino off-line nos dados de β (k = 20, γ = 0,994).

    python -m rl.diag.td --modo td --alvo-cada 2000 --arquitetura cnn1 [--id ...]
    python -m rl.diag.td --modo mc --arquitetura cnn1

As mesmas peças do rl/dqn.py: rede de desafio3.modelo, Huber δ = 1 só na ação tomada
(perda_q), Adam 1e-4, lote 256, clip 10, n = 10, rede alvo; amostragem uniforme.
- td: alvo SARSA de n passos = Σ γ^l r_{t+l} (l < n) + γⁿ·Q_alvo(s_{t+n}, a_{t+n}), com a
  ação de β em t+n; sem bootstrap se o episódio acabou antes.
- mc: regressão direta para G.
A cada 5.000 atualizações, na validação, por faixa de posição (< 30, 30–99, ≥ 100): viés
médio Q(s, a_β) − G e Huber. A cada --nota-cada e no fim: nota() na prova de avaliação.
Logs em rl/diag/logs/<id>/ (metricas.csv, notas.csv, resultado.json; pesos.pt fora do git).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from desafio3.modelo import criar, formato
from desafio3.treino import fixar_sementes, perda_q

from . import dados_td, prova
from .comum import GAMMA, entrada

PASTA_LOGS = Path(__file__).parent / "logs"
FAIXAS_POS = ((0, 30), (30, 100), (100, 10 ** 6))


def indices_n_passos(d, n, gamma):
    """R de n passos, índice de s_{t+n} e se há bootstrap, para cada transição."""
    N = len(d["acao"])
    ep, pos, r = d["ep"], d["pos"].astype(np.int64), d["r"].astype(np.float64)
    tam = np.bincount(ep)[ep]                       # comprimento do episódio de cada transição
    R = np.zeros(N)
    for l in range(n):
        ok = pos + l < tam
        R += np.where(ok, gamma ** l * r[np.minimum(np.arange(N) + l, N - 1)], 0.0)
    tem_prox = pos + n < tam
    prox = np.where(tem_prox, np.arange(N) + n, 0)
    return R.astype(np.float32), prox, tem_prox


def tensor_entrada(d, fmt, dev, bloco=8192):
    partes = []
    for i in range(0, len(d["acao"]), bloco):
        sl = slice(i, i + bloco)
        x = entrada(fmt, torch.from_numpy(d["obs"][sl]), torch.from_numpy(d["sol"][sl]),
                    torch.from_numpy(d["mascara"][sl]))
        partes.append(x.half().to(dev))
    return torch.cat(partes)


@torch.no_grad()
def medir_val(rede, X, acao, G, pos, bloco=8192):
    rede.eval()
    q = torch.cat([rede(X[i:i + bloco].float()).gather(1, acao[i:i + bloco, None]).squeeze(1)
                   for i in range(0, len(acao), bloco)])
    rede.train()
    linha = {}
    for nome, (a, b) in zip(("ini", "meio", "fim"), FAIXAS_POS):
        m = (pos >= a) & (pos < b)
        linha[f"vies_{nome}"] = float((q[m] - G[m]).mean())
        linha[f"huber_{nome}"] = float(F.huber_loss(q[m], G[m], delta=1.0))
    linha["vies"] = float((q - G).mean())
    linha["huber"] = float(F.huber_loss(q, G, delta=1.0))
    return linha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modo", choices=("td", "mc"), required=True)
    p.add_argument("--alvo-cada", type=int, default=2000)
    p.add_argument("--arquitetura", default="cnn1")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--atualizacoes", type=int, default=100_000)
    p.add_argument("--medir-cada", type=int, default=5_000)
    p.add_argument("--nota-cada", type=int, default=20_000)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--lote", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--id")
    a = p.parse_args()
    id_ = a.id or (f"{a.modo}{a.alvo_cada if a.modo == 'td' else ''}-{a.arquitetura}-s{a.semente}")
    pasta = PASTA_LOGS / id_
    pasta.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixar_sementes(a.semente)
    fmt = formato(a.arquitetura)

    tr, va = dados_td.carregar("treino"), dados_td.carregar("val")
    R, prox, tem_prox = indices_n_passos(tr, a.n, GAMMA)
    X = tensor_entrada(tr, fmt, dev)
    acao = torch.from_numpy(tr["acao"].astype(np.int64)).to(dev)
    R, prox = torch.from_numpy(R).to(dev), torch.from_numpy(prox).to(dev)
    tem_prox = torch.from_numpy(tem_prox).to(dev)
    G = torch.from_numpy(tr["G"]).to(dev)
    Xv = tensor_entrada(va, fmt, dev)
    av, Gv = torch.from_numpy(va["acao"].astype(np.int64)).to(dev), torch.from_numpy(va["G"]).to(dev)
    pv = torch.from_numpy(va["pos"].astype(np.int64)).to(dev)
    pr = prova.carregar("avaliacao")

    rede, alvo = criar(a.arquitetura).to(dev), criar(a.arquitetura).to(dev)
    alvo.load_state_dict(rede.state_dict())
    alvo.requires_grad_(False)
    opt = torch.optim.Adam(rede.parameters(), lr=a.lr)
    gn = torch.Generator(device="cpu").manual_seed(a.semente)
    gn_pot = GAMMA ** a.n

    metricas, notas = [], []
    t0 = time.time()

    def registrar(t):
        l = {"atualizacao": t, **medir_val(rede, Xv, av, Gv, pv), "segundos": time.time() - t0}
        metricas.append(l)
        if t % a.nota_cada == 0 or t == a.atualizacoes:
            nt = prova.nota(prova.q_candidatas(rede, pr, fmt, dev), pr)
            notas.append({"atualizacao": t, **nt})
            print(f"[{id_}] {t}: viés ini {l['vies_ini']:+.3f} huber ini {l['huber_ini']:.4f} "
                  f"nota {nt['nota']:.1f}% (i {nt['nota_i']:.1f}, ii {nt['nota_ii']:.1f})", flush=True)

    registrar(0)
    for t in range(1, a.atualizacoes + 1):
        idx = torch.randint(len(acao), (a.lote,), generator=gn).to(dev)
        if a.modo == "mc":
            y = G[idx]
        else:
            with torch.no_grad():
                j = prox[idx]
                qj = alvo(X[j].float()).gather(1, acao[j][:, None]).squeeze(1)
                y = R[idx] + torch.where(tem_prox[idx], gn_pot * qj, torch.zeros_like(qj))
        perda = perda_q(rede(X[idx].float()), acao[idx], y, 1.0)
        opt.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(rede.parameters(), 10.0)
        opt.step()
        if a.modo == "td" and t % a.alvo_cada == 0:
            alvo.load_state_dict(rede.state_dict())
        if t % a.medir_cada == 0:
            registrar(t)

    for nome, linhas in (("metricas.csv", metricas), ("notas.csv", notas)):
        with open(pasta / nome, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)
    torch.save(rede.state_dict(), pasta / "pesos.pt")
    (pasta / "resultado.json").write_text(json.dumps(
        {"args": vars(a), "final": metricas[-1], "nota": notas[-1],
         "minutos": round((time.time() - t0) / 60, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
