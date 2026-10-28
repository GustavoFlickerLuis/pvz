"""G3: o teto. A rede ordena as jogadas se receber a resposta?

    python -m rl.diag.teto --arquitetura cnn1d [--atualizacoes 20000]

Regressão supervisionada direta na prova de TREINO: Huber (δ = 1) entre Q(s, a) e Q̂(s, a)
em todas as candidatas de cada estado. Adam 1e-4, clip 10, lotes de 128 estados. 10% dos
estados da prova de treino ficam de fora para escolher o ponto de parada (menor Huber
nesses estados); a nota é medida na prova de AVALIAÇÃO (estados nunca vistos), nesse ponto.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from desafio3.modelo import criar, formato
from desafio3.treino import fixar_sementes

from . import prova
from .comum import entrada

PASTA_LOGS = Path(__file__).parent / "logs"


def preparar(pr, idx, fmt, dev):
    x = entrada(fmt, torch.from_numpy(pr["obs"][idx]), torch.from_numpy(pr["sol"][idx]),
                torch.from_numpy(pr["mascara"][idx])).to(dev)
    c = torch.from_numpy(pr["cand"][idx].astype(np.int64)).to(dev)
    q = torch.from_numpy(np.nan_to_num(pr["q"][idx, :, 0])).to(dev)
    return x, c, q


def perda(rede, x, c, q):
    valida = c >= 0
    pred = rede(x).gather(1, c.clamp(min=0))
    return F.huber_loss(pred[valida], q[valida], delta=1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arquitetura", default="cnn1d")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--atualizacoes", type=int, default=20_000)
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--medir-cada", type=int, default=500)
    a = p.parse_args()
    id_ = f"teto-{a.arquitetura}-s{a.semente}"
    pasta = PASTA_LOGS / id_
    pasta.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixar_sementes(a.semente)
    fmt = formato(a.arquitetura)
    pt, pa = prova.carregar("treino"), prova.carregar("avaliacao")
    ordem = np.random.default_rng(a.semente).permutation(len(pt["pos"]))
    fora = ordem[:len(ordem) // 10]
    dentro = ordem[len(ordem) // 10:]
    xt, ct, qt = preparar(pt, dentro, fmt, dev)
    xf, cf, qf = preparar(pt, fora, fmt, dev)
    rede = criar(a.arquitetura).to(dev)
    opt = torch.optim.Adam(rede.parameters(), lr=1e-4)
    gn = torch.Generator().manual_seed(a.semente)
    melhor, melhor_t, melhor_pesos, linhas = float("inf"), 0, None, []
    t0 = time.time()
    for t in range(1, a.atualizacoes + 1):
        idx = torch.randint(len(dentro), (a.lote,), generator=gn).to(dev)
        l = perda(rede, xt[idx], ct[idx], qt[idx])
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(rede.parameters(), 10.0)
        opt.step()
        if t % a.medir_cada == 0:
            with torch.no_grad():
                h = float(perda(rede, xf, cf, qf))
            linhas.append({"atualizacao": t, "huber_treino": float(l), "huber_fora": h})
            if h < melhor:
                melhor, melhor_t, melhor_pesos = h, t, copy.deepcopy(rede.state_dict())
    rede.load_state_dict(melhor_pesos)
    nt = prova.nota(prova.q_candidatas(rede, pa, fmt, dev), pa)
    nt_dentro = prova.nota(prova.q_candidatas(rede, {k: v[dentro] for k, v in pt.items()}, fmt, dev),
                           {k: v[dentro] for k, v in pt.items()})
    print(f"[{id_}] parada em {melhor_t} (huber fora {melhor:.5f}); nota na avaliação "
          f"{nt['nota']:.1f}% (i {nt['nota_i']:.1f}, ii {nt['nota_ii']:.1f}); "
          f"nos próprios estados de treino {nt_dentro['nota']:.1f}%", flush=True)
    with open(pasta / "curva.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    torch.save(melhor_pesos, pasta / "pesos.pt")
    (pasta / "resultado.json").write_text(json.dumps(
        {"args": vars(a), "parada": melhor_t, "huber_fora": melhor, "nota": nt,
         "nota_nos_estados_de_treino": nt_dentro,
         "minutos": round((time.time() - t0) / 60, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
