"""Diagnóstico do teto por braço (só registro, sem veredito), com a nota separada em
pares "longe" e "perto" (definição no LAB.md, escrita antes de ver as notas).

    python -m lab.teto --arquitetura transformer [--semente 0]

O treino é o de rl/diag/teto.py (Huber nas candidatas da prova de TREINO, Adam 1e-4,
clip 10, lotes de 128, parada pela menor Huber nos 10% de fora); a nota é medida na prova
de AVALIAÇÃO. Logs em rl/diag/logs/lab-teto-<arq>-s<semente>/.

Definição (LAB.md, "longe" × "perto"), com a observação v1 guardada na prova:
- zumbi na célula (f, c) ⇔ canais 7 + 8 + 9 > 0 em (f, c);
- para um plantio candidato em (f, c): dist = menor |c' − c| entre as células (f, c') da
  mesma faixa com zumbi; sem zumbi na faixa, dist = indefinida. "Esperar" não tem dist;
- candidata "longe" ⇔ dist > 3 (fora do campo de visão de ±3 colunas da cnn1d);
  "perto" ⇔ dist ≤ 3;
- par "longe" ⇔ pelo menos uma das duas candidatas é longe e nenhuma é perto;
  par "perto" ⇔ qualquer outro par (inclui os sem zumbi nas faixas das candidatas, que
  também são contados à parte como "sem_zumbi").
"""
import argparse
import copy
import csv
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from desafio3.modelo import criar, formato
from desafio3.treino import fixar_sementes
from pvz import config as C
from pvz.ambiente import ESPERAR, N_CELULAS, decodificar_acao
from rl.diag import prova
from rl.diag.teto import PASTA_LOGS, perda, preparar

LIMITE = 3


def classe_candidata(obs, a):
    """'longe', 'perto', 'sem' (faixa sem zumbi) ou None (esperar)."""
    if a == ESPERAR:
        return None
    _, f, c = decodificar_acao(int(a))
    zumbi = np.asarray(obs[7:10, f, :], dtype=np.float32).sum(0) > 0
    if not zumbi.any():
        return "sem"
    dist = np.abs(np.flatnonzero(zumbi) - c).min()
    return "longe" if dist > LIMITE else "perto"


def classe_par(ca, cb):
    if "perto" in (ca, cb):
        return "perto"
    if "longe" in (ca, cb):
        return "longe"
    return "sem_zumbi"


def nota_por_grupo(q_rede, pr, limiar=0.02):
    """Como prova.nota (mesmos pares claros), separada por classe do par."""
    acertos = {"longe": [], "perto": [], "sem_zumbi": []}
    for s in range(len(q_rede)):
        n = int(pr["n_cand"][s])
        qh, ep, qr = pr["q"][s, :n, 0], pr["q"][s, :n, 1], q_rede[s, :n]
        cls = [classe_candidata(pr["obs"][s], pr["cand"][s, i]) for i in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                d = qh[a] - qh[b]
                if abs(d) > limiar and abs(d) > 2 * np.sqrt(ep[a] ** 2 + ep[b] ** 2):
                    dr = qr[a] - qr[b]
                    ok = 0.5 if dr == 0 else float(np.sign(dr) == np.sign(d))
                    acertos[classe_par(cls[a], cls[b])].append(ok)
    pct = lambda l: 100 * float(np.mean(l)) if l else float("nan")
    out = {f"nota_{k}": pct(v) for k, v in acertos.items()}
    out.update({f"pares_{k}": len(v) for k, v in acertos.items()})
    perto = acertos["perto"] + acertos["sem_zumbi"]
    out["nota_perto_total"], out["pares_perto_total"] = pct(perto), len(perto)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arquitetura", default="cnn1d")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--atualizacoes", type=int, default=20_000)
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--medir-cada", type=int, default=500)
    p.add_argument("--so-contar", action="store_true", help="só conta os pares por classe")
    a = p.parse_args()
    pt, pa = prova.carregar("treino"), prova.carregar("avaliacao")
    if a.so_contar:
        print(json.dumps(nota_por_grupo(pa["q"][:, :, 0], pa), indent=1))
        return
    id_ = f"lab-teto-{a.arquitetura}-s{a.semente}"
    pasta = PASTA_LOGS / id_
    pasta.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixar_sementes(a.semente)
    fmt = formato(a.arquitetura)
    ordem = np.random.default_rng(a.semente).permutation(len(pt["pos"]))
    fora, dentro = ordem[:len(ordem) // 10], ordem[len(ordem) // 10:]
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
    q = prova.q_candidatas(rede, pa, fmt, dev)
    nt = prova.nota(q, pa)
    ng = nota_por_grupo(q, pa)
    print(f"[{id_}] parada em {melhor_t} (huber fora {melhor:.5f}); nota {nt['nota']:.1f}% "
          f"(ii {nt['nota_ii']:.1f}); longe {ng['nota_longe']:.1f}% ({ng['pares_longe']} pares), "
          f"perto {ng['nota_perto_total']:.1f}% ({ng['pares_perto_total']})", flush=True)
    with open(pasta / "curva.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    (pasta / "resultado.json").write_text(json.dumps(
        {"args": vars(a), "parada": melhor_t, "huber_fora": melhor, "nota": nt, "grupos": ng,
         "minutos": round((time.time() - t0) / 60, 1)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
