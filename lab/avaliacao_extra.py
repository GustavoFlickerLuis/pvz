"""Lab: avaliação extra, sem treino (protocolo em LAB.md, "Avaliação extra").

    python -m lab.avaliacao_extra            # roda o que falta (retoma pelo CSV)
    python -m lab.avaliacao_extra --tabela   # só imprime as tabelas

1. Estresse: modelo.pt × FASES_TREINO × 300–319 × dificuldade 1,0 / 1,25 / 1,5; Econ5 e Guloso.
2. Pico × final: melhor.pt e modelo.pt × FASES_TREINO × 320–339, dificuldade 1,0.
Nunca nas fases de teste (o assert está em rl.avaliar.partidas). Para no lab/PARAR.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pvz.agentes import Econ, Guloso
from rl import avaliar
from rl.ppo import Gulosa, criar_politica

RAIZ = Path(__file__).parent
LOGS = RAIZ.parent / "rl" / "logs"
CSV = RAIZ / "avaliacao_extra.csv"
PARAR = RAIZ / "PARAR"
SEMENTES = (1, 2, 3)
DIFS = (1.0, 1.25, 1.5)
BRACOS = [("A cnn1d", "lab-f1-v2a-s{}"), ("B atenção antes", "lab-f2-atencao_antes-s{}"),
          ("C atenção depois", "lab-f2-atencao_depois-s{}"), ("D transformer", "lab-f2-transformer-s{}"),
          ("D2 transformer d128", "lab-f2-transformer_d128-s{}"), ("E dilatada", "lab-f2-cnn1d_dilatada-s{}"),
          # Parte 1 (só entram as execuções que já terminaram)
          ("D v2b", "lab-f3-D-v2b-s{}"), ("D coef_ent 0,01", "lab-f3-D-ent01-s{}")]
CAMPOS = ["tarefa", "braco", "id", "semente", "modelo", "dificuldade", "sementes_aval", "partidas",
          "vitorias", "dano", "passo_modelo", "ref_final", "segundos"]


def passo_melhor(pasta):
    """Passo do melhor.pt: a primeira avaliação periódica gulosa com a maior vitória."""
    with open(pasta / "avaliacoes.csv", encoding="utf-8") as f:
        linhas = [l for l in csv.DictReader(f) if l["modo"] == "gulosa" and l["final"] == "0"]
    m = max(float(l["vitorias"]) for l in linhas)
    return int(next(l["passo"] for l in linhas if float(l["vitorias"]) == m))


def tarefas():
    """(chave, descrição) de cada avaliação, na ordem em que rodam."""
    t = []
    # correção: os baselines decidem com k = 20, como o PPO (as linhas "econ5"/"guloso" sem
    # sufixo, da primeira rodada, usaram o k padrão do avaliar_agente, 10; ficam só de registro)
    for nome, ag in (("Econ5 (k=20)", "econ5-k20"), ("Guloso (k=20)", "guloso-k20")):
        for d in DIFS:
            t.append(dict(tarefa="estresse", braco=nome, id=ag, semente="", modelo="", dificuldade=d,
                          sementes_aval="300-319"))
    for braco, molde in BRACOS:
        for s in SEMENTES:
            i = molde.format(s)
            if not (LOGS / i / "resultado.json").exists():
                continue
            for d in DIFS:
                t.append(dict(tarefa="estresse", braco=braco, id=i, semente=s, modelo="modelo.pt",
                              dificuldade=d, sementes_aval="300-319"))
            for m in ("melhor.pt", "modelo.pt"):
                t.append(dict(tarefa="pico_final", braco=braco, id=i, semente=s, modelo=m,
                              dificuldade=1.0, sementes_aval="320-339"))
    return t


def chave(r):
    return (r["tarefa"], r["id"], str(r["modelo"]), f"{float(r['dificuldade']):.2f}")


def feitas():
    if not CSV.exists():
        return {}
    with open(CSV, encoding="utf-8") as f:
        return {chave(r): r for r in csv.DictReader(f)}


def rodar_uma(t, dev):
    a, b = map(int, t["sementes_aval"].split("-"))
    lista = avaliar.partidas(range(a, b + 1))
    t0 = time.time()
    if t["semente"] == "":
        ag = Econ(5) if t["id"].startswith("econ5") else Guloso()
        r = avaliar.avaliar_agente(ag, lista, t["dificuldade"], k=20 if t["id"].endswith("-k20") else 10)
        extra = dict(passo_modelo="", ref_final="")
    else:
        pasta = LOGS / t["id"]
        cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))["config"]
        pol = criar_politica(cfg["arquitetura"]).to(dev)
        pol.load_state_dict(torch.load(pasta / t["modelo"], map_location=dev))
        r = avaliar.avaliar_rede(Gulosa(pol), lista, dev, t["dificuldade"], k=cfg["k_ticks"], formato="v2")
        res = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
        extra = dict(passo_modelo=passo_melhor(pasta) if t["modelo"] == "melhor.pt" else res["passos"],
                     ref_final=res["vitorias_final"])
    return {**t, **extra, "partidas": r["partidas"], "vitorias": r["vitorias"], "dano": r["dano"],
            "segundos": round(time.time() - t0, 1)}


def rodar():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ja = feitas()
    novo = not CSV.exists()
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CAMPOS)
        if novo:
            w.writeheader()
        for t in tarefas():
            if chave(t) in ja:
                continue
            if PARAR.exists():
                print(f"[parada] {PARAR} existe; para retomar, apague e chame de novo", flush=True)
                return
            r = rodar_uma(t, dev)
            w.writerow(r)
            f.flush()
            ok = ""
            if r["ref_final"] != "" and t["tarefa"] == "estresse" and t["dificuldade"] == 1.0:
                ok = " (bate com o final)" if abs(r["vitorias"] - float(r["ref_final"])) < 1e-9 else \
                     f" (NÃO BATE: final {100 * float(r['ref_final']):.1f}%)"
            print(f"[aval] {t['tarefa']} {t['braco']} {t['id']} {t['modelo']} dif {t['dificuldade']} "
                  f"{t['sementes_aval']}: {100 * r['vitorias']:.1f}%{ok} ({r['segundos']} s)", flush=True)
    print("[fim] avaliação extra completa", flush=True)


def _md(v):
    return f"{100 * np.mean(v):.1f} ± {100 * np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{100 * v[0]:.1f}"


def _sc(a, b):
    return math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)


def _veredito(a, b):
    """a: braço, b: referência; devolve Δ, 2·s_c (em pontos) e o veredito."""
    d, l = 100 * (np.mean(a) - np.mean(b)), 200 * _sc(a, b)
    return d, l, ("ganhou" if d > l else "perdeu" if d < -l else "empate")


def tabela():
    r = list(feitas().values())
    v = lambda **k: [float(x["vitorias"]) for x in sorted(r, key=lambda x: x["semente"])
                     if all(str(x[c]) == str(val) for c, val in k.items())]
    out = ["**Estresse** (modelo.pt, FASES_TREINO × 300–319, 420 partidas; média ± dp entre sementes)", "",
           "| braço | dif 1,0 | dif 1,25 | dif 1,5 | queda 1,0→1,5 | Δ queda vs cnn1d | 2·s_c | veredito (queda) |",
           "|---|---|---|---|---|---|---|---|"]
    ref_q = None
    for nome in ["Econ5 (k=20)", "Guloso (k=20)"] + [b for b, _ in BRACOS]:
        cols = [v(tarefa="estresse", braco=nome, dificuldade=d) for d in DIFS]
        if not all(cols):
            continue
        q = list(np.array(cols[0]) - np.array(cols[2]))
        if nome.startswith("A "):
            ref_q = q
        cmp = ["", "", ""]
        if ref_q is not None and len(q) > 1 and not nome.startswith("A "):
            d, l, ver = _veredito(ref_q, q)   # positivo = o braço cai menos que a cnn1d
            cmp = [f"{d:+.1f}", f"{l:.1f}", ver.replace("ganhou", "cai menos").replace("perdeu", "cai mais")]
        out.append(f"| {nome} | " + " | ".join(_md(c) for c in cols) + f" | {_md(q)} | " + " | ".join(cmp) + " |")
    out += ["", "Nível por dificuldade, braço × cnn1d (Δ / 2·s_c / veredito):", "",
            "| braço | dif 1,0 | dif 1,25 | dif 1,5 |", "|---|---|---|---|"]
    for nome, _ in BRACOS[1:]:
        cel = []
        for d in DIFS:
            a, b = v(tarefa="estresse", braco=nome, dificuldade=d), v(tarefa="estresse", braco="A cnn1d", dificuldade=d)
            if len(a) > 1 and len(b) > 1:
                dd, l, ver = _veredito(a, b)
                cel.append(f"{dd:+.1f} / {l:.1f} / {ver}")
            else:
                cel.append("")
        out.append(f"| {nome} | " + " | ".join(cel) + " |")
    out += ["", "**Pico × final** (FASES_TREINO × 320–339, 420 partidas, dificuldade 1,0)", "",
            "| braço | melhor.pt s1 / s2 / s3 | passo do melhor | pico | final (mesmas partidas) | pico − final | pico: Δ vs cnn1d | 2·s_c | veredito |",
            "|---|---|---|---|---|---|---|---|---|"]
    ref = v(tarefa="pico_final", braco="A cnn1d", modelo="melhor.pt")
    for nome, _ in BRACOS:
        p, f = v(tarefa="pico_final", braco=nome, modelo="melhor.pt"), v(tarefa="pico_final", braco=nome, modelo="modelo.pt")
        if not p:
            continue
        passos = [x["passo_modelo"] for x in sorted(r, key=lambda x: x["semente"])
                  if x["tarefa"] == "pico_final" and x["braco"] == nome and x["modelo"] == "melhor.pt"]
        cmp = ["", "", ""]
        if not nome.startswith("A ") and len(p) > 1 and len(ref) > 1:
            d, l, ver = _veredito(p, ref)
            cmp = [f"{d:+.1f}", f"{l:.1f}", ver]
        dif = list(np.array(p) - np.array(f)) if len(f) == len(p) else []
        out.append(f"| {nome} | " + " / ".join(f"{100 * x:.1f}" for x in p)
                   + f" | {' / '.join(f'{int(x) / 1e6:.1f}M' for x in passos)} | {_md(p)} | {_md(f) if f else ''} | "
                   + (_md(dif) if dif else "") + " | " + " | ".join(cmp) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tabela", action="store_true")
    a = p.parse_args()
    if not a.tabela:
        rodar()
    tabela()
