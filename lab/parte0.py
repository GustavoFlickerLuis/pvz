"""Lab, Parte 0 (só logs; protocolo em LAB.md, "Continuação"): 0a oscilação × ruído e 0b tempo.

    python -m lab.parte0            # tabelas em markdown na saída
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

LOGS = Path(__file__).parent.parent / "rl" / "logs"
GRUPOS = [("F1 atual", "lab-f1-atual-s{}", (1, 3, 4, 2, 6)), ("F1 v2a", "lab-f1-v2a-s{}", (1, 3, 4, 2, 6)),
          ("F1 v2b", "lab-f1-v2b-s{}", (1, 3, 4, 2, 6)),
          ("A cnn1d (v2a)", "lab-f1-v2a-s{}", (1, 2, 3)), ("B atenção antes", "lab-f2-atencao_antes-s{}", (1, 2, 3)),
          ("C atenção depois", "lab-f2-atencao_depois-s{}", (1, 2, 3)), ("D transformer", "lab-f2-transformer-s{}", (1, 2, 3)),
          ("D2 transformer d128", "lab-f2-transformer_d128-s{}", (1, 2, 3)), ("E dilatada", "lab-f2-cnn1d_dilatada-s{}", (1, 2, 3))]
JANELA = (1_000_000, 2_000_896)


def periodicas(pasta, modo="gulosa"):
    with open(pasta / "avaliacoes.csv", encoding="utf-8") as f:
        return [(int(l["passo"]), float(l["vitorias"])) for l in csv.DictReader(f)
                if l["modo"] == modo and l["final"] == "0"]


def razao(pasta, modo="gulosa"):
    """(n, p̄, observado, esperado, razão, leitura) nas periódicas de 1,0M a 2,0M."""
    v = np.array([x for p, x in periodicas(pasta, modo) if JANELA[0] <= p <= JANELA[1]])
    if len(v) < 2:
        return None
    pm, obs = v.mean(), v.std(ddof=1)
    esp = math.sqrt(pm * (1 - pm) / 63)
    r = obs / esp if esp > 0 else float("nan")
    ler = "—" if math.isnan(r) else "ruído" if r <= 1.5 else "oscilação real" if r > 2 else "inconclusivo"
    return len(v), pm, obs, esp, r, ler


def tempo(pasta, limiar, modo="gulosa"):
    """(passo, minutos) da 1ª média móvel de 3 periódicas > limiar; None = nunca."""
    av = periodicas(pasta, modo)
    with open(pasta / "registro.csv", encoding="utf-8") as f:
        vel = {int(l["passo"]): float(l["passos_por_s"]) for l in csv.DictReader(f)}
    for i in range(2, len(av)):
        if np.mean([x for _, x in av[i - 2:i + 1]]) > limiar:
            p = av[i][0]
            return p, p / vel[p] / 60
    return None


def entropia_final(pasta, ult=10):
    with open(pasta / "registro.csv", encoding="utf-8") as f:
        e = [float(l["entropia"]) for l in csv.DictReader(f)]
    return float(np.mean(e[-ult:]))


def _f(x, fmt="{:.2f}"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def _t(t):
    return "nunca" if t is None else f"{t[0] / 1e6:.2f}M / {t[1]:.1f} min"


def _med_t(ts):
    """Mediana de passos e minutos; 'nunca' conta como infinito."""
    if not ts:
        return "—"
    ps = [t[0] if t else math.inf for t in ts]
    ms = [t[1] if t else math.inf for t in ts]
    mp, mm = float(np.median(ps)), float(np.median(ms))
    return "nunca" if math.isinf(mp) else f"{mp / 1e6:.2f}M / {mm:.1f} min"


def tabelas(grupos=GRUPOS, modo="gulosa"):
    out = [f"**0a. Oscilação × ruído** (periódicas {modo}, 63 partidas, 1,0M–2,0M)", "",
           "| braço | semente | n | p̄ | dp observado | dp esperado | razão | leitura |", "|---|---|---|---|---|---|---|---|"]
    med = []
    for nome, molde, sems in grupos:
        rs = []
        for s in sems:
            r = razao(LOGS / molde.format(s), modo)
            if r is None:
                continue
            n, pm, obs, esp, q, ler = r
            rs.append(q)
            out.append(f"| {nome} | {s} | {n} | {100 * pm:.1f} | {100 * obs:.2f} | {100 * esp:.2f} | {_f(q)} | {ler} |")
        if rs:
            m = float(np.median(rs))
            med.append((nome, m))
    out += ["", "Mediana da razão por braço:", "", "| braço | mediana | leitura |", "|---|---|---|"]
    for nome, m in med:
        ler = "—" if math.isnan(m) else "ruído" if m <= 1.5 else "oscilação real" if m > 2 else "inconclusivo"
        out.append(f"| {nome} | {_f(m)} | {ler} |")
    out += ["", f"**0b. Tempo até a média móvel de 3 periódicas ({modo}) passar de 80% e de 90%** (passo / minutos de relógio)", "",
            "| braço | semente | 80% | 90% |", "|---|---|---|---|"]
    meds = []
    for nome, molde, sems in grupos:
        t8, t9 = [], []
        for s in sems:
            p = LOGS / molde.format(s)
            if not (p / "avaliacoes.csv").exists():
                continue
            a, b = tempo(p, 0.8, modo), tempo(p, 0.9, modo)
            t8.append(a), t9.append(b)
            out.append(f"| {nome} | {s} | {_t(a)} | {_t(b)} |")
        if t8:
            meds.append((nome, _med_t(t8), _med_t(t9)))
    out += ["", "Mediana por braço ('nunca' conta como infinito):", "", "| braço | 80% | 90% |", "|---|---|---|"]
    out += [f"| {n} | {a} | {b} |" for n, a, b in meds]
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(tabelas())
