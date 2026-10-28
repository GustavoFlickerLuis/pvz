"""Lab, Parte 1: tabelas das execuções novas do D (protocolo em LAB.md, "Continuação").

    python -m lab.parte1     # depois de `python -m lab.avaliacao_extra` (finais em 320–339 e estresse)
"""
import csv
import json
import math
import sys

import numpy as np

from lab.avaliacao_extra import CSV
from lab.parte0 import LOGS, razao, tabelas, tempo

SEMS = (1, 2, 3)
BRACOS = [("A cnn1d (v2a)", "lab-f1-v2a-s{}"), ("D (lab-f2, v2a)", "lab-f2-transformer-s{}"),
          ("D v2b", "lab-f3-D-v2b-s{}"), ("D coef_ent 0,01", "lab-f3-D-ent01-s{}")]
NOVOS = BRACOS[2:]


def extra():
    with open(CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metricas(i, ex):
    p = LOGS / i
    res = json.loads((p / "resultado.json").read_text(encoding="utf-8"))
    with open(p / "registro.csv", encoding="utf-8") as f:
        reg = [(int(l["passo"]), float(l["entropia"])) for l in csv.DictReader(f)]
    ent = lambda alvo: next(e for q, e in reg if q >= alvo)
    pega = lambda **k: next((float(r["vitorias"]) for r in ex if r["id"] == i
                             and all(r[c] == v for c, v in k.items())), math.nan)
    rg, rs = razao(p, "gulosa"), razao(p, "sorteio")
    return {"final": res["vitorias_final"],
            "novas": pega(tarefa="pico_final", modelo="modelo.pt"),
            "melhor_novas": pega(tarefa="pico_final", modelo="melhor.pt"),
            "d125": pega(tarefa="estresse", dificuldade="1.25"), "d15": pega(tarefa="estresse", dificuldade="1.5"),
            "razao_g": rg[4] if rg else math.nan, "razao_s": rs[4] if rs else math.nan,
            "t80": tempo(p, 0.8), "t90": tempo(p, 0.9),
            "ent_05": ent(500_000), "ent_1": ent(1_000_000), "ent_fim": float(np.mean([e for _, e in reg[-10:]])),
            "minutos": res["minutos"]}


def _v(a, b):
    a, b = np.array(a), np.array(b)
    d = 100 * (a.mean() - b.mean())
    l = 200 * math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return f"{d:+.1f} / {l:.1f} / " + ("ganhou" if d > l else "perdeu" if d < -l else "empate")


def _ms(v, esc=100, casas=1):
    v = np.array(v, float) * esc
    return f"{v.mean():.{casas}f} ± {v.std(ddof=1):.{casas}f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ex = extra()
    M = {nome: [metricas(m.format(s), ex) for s in SEMS] for nome, m in BRACOS}
    out = ["**Execuções novas, por semente** (s1 / s2 / s3)", "",
           "| braço | final 300–319 | final 320–339 | melhor.pt 320–339 | dif 1,25 | dif 1,5 | razão 0a gulosa | razão 0a sorteio | entropia 0,5M / 1M / fim | min |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda xs, esc=100, c=1: " / ".join(f"{x * esc:.{c}f}" for x in xs)
    for nome, ms in M.items():
        out.append(f"| {nome} | {f([m['final'] for m in ms])} | {f([m['novas'] for m in ms])} | "
                   f"{f([m['melhor_novas'] for m in ms])} | {f([m['d125'] for m in ms])} | {f([m['d15'] for m in ms])} | "
                   f"{f([m['razao_g'] for m in ms], 1, 2)} | {f([m['razao_s'] for m in ms], 1, 2)} | "
                   + " ; ".join(f"{m['ent_05']:.2f}/{m['ent_1']:.2f}/{m['ent_fim']:.2f}" for m in ms)
                   + f" | {f([m['minutos'] for m in ms], 1, 0)} |")
    out += ["", "**Médias ± dp entre sementes**", "",
            "| braço | final 300–319 | final 320–339 | dif 1,25 | dif 1,5 | mediana razão gulosa | mediana razão sorteio | entropia final | 80% (passo, mediana) | 90% (passo, mediana) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for nome, ms in M.items():
        med = lambda k: float(np.median([m[k] for m in ms]))
        mt = lambda k: (lambda v: "nunca" if math.isinf(v) else f"{v / 1e6:.2f}M")(
            float(np.median([m[k][0] if m[k] else math.inf for m in ms])))
        out.append(f"| {nome} | {_ms([m['final'] for m in ms])} | {_ms([m['novas'] for m in ms])} | "
                   f"{_ms([m['d125'] for m in ms])} | {_ms([m['d15'] for m in ms])} | {med('razao_g'):.2f} | "
                   + (f"{med('razao_s'):.2f}" if not math.isnan(med('razao_s')) else "—")
                   + f" | {_ms([m['ent_fim'] for m in ms], 1, 2)} | {mt('t80')} | {mt('t90')} |")
    out += ["", "**Vereditos pela regra** (Δ média / 2·s_c / veredito; o primeiro é a métrica principal)", "",
            "| comparação | final 300–319 | final 320–339 | dif 1,25 | dif 1,5 |", "|---|---|---|---|---|"]
    for novo, _ in NOVOS:
        for ref in ("D (lab-f2, v2a)", "A cnn1d (v2a)"):
            out.append(f"| {novo} × {ref} | " + " | ".join(
                _v([m[k] for m in M[novo]], [m[k] for m in M[ref]]) for k in ("final", "novas", "d125", "d15")) + " |")
    out.append(f"| D v2b × D coef_ent 0,01 | " + " | ".join(
        _v([m[k] for m in M["D v2b"]], [m[k] for m in M["D coef_ent 0,01"]]) for k in ("final", "novas", "d125", "d15")) + " |")
    print("\n".join(out))
    print()
    print(tabelas([(n, m, SEMS) for n, m in NOVOS], "gulosa"))
    print()
    print(tabelas([(n, m, SEMS) for n, m in NOVOS], "sorteio"))


if __name__ == "__main__":
    main()
