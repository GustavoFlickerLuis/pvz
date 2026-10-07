"""Análise de um resultado de tools/calibrar.py.

    python tools/analisar.py resultado.json

Mostra, para o guloso e o Econ5:
  - vitórias e tick médio da derrota por classe de cada segmento;
  - de qual segmento veio o zumbi que cruzou a coluna 0 (o tick da derrota engana:
    um zumbi do início leva ~450 ticks para atravessar e "perde no meio" pelo relógio).
E, para o Econ5, as vitórias em cada uma das 27 fases. Se quase todas forem 0/N ou
N/N, a semente não está variando a história da partida.
"""
import json
import statistics as st
import sys
from collections import Counter, defaultdict

SEG = ["início", "meio", "fim"]

with open(sys.argv[1]) as f:
    d = json.load(f)["res"]

for ag in ("guloso", "econ5"):
    rs = [r for r in d if r["agente"] == ag]
    print(f"\n== {ag}")
    for i in range(3):
        g = defaultdict(list)
        for r in rs:
            g[r["fase"][i]].append(r)
        for c, l in g.items():
            der = [r["tick"] for r in l if not r["venceu"]]
            print(f"  {SEG[i]:6s} {c:18s} vit {sum(r['venceu'] for r in l) / len(l):6.1%}  "
                  f"derrotas {len(der):2d}  tick médio derrota {st.mean(der) if der else float('nan'):5.0f}")
    print("  derrotas por segmento do zumbi que cruzou:",
          dict(Counter(SEG[r["culpado"]] if r["culpado"] is not None else "max_ticks"
                       for r in rs if not r["venceu"])))

print("\n== econ5 por fase")
g = defaultdict(list)
for r in d:
    if r["agente"] == "econ5":
        g["+".join(r["fase"])].append(r["venceu"])
for f, v in g.items():
    print(f"  {f:45s} {sum(v)}/{len(v)}")
print("  histograma (vitórias: nº de fases):", dict(sorted(Counter(sum(v) for v in g.values()).items())))
