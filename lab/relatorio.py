"""Tabelas e figuras do lab a partir dos logs; reescreve a seção gerada do LAB.md.

    python -m lab.relatorio

Gera, entre os marcadores <!-- GERADO:INICIO --> e <!-- GERADO:FIM --> no fim do LAB.md:
a fila, a sanidade, o teto, a Fase 1 e a Fase 2 (tabela braço/receita × semente com
vitórias finais, pico, passo em que cruza o Econ5, entropia mínima antes de 500k,
parâmetros, passos/s e tempo; média, desvio e veredito pela regra). Figuras em
lab/figuras/curvas_f1.png e curvas_f2.png (um painel por receita/braço, mesma escala).
O texto fora dos marcadores não é tocado.
"""
import csv
import json
import math
import statistics as st

from . import LOGS, PASTA_LAB, RAIZ
from .fila import MEMORIA, falhas

ECON5 = 0.652
INICIO, FIM = "<!-- GERADO:INICIO -->", "<!-- GERADO:FIM -->"
BRACOS = ("cnn1d", "atencao_antes", "atencao_depois", "transformer", "transformer_d128",
          "cnn1d_dilatada")
LETRA = {"cnn1d": "A", "atencao_antes": "B", "atencao_depois": "C", "transformer": "D",
         "transformer_d128": "D2", "cnn1d_dilatada": "E"}
PAINEL = (1, 3, 4, 2, 6)
RUINS, BOAS = (1, 3, 4), (2, 6)
F2_SEMENTES = (1, 2, 3)


def _csv(caminho):
    if not caminho.exists():
        return []
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def execucao(id_):
    """Resumo de uma execução do PPO (None se não começou)."""
    pasta = LOGS / id_
    if not (pasta / "config.json").exists():
        return None
    cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))["config"]
    res = (json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
           if (pasta / "resultado.json").exists() else None)
    reg = _csv(pasta / "registro.csv")
    av = [a for a in _csv(pasta / "avaliacoes.csv") if a["modo"] == "gulosa" and a["final"] == "0"]
    curva = [(int(a["passo"]), float(a["vitorias"])) for a in av]
    ent500 = [float(r["entropia"]) for r in reg if int(r["passo"]) <= 500_000]
    cruza = next((p for p, v in curva if v >= ECON5), None)
    ult = reg[-1] if reg else {}
    return {"id": id_, "cfg": cfg, "feito": res is not None,
            "final": res["vitorias_final"] if res else None,
            "minutos": res["minutos"] if res else None,
            "passos_por_s": res["passos_por_s"] if res else (float(ult["passos_por_s"]) if ult else None),
            "passo": res["passos"] if res else (int(ult["passo"]) if ult else 0),
            "curva": curva, "pico": max((v for _, v in curva), default=None),
            "pico_passo": max(curva, key=lambda x: x[1])[0] if curva else None,
            "cruza": cruza, "ent_min_500k": min(ent500) if ent500 else None,
            "entropia": float(ult["entropia"]) if ult else None,
            "kl": float(ult["kl"]) if ult else None,
            "perda_v": float(ult["perda_v"]) if ult else None,
            "reg": reg,
            "rss": ((res.get("rss_inicio_mb"), res.get("rss_fim_mb"), res.get("rss_pico_mb"))
                    if res and res.get("rss_fim_mb") is not None else None)}


def _rss(e):
    if not e or not e.get("rss"):
        return "—"
    i, f, pk = e["rss"]
    return f"{i:.0f}→{f:.0f} ({pk:.0f})"


def _pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def _k(p):
    return "—" if p is None else (f"{p / 1e6:.2f}M" if p >= 1e6 else f"{p / 1e3:.0f}k")


def _num(x, f="{:.3f}"):
    return "—" if x is None else f.format(x)


def parametros():
    m = PASTA_LAB / "medidas.json"
    return json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}


def media_dp(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else None)


def veredito(a, b):
    """a, b: listas de finais nas mesmas sementes. Devolve (Δ = a − b, 2·s_c, texto)."""
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return None, None, "incompleto"
    d = st.mean(a) - st.mean(b)
    sc = math.sqrt((st.stdev(a) ** 2 + st.stdev(b) ** 2) / 2)
    if abs(d) > 2 * sc:
        return d, 2 * sc, "venceu" if d > 0 else "perdeu"
    return d, 2 * sc, "empate"


def linha_exec(nome, s, e, med):
    if e is None:
        return f"| {nome} | {s} | não começou | | | | | | | | |"
    par = med.get(e["cfg"].get("arquitetura", "cnn1d"), {}).get("parametros")
    estado = _pct(e["final"]) if e["feito"] else f"em curso ({_k(e['passo'])})"
    return (f"| {nome} | {s} | {estado} | {_pct(e['pico'])} ({_k(e['pico_passo'])}) | "
            f"{_k(e['cruza']) if e['cruza'] else 'não cruza'} | {_num(e['ent_min_500k'])} | "
            f"{par if par else '—'} | {_num(e['passos_por_s'], '{:.0f}')} | "
            f"{_num(e['minutos'], '{:.1f}')} | {_rss(e)} |")


CAB = ("| | semente | vitórias finais (420) | pico na curva (63) | cruza o Econ5 (65,2%) | "
       "entropia mín. ≤ 500k | parâmetros | passos/s | minutos | RAM do treino, MB: início→fim (pico) |\n"
       "|---|---|---|---|---|---|---|---|---|---|")


def secao_fila():
    fila = PASTA_LAB / "fila.json"
    if not fila.exists():
        return ""
    jobs = json.loads(fila.read_text(encoding="utf-8"))
    fl = falhas(fila)
    out = ["### Fila (`lab/fila.json`)", "", "| id | estado |", "|---|---|"]
    for j in jobs:
        e = execucao(j["id"])
        if j["id"] in fl:
            est = f"**FALHOU** ({fl[j['id']]['tentativas']} tentativas seguidas sem avançar; pulada)"
        elif e is None:
            est = "na fila"
        elif e["feito"]:
            est = f"feito: {_pct(e['final'])} em {e['minutos']:.0f} min"
        else:
            est = f"interrompido/em curso no passo {e['passo']}"
        out.append(f"| {j['id']} | {est} |")
    for id_, f in fl.items():
        out += ["", f"**{id_} falhou** ({f['horario']}, passo do último ckpt: {f['passo_ckpt']}). "
                    f"Fim do log (`rl/logs/{id_}.log`):", "", "```", f["fim_do_log"].rstrip(), "```"]
    return "\n".join(out)


def secao_memoria():
    """RAM (RSS) da fila e do treino por execução (lab/memoria.csv) e checagem de vazamento."""
    linhas = _csv(MEMORIA)
    if not linhas:
        return ""
    out = ["### Memória (RSS em MB; `lab/memoria.csv`, últimas 40 linhas)", "",
           "| horário | execução | momento | PID da fila | RSS da fila | treino: início | fim | pico |",
           "|---|---|---|---|---|---|---|---|"]
    for l in linhas[-40:]:
        out.append(f"| {l['horario']} | {l['id']} | {l['momento']} | {l['pid_fila']} | "
                   f"{l['rss_fila_mb']} | {l['rss_treino_inicio_mb'] or '—'} | "
                   f"{l['rss_treino_fim_mb'] or '—'} | {l['rss_treino_pico_mb'] or '—'} |")
    # vazamento: a mesma fila (PID) crescendo de uma execução para a outra, ou o treino de um
    # braço terminando cada vez maior
    notas, por_pid, fins = [], {}, {}
    for l in linhas:
        if l["momento"] == "inicio":
            por_pid.setdefault(l["pid_fila"], []).append(float(l["rss_fila_mb"]))
        if l["rss_treino_fim_mb"]:
            arq = ((execucao(l["id"]) or {}).get("cfg") or {}).get("arquitetura", "?")
            fins.setdefault(arq, []).append(float(l["rss_treino_fim_mb"]))
    for pid, v in por_pid.items():
        if len(v) >= 3 and v[-1] - v[0] > 50:
            notas.append(f"a fila (PID {pid}) cresceu de {v[0]:.0f} para {v[-1]:.0f} MB em {len(v)} execuções")
    for arq, v in fins.items():
        if len(v) >= 3 and all(b > a + 10 for a, b in zip(v, v[1:])):
            notas.append(f"o treino de {arq} termina cada vez maior: {', '.join(f'{x:.0f}' for x in v)} MB")
    out += ["", ("**Possível vazamento:** " + "; ".join(notas) + ".") if notas else
            "Sem sinal de vazamento até aqui (a fila não cresce entre execuções e o treino de "
            "cada braço não termina cada vez maior)."]
    return "\n".join(out)


def secao_sanidade(med):
    out = ["### Sanidade (calmo, dificuldade 0,5, ~300k, 200 partidas; aprovado ≥ 80%)", "",
           "| braço | vitórias (200) | aprovado | entropia final | kl final | perda_v final | "
           "passos/s | minutos |", "|---|---|---|---|---|---|---|---|"]
    for b in BRACOS:
        e = execucao(f"lab-san-{b}")
        if e is None:
            out.append(f"| {LETRA[b]} {b} | não rodou | | | | | | |")
            continue
        apr = "—" if not e["feito"] else ("sim" if e["final"] >= 0.8 else "**não**")
        out.append(f"| {LETRA[b]} {b} | {_pct(e['final']) if e['feito'] else 'em curso'} | {apr} | "
                   f"{_num(e['entropia'])} | {_num(e['kl'], '{:.4f}')} | {_num(e['perda_v'], '{:.4f}')} | "
                   f"{_num(e['passos_por_s'], '{:.0f}')} | {_num(e['minutos'], '{:.1f}')} |")
    return "\n".join(out)


def aprovados():
    return [b for b in BRACOS if (e := execucao(f"lab-san-{b}")) and e["feito"] and e["final"] >= 0.8]


def secao_teto():
    out = ["### Teto (só registro, sem veredito; prova de avaliação, 1 semente)", "",
           "| braço | nota | grupo ii | longe (pares) | perto (pares) | sem zumbi (pares) | parada | minutos |",
           "|---|---|---|---|---|---|---|---|"]
    for b in BRACOS:
        r = RAIZ / "rl" / "diag" / "logs" / f"lab-teto-{b}-s0" / "resultado.json"
        if not r.exists():
            out.append(f"| {LETRA[b]} {b} | não rodou | | | | | | |")
            continue
        r = json.loads(r.read_text(encoding="utf-8"))
        n, g = r["nota"], r["grupos"]
        out.append(f"| {LETRA[b]} {b} | {n['nota']:.1f}% | {n['nota_ii']:.1f}% | "
                   f"{g['nota_longe']:.1f}% ({g['pares_longe']}) | {g['nota_perto_total']:.1f}% "
                   f"({g['pares_perto_total']}) | {g['nota_sem_zumbi']:.1f}% ({g['pares_sem_zumbi']}) | "
                   f"{r['parada']} | {r['minutos']} |")
    return "\n".join(out)


def finais(prefixo, sementes):
    return {s: e["final"] for s in sementes if (e := execucao(f"{prefixo}-s{s}")) and e["feito"]}


def secao_f1(med):
    out = ["### Fase 1 (cnn1d, 2M passos)", "", CAB]
    for rec in ("atual", "v2a", "v2b"):
        for s in PAINEL:
            out.append(linha_exec(rec, s, execucao(f"lab-f1-{rec}-s{s}"), med))
    out += ["", "| receita | " + " | ".join(f"s{s}" for s in PAINEL) + " | média | dp | "
            "ruins > 50% | boas: maior queda vs atual | aceita? |",
            "|---" * (len(PAINEL) + 6) + "|"]
    ref = finais("lab-f1-atual", PAINEL)
    for rec in ("atual", "v2a", "v2b"):
        f = finais(f"lab-f1-{rec}", PAINEL)
        m, dp = media_dp(list(f.values()))
        cel = " | ".join(_pct(f.get(s)) for s in PAINEL)
        if rec == "atual":
            out.append(f"| {rec} | {cel} | {_pct(m)} | {_pct(dp)} | — | — | (referência) |")
            continue
        ruins = sum(f[s] > 0.5 for s in RUINS if s in f)
        quedas = [ref[s] - f[s] for s in BOAS if s in f and s in ref]
        completo = all(s in f for s in PAINEL) and all(s in ref for s in PAINEL)
        aceita = ("—" if not completo else
                  ("sim" if ruins >= 2 and max(quedas) <= 0.10 else "não"))
        out.append(f"| {rec} | {cel} | {_pct(m)} | {_pct(dp)} | {ruins}/{sum(s in f for s in RUINS)} | "
                   f"{_pct(max(quedas)) if quedas else '—'} | {aceita} |")
    return "\n".join(out)


def secao_f2(med, receita_f2):
    out = [f"### Fase 2 (receita da Fase 1: `{receita_f2 or '?'}`, 2M passos, sementes 1, 2, 3)", "",
           "A cnn1d reaproveita as execuções da Fase 1 com essa receita.", "", CAB]
    ids = {b: (f"lab-f1-{receita_f2}" if b == "cnn1d" else f"lab-f2-{b}") for b in BRACOS}
    for b in BRACOS:
        for s in F2_SEMENTES:
            out.append(linha_exec(f"{LETRA[b]} {b}", s, execucao(f"{ids[b]}-s{s}"), med))
    out += ["", "| braço | média | dp | Δ vs cnn1d | 2·s_c | veredito | passos/s médio | minutos médio |",
            "|---|---|---|---|---|---|---|---|"]
    fin = {b: finais(ids[b], F2_SEMENTES) for b in BRACOS}
    for b in BRACOS:
        m, dp = media_dp(list(fin[b].values()))
        es = [e for s in F2_SEMENTES if (e := execucao(f"{ids[b]}-s{s}")) and e["feito"]]
        vel = st.mean(e["passos_por_s"] for e in es) if es else None
        mins = st.mean(e["minutos"] for e in es) if es else None
        if b == "cnn1d":
            v = (None, None, "(referência)")
        else:
            comuns = [s for s in F2_SEMENTES if s in fin[b] and s in fin["cnn1d"]]
            v = veredito([fin[b][s] for s in comuns], [fin["cnn1d"][s] for s in comuns])
        out.append(f"| {LETRA[b]} {b} | {_pct(m)} | {_pct(dp)} | "
                   f"{'—' if v[0] is None else f'{100 * v[0]:+.1f}'} | "
                   f"{'—' if v[1] is None else f'{100 * v[1]:.1f}'} | {v[2]} | "
                   f"{_num(vel, '{:.0f}')} | {_num(mins, '{:.1f}')} |")
    comuns = [s for s in F2_SEMENTES if s in fin["transformer_d128"] and s in fin["transformer"]]
    v = veredito([fin["transformer_d128"][s] for s in comuns], [fin["transformer"][s] for s in comuns])
    out.append("")
    out.append(f"D2 × D: Δ = {'—' if v[0] is None else f'{100 * v[0]:+.1f}'} pontos, 2·s_c = "
               f"{'—' if v[1] is None else f'{100 * v[1]:.1f}'} → {v[2]}.")
    return "\n".join(out), ids


def figura(nome, paineis):
    """paineis: lista de (título, [(semente, curva)]). Mesma escala em todos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paineis = [p for p in paineis if any(c for _, c in p[1])]
    if not paineis:
        return None
    n = len(paineis)
    col = min(3, n)
    lin = math.ceil(n / col)
    fig, eixos = plt.subplots(lin, col, figsize=(4.2 * col, 3.2 * lin), squeeze=False,
                              sharex=True, sharey=True)
    cores = {1: "#c0392b", 2: "#2471a3", 3: "#d68910", 4: "#7d3c98", 6: "#229954"}
    for ax, (tit, curvas) in zip(eixos.flat, paineis):
        for s, c in curvas:
            if c:
                ax.plot([p / 1e6 for p, _ in c], [100 * v for _, v in c], marker=".",
                        color=cores.get(s), label=f"s{s}")
        ax.axhline(100 * ECON5, color="gray", ls="--", lw=0.8)
        ax.set_title(tit, fontsize=10)
        ax.set_ylim(-3, 103)
        ax.set_xlim(0, 2.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    for ax in list(eixos.flat)[n:]:
        ax.axis("off")
    for ax in eixos[-1]:
        ax.set_xlabel("passos (M)")
    for ax in eixos[:, 0]:
        ax.set_ylabel("vitórias, gulosa (63) %")
    fig.suptitle("Vitórias na avaliação periódica (tracejado: Econ5, 65,2%)", fontsize=10)
    fig.tight_layout()
    (PASTA_LAB / "figuras").mkdir(exist_ok=True)
    caminho = PASTA_LAB / "figuras" / nome
    fig.savefig(caminho, dpi=110)
    plt.close(fig)
    return caminho


def receita_escolhida():
    arq = PASTA_LAB / "receita_f2.txt"
    return arq.read_text(encoding="utf-8").strip() if arq.exists() else None


def main():
    med = parametros()
    rec = receita_escolhida()
    partes = [secao_fila(), secao_memoria(), secao_sanidade(med), secao_teto(), secao_f1(med)]
    f1 = figura("curvas_f1.png", [(rec_, [(s, (e or {}).get("curva", []))
                                          for s in PAINEL for e in [execucao(f"lab-f1-{rec_}-s{s}")]])
                                  for rec_ in ("atual", "v2a", "v2b")])
    if f1:
        partes.append("![curvas da Fase 1](lab/figuras/curvas_f1.png)")
    if rec:
        txt, ids = secao_f2(med, rec)
        partes.append(txt)
        f2 = figura("curvas_f2.png", [(f"{LETRA[b]} {b}", [(s, (e or {}).get("curva", []))
                                                          for s in F2_SEMENTES
                                                          for e in [execucao(f"{ids[b]}-s{s}")]])
                                     for b in BRACOS])
        if f2:
            partes.append("![curvas da Fase 2](lab/figuras/curvas_f2.png)")
    corpo = "\n\n".join(p for p in partes if p)
    lab = RAIZ / "LAB.md"
    texto = lab.read_text(encoding="utf-8")
    bloco = (f"{INICIO}\n## RESULTADOS (gerado por `python -m lab.relatorio`; não editar à mão)\n\n"
             f"{corpo}\n{FIM}")
    if INICIO in texto:
        texto = texto[:texto.index(INICIO)] + bloco + texto[texto.index(FIM) + len(FIM):]
    else:
        texto = texto.rstrip() + "\n\n" + bloco + "\n"
    lab.write_text(texto, encoding="utf-8")
    print("LAB.md atualizado")


if __name__ == "__main__":
    main()
