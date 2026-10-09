"""rl/STATUS.md: painel da fila (atualizado a cada avaliação e entre execuções).

A fila corrente é a de rl/fila.json, gravada pelo rl/rodar.py; sem ele, a do lote 1.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

RAIZ = Path(__file__).parent
PASTA_LOGS = RAIZ / "logs"
ARQ_STATUS = RAIZ / "STATUS.md"
RECEITAS = ("baseline", "vencedora", "muon")
# lote 1: semente 0 das 3 receitas → semente 1 das 3 → semente 2 das 3
FILA = [(r, s) for s in range(3) for r in RECEITAS]
ARQ_FILA = RAIZ / "fila.json"      # {"ids": [...]}, gravado pelo rl/rodar.py
NOTAS = RAIZ / "NOTAS_STATUS.md"   # anotações livres (bloqueios, decisões), coladas no fim


def ids_da_fila():
    if ARQ_FILA.exists():
        return json.loads(ARQ_FILA.read_text(encoding="utf-8"))["ids"]
    return [f"{r}-s{s}" for r, s in FILA]


def mil(n):
    return f"{n:,}".replace(",", ".")


def ler(id_):
    pasta = PASTA_LOGS / id_
    prog = res = None
    if (pasta / "progresso.json").exists():
        prog = json.loads((pasta / "progresso.json").read_text(encoding="utf-8"))
    if (pasta / "resultado.json").exists():
        res = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
    return prog, res


def escrever():
    linhas = [f"# STATUS do treino (rl/)", "",
              f"Atualizado em {datetime.now():%d/%m/%Y %H:%M:%S}.", ""]
    atual = None
    tabela = ["| execução | status | passos | vitórias nas avaliações periódicas (63 partidas) "
              "| vitórias final (420) |", "|---|---|---|---|---|"]
    ids = ids_da_fila()
    for id_ in ids:
        prog, res = ler(id_)
        if prog is None:
            tabela.append(f"| {id_} | na fila | — | — | — |")
            continue
        curva = ", ".join(f"{p // 1000}k: {100 * v:.0f}%" for p, v in prog["avaliacoes"])
        final = f"{100 * res['vitorias_final']:.1f}%" if res else "—"
        st = res["status"] if res else prog["status"]
        if prog.get("nivel"):
            st += f" (currículo: nível {prog['nivel']})"
        tabela.append(f"| {id_} | {st} | {mil(prog['passo'])}/{mil(prog['orcamento'])} | {curva} | {final} |")
        if not res and prog["status"] == "rodando":
            atual = (id_, prog)
    if atual:
        id_, prog = atual
        pps = prog["passos_por_s"] or 1
        resta = (prog["orcamento"] - prog["passo"]) / pps
        na_fila = sum(1 for i in ids if ler(i)[0] is None)
        total = resta + na_fila * prog["orcamento"] / pps
        linhas += [f"**Execução atual:** {id_}, passo {mil(prog['passo'])} de {mil(prog['orcamento'])} "
                   f"({100 * prog['passo'] / prog['orcamento']:.0f}%), {pps:.0f} passos/s "
                   f"(última atualização {prog['atualizado']}).",
                   f"**ETA desta execução:** ~{resta / 60:.0f} min "
                   f"(≈ {datetime.now() + timedelta(seconds=resta):%H:%M}). "
                   f"**ETA da fila inteira:** ~{total / 3600:.1f} h "
                   f"(≈ {datetime.now() + timedelta(seconds=total):%d/%m %H:%M}), sem contar as avaliações.",
                   ""]
    else:
        linhas += ["Nenhuma execução rodando agora.", ""]
    linhas += ["## Vitórias até agora", ""] + tabela
    if NOTAS.exists():
        linhas += ["", NOTAS.read_text(encoding="utf-8").strip()]
    ARQ_STATUS.write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    escrever()
    print(ARQ_STATUS.read_text(encoding="utf-8"))
