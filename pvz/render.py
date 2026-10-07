"""Visualização em texto do gramado, para assistir às partidas."""
from collections import Counter

from . import config as C

LARGURA = 5   # caracteres por célula


def _celula(planta, zumbis):
    if planta is None:
        p = "."
    elif planta.tipo == C.MINA and not planta.armada:
        p = "m"   # mina ainda armando
    else:
        p = C.PLANTAS[planta.tipo].simbolo
    if not zumbis:
        z = ""
    else:
        cont = Counter(C.ZUMBIS[t].simbolo for t in zumbis)
        z = "".join(s * n for s, n in sorted(cont.items()))
        if len(z) > LARGURA - 1:
            z = "".join(f"{s}{n}" for s, n in sorted(cont.items()))[:LARGURA - 1]
    return (p + z).ljust(LARGURA)


def desenhar(amb):
    rec = "  ".join(
        f"{t.simbolo}:{'ok' if r == 0 else r}" for t, r in zip(C.PLANTAS, amb.recargas)
    )
    info = amb.info()
    linhas = [
        f"tick {amb.tick:5d}   sol {amb.sol:4d}   dano {info['dano_frac']:5.1%}   "
        f"zumbis: {info['zumbis_vivos']} no gramado, {info['zumbis_por_vir']} por vir",
        f"recargas  {rec}",
        "       " + "".join(str(c).ljust(LARGURA) for c in range(C.COLUNAS)),
    ]
    for f in range(C.FAIXAS):
        por_coluna = [[] for _ in range(C.COLUNAS)]
        for z in amb.zumbis[f]:
            por_coluna[z.coluna].append(z.tipo)
        celulas = "".join(_celula(amb.grade[f][c], por_coluna[c]) for c in range(C.COLUNAS))
        linhas.append(f"  {f}  | {celulas}<")
    if amb.fim:
        linhas.append("  >>> VITÓRIA <<<" if amb.venceu else "  >>> DERROTA <<<")
    return "\n".join(linhas)


LEGENDA = (
    "plantas: G girassol  E ervilha  R repetidora  B batata  M mina armada (m = armando)\n"
    "zumbis:  z comum  b blindado  r rápido   (| = casa, < = entrada)"
)
