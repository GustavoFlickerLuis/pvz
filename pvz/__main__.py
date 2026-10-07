"""Linha de comando para testar o simulador.

    python -m pvz assistir --agente guloso --fase calmo,misto,blindados
    python -m pvz jogar --fase calmo,misto,blindados          (você joga)
    python -m pvz avaliar --sementes 3                        (baselines nas 27 fases)
    python -m pvz avaliar --conjunto teste --sementes 5       (treino, teste ou vitrine)
    python -m pvz bench                                       (passos por segundo)
"""
import argparse
import statistics
import time

from . import config as C
from .agentes import AGENTES
from .ambiente import ESPERAR, Ambiente, codificar_acao
from .divisao import FASES_TESTE, FASES_TREINO, SEMENTES_TESTE, SEMENTES_TREINO, VITRINE
from .hordas import todas_as_fases
from .render import LEGENDA, desenhar

FASE_PADRAO = "calmo,misto,blindados"


def jogar_partida(agente, fase, semente, dificuldade=1.0, a_cada=None, atraso=0.0):
    """Joga uma partida inteira. Se `a_cada` for dado, desenha o gramado a cada tantos passos."""
    amb = Ambiente()
    amb.reset(fase, semente, dificuldade)
    retorno, passos = 0.0, 0
    while True:
        acao = agente.agir(amb)
        _, r, fim = amb.step(acao)
        retorno += r
        passos += 1
        if a_cada and (fim or passos % a_cada == 0):
            print(desenhar(amb), "\n")
            if atraso:
                time.sleep(atraso)
        if fim:
            return amb, retorno


# ---------------------------------------------------------------- comandos
def cmd_assistir(args):
    agente = AGENTES[args.agente]()
    print(LEGENDA, "\n")
    amb, retorno = jogar_partida(agente, args.fase, args.semente, args.dificuldade,
                                 a_cada=args.a_cada, atraso=args.atraso)
    print(f"retorno {retorno:+.3f}   info {amb.info()}")


LETRAS = {t.simbolo.lower(): i for i, t in enumerate(C.PLANTAS)}

AJUDA_MANUAL = """comandos:
  e 2 3     planta ervilha na faixa 2, coluna 3   (letras: g e r b m)
  <enter>   espera 1 passo ({k} ticks)
  w 20      espera 20 passos
  q         sai""".format(k=C.K_TICKS)


def cmd_jogar(args):
    amb = Ambiente()
    amb.reset(args.fase, args.semente, args.dificuldade)
    retorno = 0.0
    print(LEGENDA, "\n", AJUDA_MANUAL, sep="")
    while not amb.fim:
        print("\n" + desenhar(amb))
        cmd = input("> ").strip().lower().split()
        if cmd and cmd[0] == "q":
            return
        if cmd and cmd[0] in ("?", "h"):
            print(AJUDA_MANUAL)
            continue
        acoes = [ESPERAR]
        if cmd and cmd[0] == "w":
            acoes = [ESPERAR] * (int(cmd[1]) if len(cmd) > 1 else 1)
        elif cmd:
            try:
                acao = codificar_acao(LETRAS[cmd[0]], int(cmd[1]), int(cmd[2]))
            except (KeyError, IndexError, ValueError):
                print("não entendi.", AJUDA_MANUAL)
                continue
            if not amb.acao_valida(acao):
                print("ação inválida (sol, recarga ou célula ocupada)")
                continue
            acoes = [acao]
        for a in acoes:
            _, r, fim = amb.step(a)
            retorno += r
            if fim:
                break
    print("\n" + desenhar(amb))
    print(f"retorno {retorno:+.3f}   info {amb.info()}")


def partidas_do_conjunto(conjunto, n_sementes):
    """Lista de (classes, semente) de um conjunto da divisão (pvz/divisao.py)."""
    if conjunto == "vitrine":
        return list(VITRINE)
    fases, sementes = {
        "treino": (FASES_TREINO, SEMENTES_TREINO),
        "teste": (FASES_TESTE, SEMENTES_TESTE),
        "todas": (todas_as_fases(), range(n_sementes)),
    }[conjunto]
    return [(f, s) for f in fases for s in sementes[:n_sementes]]


def cmd_avaliar(args):
    if args.fase != "todas":
        partidas = [(args.fase, s) for s in range(args.sementes)]
        print(f"fase {args.fase} x {args.sementes} semente(s), dificuldade {args.dificuldade}\n")
    else:
        partidas = partidas_do_conjunto(args.conjunto, args.sementes)
        n_fases = len({tuple(f) for f, _ in partidas})
        print(f"conjunto {args.conjunto}: {n_fases} fase(s), {len(partidas)} partida(s), "
              f"dificuldade {args.dificuldade}\n")
    for nome in args.agentes:
        vitorias, retornos, danos, ticks, sois = [], [], [], [], []
        por_fase = {}
        t0 = time.perf_counter()
        for fase, s in partidas:
            amb, ret = jogar_partida(AGENTES[nome](), fase, s, args.dificuldade)
            vitorias.append(amb.venceu)
            retornos.append(ret)
            danos.append(amb.info()["dano_frac"])
            ticks.append(amb.tick)
            sois.append(amb.sol)
            por_fase.setdefault("+".join(fase) if not isinstance(fase, str) else fase, []).append(amb.venceu)
        print(f"{nome:10s} vitórias {sum(vitorias) / len(vitorias):6.1%}   "
              f"retorno {statistics.mean(retornos):+.3f} ± {statistics.pstdev(retornos):.3f}   "
              f"dano {statistics.mean(danos):5.1%}   sol final {statistics.mean(sois):5.0f}   "
              f"tick final médio {statistics.mean(ticks):6.0f}   "
              f"({time.perf_counter() - t0:.1f}s)")
        if args.por_fase:
            for f, v in por_fase.items():
                print(f"    {f:45s} {sum(v)}/{len(v)}")


def cmd_bench(args):
    from .agentes import Aleatorio
    amb, agente = Ambiente(), Aleatorio(0)
    passos, ticks, partidas = 0, 0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.segundos:
        amb.reset(FASE_PADRAO, partidas)
        fim = False
        while not fim:
            _, _, fim = amb.step(agente.agir(amb))
            passos += 1
        ticks += amb.tick
        partidas += 1
    dt = time.perf_counter() - t0
    print(f"{partidas} partidas em {dt:.1f}s: {passos / dt:,.0f} passos/s ({ticks / dt:,.0f} ticks/s), "
          f"incluindo o agente aleatório e o estado/máscara a cada passo")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pvz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def fase_args(p):
        p.add_argument("--fase", default=FASE_PADRAO, help="classes início,meio,fim (1 a 3)")
        p.add_argument("--semente", type=int, default=0)
        p.add_argument("--dificuldade", type=float, default=1.0)

    p = sub.add_parser("assistir", help="assiste um agente jogar")
    fase_args(p)
    p.add_argument("--agente", choices=AGENTES, default="guloso")
    p.add_argument("--a-cada", type=int, default=10, help="desenha o gramado a cada N passos")
    p.add_argument("--atraso", type=float, default=0.0, help="pausa em segundos entre quadros")
    p.set_defaults(func=cmd_assistir)

    p = sub.add_parser("jogar", help="você joga pelo terminal")
    fase_args(p)
    p.set_defaults(func=cmd_jogar)

    p = sub.add_parser("avaliar", help="taxa de vitória dos baselines")
    p.add_argument("--agentes", nargs="+", choices=AGENTES, default=list(AGENTES))
    p.add_argument("--fase", default="todas", help="uma fase início,meio,fim (ignora --conjunto)")
    p.add_argument("--conjunto", choices=("treino", "teste", "vitrine", "todas"), default="todas",
                   help="treino/teste usam as sementes reservadas de cada um; vitrine tem sementes fixas")
    p.add_argument("--sementes", type=int, default=3, help="sementes por fase (ignorado na vitrine)")
    p.add_argument("--dificuldade", type=float, default=1.0)
    p.add_argument("--por-fase", action="store_true", help="mostra vitórias em cada fase")
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("bench", help="mede passos por segundo")
    p.add_argument("--segundos", type=float, default=5.0)
    p.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
