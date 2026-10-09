"""Fila de execuções (arquitetura, currículo, semente), cada uma num processo separado.

    python -m rl.rodar                                # o LOTE2 inteiro; saída em rl/rodar.log
    python -m rl.rodar --fila cnn2:sim:0 cnn3:nao:1   # só estas, nesta ordem
    python -m rl.rodar --passos 400000                # orçamento por execução (padrão: ORCAMENTO)

Todas com a receita baseline (adam 1e-4 constante, clip 10). Ao ser chamado de novo, pula
as execuções terminadas (resultado.json) e retoma a interrompida do checkpoint. Se uma
execução sair por rl/PARAR (código 3), a fila para também. Um rl/PARAR que sobrou de uma
parada anterior é apagado ao iniciar (chamar o rodar.py = querer continuar).

Checkpoints: cada execução termina gravando o seu (rede, otimizador, replay...). Quando uma
execução termina, os checkpoints das execuções anteriores da MESMA receita (arquitetura +
currículo, outra semente) são apagados: fica só o da última de cada receita.
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime

from desafio3.modelo import ARQUITETURAS

from . import status
from .dqn import ARQ_PARAR, CODIGO_PARADO, PASTA_LOGS, id_padrao

ORCAMENTO = 400_000   # passos de jogo por execução (cálculo em rl/RELATORIO.md, seção 1)

# Lote 2: 4 arquiteturas × com e sem currículo × 3 sementes, na ordem semente → arquitetura.
# A cnn1 sem currículo é a baseline do lote 1 (mesma rede, receita, sorteios e sementes:
# baseline-s0..s2 em rl/logs/), por isso fica fora; dá para pedir com --fila cnn1:nao:0.
LOTE2 = [(a, c, s) for s in range(3) for a in ("cnn1d", "cnn2", "cnn3", "cnn1")
         for c in ("nao", "sim") if (a, c) != ("cnn1", "nao")]


def log(msg):
    print(f"[{datetime.now():%d/%m %H:%M:%S}] {msg}", flush=True)


def ler_fila(itens):
    fila = []
    for it in itens:
        a, c, s = it.split(":")
        if a not in ARQUITETURAS or c not in ("nao", "sim"):
            raise SystemExit(f"item inválido: {it} (arquitetura:nao|sim:semente)")
        fila.append((a, c, int(s)))
    return fila


def apagar_ckpts_anteriores(fila, arq, cur, semente):
    """Apaga o checkpoint das execuções terminadas da mesma receita, menos a desta semente."""
    for a, c, s in fila:
        if (a, c) == (arq, cur) and s != semente:
            pasta = PASTA_LOGS / id_padrao(a, c, s)
            if (pasta / "resultado.json").exists() and (pasta / "ckpt").exists():
                shutil.rmtree(pasta / "ckpt")
                log(f"{pasta.name}: checkpoint apagado (fica o de {id_padrao(arq, cur, semente)})")


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fila", nargs="+", metavar="ARQ:CUR:SEMENTE")
    p.add_argument("--passos", type=int, default=ORCAMENTO)
    a = p.parse_args()
    fila = ler_fila(a.fila) if a.fila else LOTE2
    ids = [id_padrao(*f) for f in fila]
    status.ARQ_FILA.write_text(json.dumps({"ids": ids}, indent=1), encoding="utf-8")
    if ARQ_PARAR.exists():
        ARQ_PARAR.unlink()
        log("rl/PARAR antigo apagado")
    for (arq, cur, semente), id_ in zip(fila, ids):
        if (PASTA_LOGS / id_ / "resultado.json").exists():
            log(f"{id_}: já terminou, pulando")
            continue
        retoma = (PASTA_LOGS / id_ / "ckpt" / "estado.pt").exists()
        log(f"{id_}: {'retomando do checkpoint' if retoma else 'começando'}")
        cmd = [sys.executable, "-u", "-m", "rl.dqn", "--arquitetura", arq, "--curriculo", cur,
               "--semente", str(semente), "--passos", str(a.passos), "--id", id_]
        codigo = subprocess.run(cmd).returncode
        status.escrever()
        if codigo == CODIGO_PARADO:
            log(f"{id_}: parado por rl/PARAR; fila interrompida (chame de novo para retomar)")
            return
        if codigo != 0:
            log(f"{id_}: saiu com código {codigo}; seguindo para a próxima")
            continue
        log(f"{id_}: terminado")
        apagar_ckpts_anteriores(fila, arq, cur, semente)
    log("fila completa")
    status.escrever()


if __name__ == "__main__":
    main()
