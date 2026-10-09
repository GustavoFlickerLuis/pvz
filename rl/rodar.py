"""Fila das 9 execuções (3 receitas × 3 sementes), cada uma num processo separado.

    python -m rl.rodar            # saída em rl/rodar.log

Ordem: semente 0 das 3 receitas → semente 1 das 3 → semente 2 das 3. Ao ser chamado de
novo, pula as execuções terminadas (resultado.json) e retoma a interrompida do checkpoint.
Se uma execução sair por rl/PARAR (código 3), a fila para também. Um rl/PARAR que sobrou
de uma parada anterior é apagado ao iniciar (chamar o rodar.py = querer continuar).
"""
import subprocess
import sys
from datetime import datetime

from . import status
from .dqn import ARQ_PARAR, CODIGO_PARADO, PASTA_LOGS

ORCAMENTO = 400_000   # passos de jogo por execução (cálculo em rl/RELATORIO.md, seção 1)


def log(msg):
    print(f"[{datetime.now():%d/%m %H:%M:%S}] {msg}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if ARQ_PARAR.exists():
        ARQ_PARAR.unlink()
        log("rl/PARAR antigo apagado")
    for receita, semente in status.FILA:
        id_ = f"{receita}-s{semente}"
        if (PASTA_LOGS / id_ / "resultado.json").exists():
            log(f"{id_}: já terminou, pulando")
            continue
        retoma = (PASTA_LOGS / id_ / "ckpt" / "estado.pt").exists()
        log(f"{id_}: {'retomando do checkpoint' if retoma else 'começando'}")
        cmd = [sys.executable, "-u", "-m", "rl.dqn", "--receita", receita, "--semente", str(semente),
               "--passos", str(ORCAMENTO), "--id", id_]
        codigo = subprocess.run(cmd).returncode
        status.escrever()
        if codigo == CODIGO_PARADO:
            log(f"{id_}: parado por rl/PARAR; fila interrompida (chame de novo para retomar)")
            return
        if codigo != 0:
            log(f"{id_}: saiu com código {codigo}; seguindo para a próxima")
            continue
        log(f"{id_}: terminado")
    log("fila completa")
    status.escrever()


if __name__ == "__main__":
    main()
