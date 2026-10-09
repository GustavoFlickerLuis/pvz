"""Teste 3: a retomada do checkpoint é exata.

    python -m rl.teste_retomada

A: 3.000 passos direto. B: a mesma config, mas cria rl/PARAR assim que B grava o
progresso da avaliação do passo 1.000 (o treino para no fim do episódio seguinte) e
depois retoma. registro.csv, episodios.csv e avaliacoes.csv de A e B têm de ser idênticos.
Para exercitar tudo em 3.000 passos: treino a partir do passo 500, rede alvo a cada 100
atualizações e avaliação a cada 1.000 passos.
"""
import filecmp
import json
import shutil
import subprocess
import sys
import time

from .dqn import ARQ_PARAR, CODIGO_PARADO, PASTA_LOGS

ARGS = ["--receita", "{receita}", "--semente", "0", "--passos", "3000", "--inicio", "500",
        "--alvo-cada", "100", "--aval-cada", "1000"]


def comando(id_, receita):
    return [sys.executable, "-m", "rl.dqn", "--id", id_] + [a.format(receita=receita) for a in ARGS]


def main(receita="baseline"):
    sys.stdout.reconfigure(encoding="utf-8")
    a, b = f"teste-retomada-{receita}-A", f"teste-retomada-{receita}-B"
    for id_ in (a, b):
        shutil.rmtree(PASTA_LOGS / id_, ignore_errors=True)
    ARQ_PARAR.unlink(missing_ok=True)
    assert subprocess.run(comando(a, receita)).returncode == 0

    proc = subprocess.Popen(comando(b, receita))
    prog = PASTA_LOGS / b / "progresso.json"
    while proc.poll() is None:
        if prog.exists() and json.loads(prog.read_text(encoding="utf-8"))["passo"] >= 1000:
            ARQ_PARAR.touch()
            break
        time.sleep(0.05)
    codigo = proc.wait()
    ARQ_PARAR.unlink(missing_ok=True)
    passo_parada = json.loads(prog.read_text(encoding="utf-8"))["passo"]
    print(f"B parou com código {codigo} no passo {passo_parada}")
    assert codigo == CODIGO_PARADO and passo_parada < 3000, "B não parou no meio"
    assert subprocess.run(comando(b, receita)).returncode == 0

    ok = True
    for arq in ("registro.csv", "episodios.csv", "avaliacoes.csv", "avaliacao_final.csv"):
        igual = filecmp.cmp(PASTA_LOGS / a / arq, PASTA_LOGS / b / arq, shallow=False)
        print(f"{arq}: {'idêntico' if igual else 'DIFERENTE'}")
        ok &= igual
    linhas = sum(1 for _ in open(PASTA_LOGS / a / "registro.csv", encoding="utf-8")) - 1
    print(f"registro.csv tem {linhas} linhas; parada no passo {passo_parada} de 3000")
    print("RETOMADA EXATA" if ok else "RETOMADA DIFERENTE")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:]) else 1)
