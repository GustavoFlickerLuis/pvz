"""Teste 3: a retomada do checkpoint é exata.

    python -m rl.teste_retomada [receita]                     # lote 1: baseline, vencedora, muon
    python -m rl.teste_retomada --arquitetura cnn2 [--curriculo sim] [--exploracao intencional]

A: 3.000 passos direto. B: a mesma config, mas cria rl/PARAR assim que B grava o
progresso da avaliação do passo 1.000 (o treino para no fim do episódio seguinte) e
depois retoma. registro.csv, episodios.csv, avaliacoes.csv (e curriculo.csv) de A e B
têm de ser idênticos. Para exercitar tudo em 3.000 passos: treino a partir do passo 500,
rede alvo a cada 100 atualizações e avaliação a cada 1.000 passos; com currículo,
avaliação do nível a cada 500 passos e no máximo 1.000 passos por nível (passa pelos 3
níveis, e a parada cai no meio do currículo).
"""
import argparse
import filecmp
import json
import shutil
import subprocess
import sys
import time

from .dqn import ARQ_PARAR, CODIGO_PARADO, PASTA_LOGS

ARGS = ["--semente", "0", "--passos", "3000", "--inicio", "500", "--alvo-cada", "100",
        "--aval-cada", "1000"]
ARGS_CURRICULO = ["--cur-aval-cada", "500", "--cur-max-nivel", "1000"]


def comando(id_, receita, arquitetura, curriculo, exploracao):
    extra = ARGS_CURRICULO if curriculo == "sim" else []
    return [sys.executable, "-m", "rl.dqn", "--id", id_, "--receita", receita,
            "--arquitetura", arquitetura, "--curriculo", curriculo,
            "--exploracao", exploracao] + ARGS + extra


def main(receita="baseline", arquitetura="cnn1", curriculo="nao", exploracao="uniforme"):
    sys.stdout.reconfigure(encoding="utf-8")
    nome = receita if arquitetura == "cnn1" and curriculo == "nao" else \
        f"{arquitetura}{'-cur' if curriculo == 'sim' else ''}"
    nome += "-int" if exploracao == "intencional" else ""
    a, b = f"teste-retomada-{nome}-A", f"teste-retomada-{nome}-B"
    cmd = lambda id_: comando(id_, receita, arquitetura, curriculo, exploracao)
    for id_ in (a, b):
        shutil.rmtree(PASTA_LOGS / id_, ignore_errors=True)
    ARQ_PARAR.unlink(missing_ok=True)
    assert subprocess.run(cmd(a)).returncode == 0

    proc = subprocess.Popen(cmd(b))
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
    assert subprocess.run(cmd(b)).returncode == 0

    ok = True
    arqs = ["registro.csv", "episodios.csv", "avaliacoes.csv", "avaliacao_final.csv"]
    if curriculo == "sim":
        arqs.append("curriculo.csv")
    arqs.append("exploracao.csv")
    for arq in arqs:
        igual = filecmp.cmp(PASTA_LOGS / a / arq, PASTA_LOGS / b / arq, shallow=False)
        print(f"{arq}: {'idêntico' if igual else 'DIFERENTE'}")
        ok &= igual
    linhas = sum(1 for _ in open(PASTA_LOGS / a / "registro.csv", encoding="utf-8")) - 1
    print(f"registro.csv tem {linhas} linhas; parada no passo {passo_parada} de 3000")
    for id_ in (a, b):   # o checkpoint final fica por padrão; aqui não serve para nada
        shutil.rmtree(PASTA_LOGS / id_ / "ckpt", ignore_errors=True)
    print("RETOMADA EXATA" if ok else "RETOMADA DIFERENTE")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("receita", nargs="?", default="baseline")
    p.add_argument("--arquitetura", default="cnn1")
    p.add_argument("--curriculo", default="nao")
    p.add_argument("--exploracao", default="uniforme")
    sys.exit(0 if main(**vars(p.parse_args())) else 1)
