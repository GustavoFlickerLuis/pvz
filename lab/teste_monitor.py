"""Teste de ponta a ponta do monitor (processos reais, mortes à força).

    python -m lab.teste_monitor

Fila de teste (lab/fila_teste_monitor.json), 4 execuções curtas (calmo, 0,5, 80k passos,
checkpoint a cada 20k):
  a      : quando passa o checkpoint de 20k, o TREINO é morto à mão (taskkill /F) →
           o monitor tem de registrar a queda, religar e a execução retomar do checkpoint;
  falha  : arquitetura inexistente (quebra no começo) → 3 quedas seguidas sem avançar →
           marcada como falha, com o fim do log, e a fila segue;
  b      : quando passa o checkpoint de 20k, o processo DA FILA é morto sozinho (o treino
           fica órfão) → o monitor espera o órfão acabar, sem abrir um segundo treino;
  c      : no meio, cria-se lab/PARAR → a fila para com checkpoint e o monitor sai sem
           religar; depois apaga-se o PARAR, chama-se o monitor de novo e a fila termina.
Não rode com a fila de verdade ligada (o lab/PARAR é do lab inteiro).
"""
import json
import shutil
import subprocess
import sys
import time

from . import LOGS, PARAR, PASTA_LAB
from .fila import FILHO, arquivo_falhas
from .processos import matar, vivo

BASE = dict(fases="calmo", dificuldade=0.5, passos=80_000, aval_cada=10 ** 9,
            aval_sementes=[300, 302], aval_final_sementes=[300, 302], ckpt_cada=20_000)
FILA = PASTA_LAB / "fila_teste_monitor.json"
LOG = PASTA_LAB / "teste_monitor_monitor.log"
SAIDA = PASTA_LAB / "teste_monitor_fila.log"
IDS = {"a": "lab-teste-mon-a", "falha": "lab-teste-mon-falha", "b": "lab-teste-mon-b",
       "c": "lab-teste-mon-c"}


def ck(id_):
    f = LOGS / id_ / "ckpt.json"
    return json.loads(f.read_text())["passo"] if f.exists() else 0


def esperar(cond, monitor, limite=300):
    t0 = time.time()
    while not cond():
        assert monitor.poll() is None, "o monitor saiu antes da hora"
        assert time.time() - t0 < limite, "tempo esgotado"
        time.sleep(0.2)


def filho():
    try:
        return json.loads(FILHO.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def monitor():
    return subprocess.Popen([sys.executable, "-u", "-m", "lab.monitor", "--fila", str(FILA),
                             "--log", str(LOG), "--saida", str(SAIDA), "--espera", "3",
                             "--depois", ""], cwd=PASTA_LAB.parent)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    assert not PARAR.exists(), f"apague {PARAR} antes"
    for i in IDS.values():
        shutil.rmtree(LOGS / i, ignore_errors=True)
        (LOGS / f"{i}.log").unlink(missing_ok=True)
    for f in (LOG, SAIDA, arquivo_falhas(FILA)):
        f.unlink(missing_ok=True)
    FILA.write_text(json.dumps([
        {"id": IDS["a"], "config": {**BASE, "semente": 11}},
        {"id": IDS["falha"], "config": {**BASE, "arquitetura": "nao_existe"}},
        {"id": IDS["b"], "config": {**BASE, "semente": 12}},
        {"id": IDS["c"], "config": {**BASE, "semente": 13}}]), encoding="utf-8")
    m = monitor()

    # a: matar o treino
    esperar(lambda: ck(IDS["a"]) >= 20_000 and (filho() or {}).get("id") == IDS["a"], m)
    f, passo_a = filho(), ck(IDS["a"])
    matar(f["pid"], arvore=False)
    print(f"[a] treino PID {f['pid']} morto à mão; último checkpoint no passo {passo_a}")
    esperar(lambda: (LOGS / IDS["a"] / "resultado.json").exists(), m)
    log_a = (LOGS / f"{IDS['a']}.log").read_text(encoding="utf-8")
    assert f"no passo {passo_a}" in log_a and "retomando" in log_a, "a não retomou do checkpoint"
    print(f"[a] OK: o monitor religou e a execução retomou do passo {passo_a} e terminou")

    # falha: 3 quedas seguidas → marcada e pulada
    esperar(lambda: arquivo_falhas(FILA).exists(), m)
    fl = json.loads(arquivo_falhas(FILA).read_text(encoding="utf-8"))
    assert list(fl) == [IDS["falha"]] and fl[IDS["falha"]]["tentativas"] == 3
    print(f"[falha] OK: marcada depois de 3 quedas; fim do log guardado:\n    "
          + fl[IDS["falha"]]["fim_do_log"].strip().splitlines()[-1])

    # b: matar só a fila; o treino fica órfão
    esperar(lambda: ck(IDS["b"]) >= 20_000 and (filho() or {}).get("id") == IDS["b"], m)
    f = filho()
    matar(f["pid_fila"], arvore=False)
    print(f"[b] processo da fila PID {f['pid_fila']} morto à mão; o treino PID {f['pid']} continua")
    time.sleep(8)
    assert vivo(f["pid"]), "o treino órfão morreu junto"
    assert not (LOGS / IDS["c"]).exists(), "o monitor religou com o órfão vivo"
    esperar(lambda: (LOGS / IDS["b"] / "resultado.json").exists(), m)
    print("[b] OK: o monitor esperou o órfão terminar antes de religar")

    # c: PARAR no meio → para com checkpoint e o monitor não religa
    esperar(lambda: ck(IDS["c"]) >= 20_000 and (filho() or {}).get("id") == IDS["c"], m)
    PARAR.write_text("")
    m.wait(timeout=300)
    assert not (LOGS / IDS["c"] / "resultado.json").exists()
    passo_c = ck(IDS["c"])
    time.sleep(10)
    assert not FILHO.exists() and "parado" in LOG.read_text(encoding="utf-8")
    print(f"[c] OK: com {PARAR.name} a fila parou com checkpoint no passo {passo_c} e o monitor "
          f"saiu sem religar (nada rodou nos 10 s seguintes)")
    PARAR.unlink()
    m = monitor()
    m.wait(timeout=300)
    assert (LOGS / IDS["c"] / "resultado.json").exists()
    assert f"no passo {passo_c}" in (LOGS / f"{IDS['c']}.log").read_text(encoding="utf-8")
    assert "completa" in LOG.read_text(encoding="utf-8").splitlines()[-1]
    print(f"[c] OK: sem o {PARAR.name}, o monitor religou, retomou do passo {passo_c} e saiu "
          f"com a fila completa")
    print("\n--- log do monitor ---\n" + LOG.read_text(encoding="utf-8"))
    FILA.unlink()
    arquivo_falhas(FILA).unlink()


if __name__ == "__main__":
    main()
