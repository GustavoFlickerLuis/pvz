"""Teste de ponta a ponta da parada da fila (com processos, como no uso real).

    python -m lab.teste_parada

1. Roda uma referência sem parar (lab-teste-parada-ref).
2. Começa a fila [lab-teste-parada-a, lab-teste-parada-b] (a = mesma config da referência).
   Quando a execução "a" passa da atualização 10, cria lab/PARAR.
3. Confere: a fila saiu, "a" tem ckpt.pt e não tem resultado.json, "b" nem começou.
4. Apaga lab/PARAR e chama a fila de novo; confere que ela termina as duas e que "a"
   é idêntica à referência (registro.csv sem a coluna passos/s, avaliação final e pesos).
Não rode com outra execução do lab ao mesmo tempo (o lab/PARAR é da fila inteira).
"""
import json
import shutil
import subprocess
import sys
import time

import torch

from . import LOGS, PARAR, PASTA_LAB

CFG = dict(fases="calmo", dificuldade=0.5, passos=30 * 2048, aval_cada=10 ** 9,
           aval_sementes=[300, 301], aval_final_sementes=[300, 320], semente=7)
IDS = ("lab-teste-parada-ref", "lab-teste-parada-a", "lab-teste-parada-b")


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    assert not PARAR.exists(), f"apague {PARAR} antes"
    for i in IDS:
        shutil.rmtree(LOGS / i, ignore_errors=True)
        (LOGS / f"{i}.log").unlink(missing_ok=True)
    raiz = PASTA_LAB.parent
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "rl.ppo", "--id", IDS[0], "--json", json.dumps(CFG)],
                   cwd=raiz, check=True, stdout=subprocess.DEVNULL)
    print(f"referência: {time.time() - t0:.0f} s")
    fila = PASTA_LAB / "fila_teste_parada.json"
    fila.write_text(json.dumps([{"id": IDS[1], "config": CFG}, {"id": IDS[2], "config": CFG}]),
                    encoding="utf-8")
    cmd = [sys.executable, "-m", "lab.fila", "--fila", str(fila), "--depois", ""]
    proc = subprocess.Popen(cmd, cwd=raiz, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    log = LOGS / f"{IDS[1]}.log"
    while not (log.exists() and "atualização 10 " in log.read_text(encoding="utf-8")):
        time.sleep(0.5)
    PARAR.write_text("")
    print("criado", PARAR)
    saida = proc.communicate()[0]
    print(saida)
    assert proc.returncode == 0
    assert (LOGS / IDS[1] / "ckpt.pt").exists(), "sem checkpoint"
    assert not (LOGS / IDS[1] / "resultado.json").exists(), "a execução não parou"
    assert not (LOGS / IDS[2]).exists(), "a fila começou a próxima"
    passo = torch.load(LOGS / IDS[1] / "ckpt.pt", weights_only=False)["passo"]
    print(f"OK: parou no passo {passo} com checkpoint; a próxima não começou")
    PARAR.unlink()
    saida = subprocess.run(cmd, cwd=raiz, capture_output=True, text=True, encoding="utf-8").stdout
    print(saida)
    assert "retomando" in log.read_text(encoding="utf-8")
    for i in IDS[1:]:
        assert (LOGS / i / "resultado.json").exists(), i
    corta = lambda i: [l.rsplit(",", 1)[0] for l in
                       (LOGS / i / "registro.csv").read_text(encoding="utf-8").splitlines()]
    assert corta(IDS[0]) == corta(IDS[1]), "registro diferente da referência"
    r = [json.loads((LOGS / i / "resultado.json").read_text(encoding="utf-8")) for i in IDS[:2]]
    assert r[0]["vitorias_final"] == r[1]["vitorias_final"] and r[0]["passos"] == r[1]["passos"]
    w = [torch.load(LOGS / i / "modelo.pt") for i in IDS[:2]]
    assert all(torch.equal(w[0][k], w[1][k]) for k in w[0])
    fila.unlink()
    print(f"OK: retomou do passo {passo}; registro, avaliação final ({100 * r[0]['vitorias_final']:.1f}%, "
          f"20 partidas) e pesos idênticos à referência sem parada")


if __name__ == "__main__":
    main()
