"""Teste do checkpoint automático com morte de verdade (como a do Claude Code às 17h).

    python -m lab.teste_ckpt

1. Referência: 100k passos direto, sem checkpoint automático (lab-teste-ckpt-ref).
2. Mesma config com ckpt_cada = 50k (lab-teste-ckpt-b): assim que o checkpoint de ~50k sai,
   o processo é MORTO (taskkill /F, sem chance de gravar nada); depois chama-se de novo com
   --continuar, que retoma do checkpoint e vai até 100k.
3. Confere: registro.csv (sem a coluna passos/s), avaliacoes.csv, vitórias finais e pesos
   iguais aos da referência.
Receita v2b (coef_ent 0,03 + η linear até 0), para testar também a agenda de η.
"""
import json
import shutil
import subprocess
import sys
import time

import torch

from . import LOGS, PASTA_LAB
from .processos import matar

CFG = dict(passos=100_000, aval_cada=50_000, aval_sementes=[300, 302],
           aval_final_sementes=[300, 302], semente=3, coef_ent=0.03, receita="atual_linear")
REF, B = "lab-teste-ckpt-ref", "lab-teste-ckpt-b"


def ppo(id_, cfg, **kw):
    cmd = [sys.executable, "-u", "-m", "rl.ppo", "--id", id_, "--continuar", "--json", json.dumps(cfg)]
    log = open(LOGS / f"{id_}.log", "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=PASTA_LAB.parent, stdout=log, stderr=subprocess.STDOUT, **kw)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    for i in (REF, B):
        shutil.rmtree(LOGS / i, ignore_errors=True)
        (LOGS / f"{i}.log").unlink(missing_ok=True)
    t0 = time.time()
    assert ppo(REF, {**CFG, "ckpt_cada": 0}).wait() == 0
    print(f"referência: {time.time() - t0:.0f} s")
    assert not (LOGS / REF / "ckpt.json").exists() or \
        json.loads((LOGS / REF / "ckpt.json").read_text())["passo"] >= 100_000, "ckpt intermediário na ref"
    cfg_b = {**CFG, "ckpt_cada": 50_000}
    proc = ppo(B, cfg_b)
    ck = LOGS / B / "ckpt.json"
    while not ck.exists():
        assert proc.poll() is None, "o processo saiu antes do checkpoint"
        time.sleep(0.2)
    passo = json.loads(ck.read_text())["passo"]
    matar(proc.pid)
    proc.wait()
    assert not (LOGS / B / "resultado.json").exists()
    print(f"morto à força logo depois do checkpoint automático do passo {passo} "
          f"(código {proc.returncode})")
    assert ppo(B, cfg_b).wait() == 0
    log = (LOGS / f"{B}.log").read_text(encoding="utf-8")
    assert f"retomando de" in log and f"no passo {passo}" in log, "não retomou do checkpoint"
    corta = lambda i, n: [l.rsplit(",", 1)[0] if n == "registro.csv" else l for l in
                          (LOGS / i / n).read_text(encoding="utf-8").splitlines()]
    difs = {}
    for n in ("registro.csv", "avaliacoes.csv"):
        a, b = corta(REF, n), corta(B, n)
        difs[n] = [k for k, (x, y) in enumerate(zip(a, b)) if x != y] + \
                  (["tamanhos diferentes"] if len(a) != len(b) else [])
    r = [json.loads((LOGS / i / "resultado.json").read_text(encoding="utf-8")) for i in (REF, B)]
    w = [torch.load(LOGS / i / "modelo.pt") for i in (REF, B)]
    pesos_iguais = all(torch.equal(w[0][k], w[1][k]) for k in w[0])
    print(f"registro.csv: {len(corta(REF, 'registro.csv')) - 1} atualizações; linhas diferentes: "
          f"{difs['registro.csv'] or 'nenhuma'}")
    print(f"avaliacoes.csv: linhas diferentes: {difs['avaliacoes.csv'] or 'nenhuma'}")
    print(f"vitórias finais: ref {r[0]['vitorias_final']}, retomada {r[1]['vitorias_final']}; "
          f"pesos idênticos: {pesos_iguais}")
    print(f"RSS do treino (MB) ref: {r[0]['rss_inicio_mb']}→{r[0]['rss_fim_mb']} "
          f"(pico {r[0]['rss_pico_mb']}); retomada: {r[1]['rss_inicio_mb']}→{r[1]['rss_fim_mb']}")
    ok = not difs["registro.csv"] and not difs["avaliacoes.csv"] and pesos_iguais and \
        r[0]["vitorias_final"] == r[1]["vitorias_final"]
    print("OK: bit a bit igual à referência" if ok else "DIFERENTE da referência (ver acima)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
