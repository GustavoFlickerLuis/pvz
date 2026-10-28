"""Fila do lab: uma execução do PPO por vez, com parada e retomada.

    python -m lab.fila                       # roda a fila de lab/fila.json, de onde parou
    python -m lab.fila --fila outra.json     # outra fila (mesmo formato)
    python -m lab.fila --status              # só mostra o estado

Parar: criar o arquivo lab/PARAR (caminho em lab.PARAR). A execução em curso grava o
checkpoint no fim da atualização em andamento e sai; a fila não começa a próxima.
Retomar: apagar lab/PARAR e chamar de novo. Execuções com resultado.json são puladas;
a interrompida continua do ckpt.pt (retomada exata: partidas em voo e geradores salvos).

Formato da fila (JSON): lista de {"id": "...", "config": {campos da rl.ppo.Config}}.
Saída de cada execução em rl/logs/<id>.log (acrescentada); logs em rl/logs/<id>/.
Se uma execução falhar (código ≠ 0), a fila para e não pula para a próxima; quem decide
religar ou pular é o lab.monitor. Execuções marcadas como falhas pelo monitor (arquivo
<fila>_falhas.json) são puladas. A fila grava o PID do treino em curso em lab/filho.json e a
RAM (RSS) dela e do treino em lab/memoria.csv, no começo e no fim de cada execução.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import LOGS, PARAR, PASTA_LAB
from .processos import rss_mb

FILA = PASTA_LAB / "fila.json"
FILHO = PASTA_LAB / "filho.json"
MEMORIA = PASTA_LAB / "memoria.csv"


def arquivo_falhas(fila):
    fila = Path(fila)
    return fila.with_name(fila.stem + "_falhas.json")


def falhas(fila):
    f = arquivo_falhas(fila)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def anotar_memoria(**linha):
    novo = not MEMORIA.exists()
    with open(MEMORIA, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["horario", "id", "momento", "pid_fila", "rss_fila_mb",
                                          "rss_treino_inicio_mb", "rss_treino_fim_mb",
                                          "rss_treino_pico_mb"])
        if novo:
            w.writeheader()
        w.writerow({"horario": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "pid_fila": os.getpid(),
                    "rss_fila_mb": rss_mb(), **linha})


def estado(job, fl=None):
    pasta = LOGS / job["id"]
    if fl and job["id"] in fl:
        return "FALHOU (marcada pelo monitor; pulada)"
    if (pasta / "resultado.json").exists():
        r = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
        return f"feito {100 * r['vitorias_final']:.1f}% ({r['minutos']:.0f} min)"
    if (pasta / "ckpt.pt").exists():
        passo = json.loads((pasta / "ckpt.json").read_text())["passo"] if (pasta / "ckpt.json").exists() else "?"
        return f"em curso ou interrompido (ckpt no passo {passo})"
    return "na fila"


def rodar_job(job):
    cfg = dict(job["config"])
    cfg["arquivo_parar"] = str(PARAR)
    cmd = [sys.executable, "-u", "-m", "rl.ppo", "--id", job["id"], "--continuar",
           "--json", json.dumps(cfg)]
    with open(LOGS / f"{job['id']}.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PASTA_LAB.parent)
        FILHO.write_text(json.dumps({"pid": proc.pid, "id": job["id"], "pid_fila": os.getpid(),
                                     "inicio": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}),
                         encoding="utf-8")
        cod = proc.wait()
    FILHO.unlink(missing_ok=True)
    return cod


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fila", default=str(FILA))
    p.add_argument("--status", action="store_true")
    p.add_argument("--depois", help="comando (módulo python) a rodar depois de cada execução",
                   default="lab.relatorio")
    a = p.parse_args()
    fila = json.loads(open(a.fila, encoding="utf-8").read())
    fl = falhas(a.fila)
    for job in fila:
        print(f"{job['id']:<34} {estado(job, fl)}", flush=True)
    if a.status:
        return
    for job in fila:
        if (LOGS / job["id"] / "resultado.json").exists() or job["id"] in fl:
            continue
        if PARAR.exists():
            print(f"[fila] {PARAR} existe: parada antes de {job['id']}", flush=True)
            return
        t0 = time.time()
        print(f"[fila] {datetime.now():%H:%M} começando {job['id']}", flush=True)
        anotar_memoria(id=job["id"], momento="inicio")
        cod = rodar_job(job)
        res = LOGS / job["id"] / "resultado.json"
        r = json.loads(res.read_text(encoding="utf-8")) if cod == 0 and res.exists() else {}
        anotar_memoria(id=job["id"], momento=f"fim (código {cod})",
                       rss_treino_inicio_mb=r.get("rss_inicio_mb", ""),
                       rss_treino_fim_mb=r.get("rss_fim_mb", ""),
                       rss_treino_pico_mb=r.get("rss_pico_mb", ""))
        print(f"[fila] {job['id']}: código {cod}, {estado(job)}, {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if a.depois:
            subprocess.run([sys.executable, "-m", a.depois], cwd=PASTA_LAB.parent)
        if cod != 0:
            print(f"[fila] {job['id']} falhou (código {cod}); a fila para aqui. Ver "
                  f"rl/logs/{job['id']}.log", flush=True)
            sys.exit(cod)
        if PARAR.exists():
            print(f"[fila] {PARAR} existe: fila parada depois de {job['id']}", flush=True)
            return
    print("[fila] fila completa", flush=True)


if __name__ == "__main__":
    main()
