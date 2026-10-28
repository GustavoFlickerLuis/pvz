"""Monitor da fila do lab: religa a fila se ela morrer. Rode numa janela própria do
PowerShell, fora do Claude Code:

    cd C:\\Users\\gului\\Desktop\\pvz-lab
    python -m lab.monitor

Regras:
- se o processo da fila morrer (código ≠ 0, ou saiu sem terminar), espera ~30 s e religa;
  a fila pula as execuções feitas e retoma a interrompida do último checkpoint (ckpt.pt,
  gravado a cada 250k passos, no PARAR e no fim);
- se a fila morrer sozinha e o treino dela continuar vivo (órfão), espera o treino acabar
  antes de religar, para nunca haver dois treinos na GPU nem na mesma pasta;
- NÃO religa se existir o arquivo de parada do lab (lab/PARAR) ou se a fila tiver terminado;
  nesses casos o monitor sai;
- se a mesma execução falhar 3 vezes seguidas sem o checkpoint avançar, ela é marcada como
  falha (lab/<fila>_falhas.json, com o fim do log; o LAB.md mostra na seção da fila) e a fila
  segue para a próxima. Uma queda em que o checkpoint avançou zera a contagem (uma morte por
  falta de memória no passo 1,2M não é a mesma falha que um erro que se repete no mesmo lugar);
- cada início, queda, religada, falha e parada vai para lab/monitor.log (horário, execução,
  passo, motivo). A saída da fila vai para lab/fila_saida.log.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import LOGS, PARAR, PASTA_LAB
from .fila import FILA, FILHO, arquivo_falhas, falhas
from .processos import vivo


def registrar(log, evento, execucao="", passo="", motivo=""):
    linha = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {evento:<9} | {execucao} | passo {passo} | {motivo}"
    print(linha, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def jobs(fila):
    return json.loads(Path(fila).read_text(encoding="utf-8"))


def pendente(fila):
    """Primeira execução sem resultado.json e não marcada como falha (None = fila terminada)."""
    fl = falhas(fila)
    return next((j["id"] for j in jobs(fila)
                 if not (LOGS / j["id"] / "resultado.json").exists() and j["id"] not in fl), None)


def passo_ckpt(id_):
    f = LOGS / id_ / "ckpt.json"
    return json.loads(f.read_text(encoding="utf-8"))["passo"] if f.exists() else 0


def passo_no_log(id_):
    """Último passo impresso no log da execução (para registrar onde ela estava ao morrer)."""
    f = LOGS / f"{id_}.log"
    if not f.exists():
        return ""
    achados = re.findall(r"passo (\d+)", f.read_text(encoding="utf-8", errors="replace")[-20_000:])
    return achados[-1] if achados else ""


def fim_do_log(id_, linhas=40):
    f = LOGS / f"{id_}.log"
    return "\n".join(f.read_text(encoding="utf-8", errors="replace").splitlines()[-linhas:]) if f.exists() else ""


def esperar_orfao(log):
    """Se o treino da fila morta ainda estiver vivo, espera ele acabar."""
    if not FILHO.exists():
        return
    f = json.loads(FILHO.read_text(encoding="utf-8"))
    if vivo(f["pid"]):
        registrar(log, "órfão", f["id"], passo_no_log(f["id"]),
                  f"o treino (PID {f['pid']}) continua vivo sem a fila; esperando ele terminar")
        while vivo(f["pid"]):
            time.sleep(5)
    FILHO.unlink(missing_ok=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fila", default=str(FILA))
    p.add_argument("--log", default=str(PASTA_LAB / "monitor.log"))
    p.add_argument("--saida", default=str(PASTA_LAB / "fila_saida.log"))
    p.add_argument("--espera", type=float, default=30.0, help="segundos antes de religar")
    p.add_argument("--tentativas", type=int, default=3)
    p.add_argument("--depois", default="lab.relatorio")
    a = p.parse_args()
    seguidas, primeira = {}, True
    esperar_orfao(a.log)
    while True:
        if PARAR.exists():
            registrar(a.log, "parado", motivo=f"{PARAR} existe; o monitor não religa e sai")
            return
        id_ = pendente(a.fila)
        if id_ is None:
            registrar(a.log, "completa", motivo="fila terminada; o monitor sai")
            return
        ck0 = {j["id"]: passo_ckpt(j["id"]) for j in jobs(a.fila)}   # onde cada uma estava
        registrar(a.log, "início" if primeira else "religada", id_, ck0[id_],
                  "monitor começou" if primeira else f"tentativa {seguidas.get(id_, 0) + 1} desde a última vez que avançou")
        primeira = False
        cmd = [sys.executable, "-u", "-m", "lab.fila", "--fila", a.fila, "--depois", a.depois]
        with open(a.saida, "a", encoding="utf-8") as saida:
            saida.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} monitor: {' '.join(cmd)}\n")
            saida.flush()
            cod = subprocess.run(cmd, stdout=saida, stderr=subprocess.STDOUT,
                                 cwd=PASTA_LAB.parent).returncode
        if cod == 0 and (PARAR.exists() or pendente(a.fila) is None):
            continue                      # parada pedida ou fila completa: o topo do laço registra e sai
        # queda: a fila saiu sem terminar. A execução que caiu é a do treino em curso (lab/filho.json),
        # ou, se a fila caiu entre execuções, a primeira pendente
        id_ = (json.loads(FILHO.read_text(encoding="utf-8"))["id"] if FILHO.exists()
               else pendente(a.fila) or id_)
        passo = passo_no_log(id_)
        esperar_orfao(a.log)
        registrar(a.log, "queda", id_, passo, f"a fila saiu com código {cod}")
        avancou = (LOGS / id_ / "resultado.json").exists() or passo_ckpt(id_) > ck0.get(id_, 0)
        seguidas[id_] = 0 if avancou else seguidas.get(id_, 0) + 1
        if seguidas[id_] >= a.tentativas:
            fl = falhas(a.fila)
            fl[id_] = {"horario": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "tentativas": seguidas[id_],
                       "passo_ckpt": passo_ckpt(id_), "codigo": cod, "fim_do_log": fim_do_log(id_)}
            arquivo_falhas(a.fila).write_text(json.dumps(fl, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
            registrar(a.log, "falhou", id_, passo,
                      f"{seguidas[id_]} quedas seguidas sem o checkpoint avançar; marcada como falha "
                      f"e pulada (fim do log no LAB.md)")
            if a.depois:
                subprocess.run([sys.executable, "-m", a.depois], cwd=PASTA_LAB.parent)
        if PARAR.exists():
            continue
        time.sleep(a.espera)


if __name__ == "__main__":
    main()
