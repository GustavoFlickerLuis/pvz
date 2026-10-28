"""Parâmetros e passos/s de cada braço no PPO (receita atual, FASES_TREINO, semente 0).

    python -m lab.medir [--coletas 12]      # → lab/medidas.json

Mede o PPO completo (coleta de 16 partidas + 4 épocas de atualização), descartando a
primeira coleta (aquecimento da GPU). No começo do treino as partidas são curtas e a
política é quase aleatória; a velocidade cai ao longo do treino (a cnn1d faz ~2.200 passos/s
no começo e ~1.700 no fim), então a estimativa para 2M é feita pela razão com a cnn1d:
tempo_braço ≈ tempo real da cnn1d a 2M × (passos/s da cnn1d ÷ passos/s do braço).
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from desafio3.modelo import ARQUITETURAS
from rl.ppo import Config, Treino

from . import PASTA_LAB

BRACOS = ("cnn1d", "atencao_antes", "atencao_depois", "transformer", "transformer_d128",
          "cnn1d_dilatada")


def medir(arq, coletas):
    tr = Treino(Config(arquitetura=arq, passos=10 ** 9), Path(tempfile.mkdtemp()))
    tr.atualizar(*tr.coletar()[:5])
    torch.cuda.synchronize()
    p0, t0 = tr.passo, time.time()
    tc = ta = 0.0
    for _ in range(coletas):
        t = time.time()
        X, A, LP, adv, ret, _ = tr.coletar()
        torch.cuda.synchronize()
        tc += time.time() - t
        t = time.time()
        tr.atualizar(X, A, LP, adv, ret)
        torch.cuda.synchronize()
        ta += time.time() - t
    dt = time.time() - t0
    return {"parametros": sum(p.numel() for p in tr.pol.parameters()),
            "passos_por_s": (tr.passo - p0) / dt, "s_coleta": tc / coletas,
            "s_atualizacao": ta / coletas}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coletas", type=int, default=12)
    a = p.parse_args()
    res = {}
    for arq in BRACOS:
        res[arq] = medir(arq, a.coletas)
        print(arq, {k: round(v, 3) for k, v in res[arq].items()}, flush=True)
    ref = res["cnn1d"]["passos_por_s"]
    for arq in BRACOS:
        res[arq]["razao_cnn1d"] = ref / res[arq]["passos_por_s"]
    res["_nota"] = (f"{a.coletas} coletas (2.048 passos cada) depois de 1 de aquecimento, "
                    f"FASES_TREINO, semente 0, receita atual, {torch.cuda.get_device_name()}")
    (PASTA_LAB / "medidas.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
