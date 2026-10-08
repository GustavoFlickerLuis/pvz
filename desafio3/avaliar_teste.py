"""Avaliação no conjunto de TESTE (Desafio 3): roda UMA vez.

    python -m desafio3.avaliar_teste

1. Para cada peso em desafio3/pesos/ (os retreinos com --salvar-pesos), confere que o
   val_final do retreino é idêntico ao da execução correspondente no experimentos.csv.
   Se algum não bater, para sem carregar o teste.
2. Carrega o teste (dados.carregar("teste", permitir_teste=True)) e mede a Huber total,
   esperar e plantio de cada modelo.
3. Grava desafio3/teste.csv e imprime média ± desvio por config e o critério de vitória.
Recusa rodar se desafio3/teste.csv já existir.
"""
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from pvz.ambiente import ESPERAR

from .dados import carregar
from .modelo import CNN
from .treino import EXPERIMENTOS, PASTA_LOGS, PASTA_PESOS, huber_medio

SAIDA = Path(__file__).parent / "teste.csv"
CONFIGS = {   # rótulo: (otimizador, lr, agenda, etapa no experimentos.csv)
    "vencedora: sgd 3e-2 cosseno": ("sgd", 3e-2, "cosseno", "confirmacao"),
    "baseline: adam 1e-4 constante": ("adam", 1e-4, "constante", "baseline"),
    "muon 1e-3 warmup_cosseno": ("muon", 1e-3, "warmup_cosseno", "confirmacao"),
}


def rotulo_de(cfg):
    for r, (o, lr, ag, _) in CONFIGS.items():
        if cfg["otimizador"] == o and math.isclose(cfg["lr"], lr) and cfg["agenda"] == ag:
            return r
    return None


def original(exps, rotulo, semente):
    o, lr, ag, etapa = CONFIGS[rotulo]
    achados = [e for e in exps if e["etapa"] == etapa and e["otimizador"] == o
               and math.isclose(float(e["lr"]), lr) and e["agenda"] == ag
               and int(e["semente"]) == semente and e["val_esperar"] != ""]
    assert len(achados) == 1, (rotulo, semente, len(achados))
    return achados[0]


def main():
    if SAIDA.exists():
        raise SystemExit(f"{SAIDA} já existe: o teste só é avaliado uma vez")
    with open(EXPERIMENTOS, newline="", encoding="utf-8") as f:
        exps = list(csv.DictReader(f))

    modelos = []
    for peso in sorted(PASTA_PESOS.glob("*.pt")):
        with open(PASTA_LOGS / "rascunho" / peso.stem / "config.json", encoding="utf-8") as f:
            info = json.load(f)
        r = rotulo_de(info["config"])
        if r is None:
            continue
        semente = info["config"]["semente"]
        orig = original(exps, r, semente)
        igual = float(orig["val_final"]) == info["resultado"]["val_final"]
        print(f"{r} s{semente}: retreino {info['resultado']['val_final']!r} × "
              f"experimentos {orig['val_final']} → {'idêntico' if igual else 'DIFERENTE'}")
        if not igual:
            raise SystemExit("retreino não reproduz o val_final; o teste NÃO foi carregado")
        modelos.append((r, semente, peso, orig["id"]))
    contagem = {r: sum(m[0] == r for m in modelos) for r in CONFIGS}
    if any(n != 5 for n in contagem.values()):
        raise SystemExit(f"esperava 5 sementes por config: {contagem}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teste = carregar("teste", permitir_teste=True)
    obs = torch.from_numpy(teste["obs"]).to(dev)
    acao = torch.from_numpy(teste["acao"]).to(dev, torch.long)
    alvo = torch.from_numpy(teste["G"]).to(dev)
    esperar = acao == ESPERAR

    linhas = []
    for r, semente, peso, id_original in modelos:
        modelo = CNN().to(dev)
        modelo.load_state_dict(torch.load(peso, map_location=dev))
        total, h_esp, h_pla = huber_medio(modelo, obs, acao, alvo, 1.0, grupo=esperar)
        linhas.append({"config": r, "semente": semente, "teste_huber": total,
                       "teste_esperar": h_esp, "teste_plantio": h_pla,
                       "id_validacao": id_original, "pesos": peso.name})
    with open(SAIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    print(f"\nteste: {len(teste['G'])} amostras ({int(esperar.sum())} esperar)\n")
    est = {}
    for r in CONFIGS:
        vals = {k: np.array([l[k] for l in linhas if l["config"] == r])
                for k in ("teste_huber", "teste_esperar", "teste_plantio")}
        est[r] = vals
        print(r, " | ".join(f"{k} {v.mean():.5f} ± {v.std(ddof=1):.5f}" for k, v in vals.items()))
    print()
    for a, b in itertools.combinations(CONFIGS, 2):
        x, y = est[a]["teste_huber"], est[b]["teste_huber"]
        sc = math.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2)
        z = (x.mean() - y.mean()) / sc
        quem = "empate" if abs(z) <= 2 else (f"{b} vence" if z > 0 else f"{a} vence")
        print(f"{a} × {b}: Δ = {x.mean() - y.mean():+.5f}, Δ/s_c = {z:+.2f} → {quem}")


if __name__ == "__main__":
    main()
