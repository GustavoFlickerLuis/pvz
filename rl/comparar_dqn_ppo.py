"""DQN × PPO nas mesmas condições (cnn1d, k = 20, γ = 0,994, 2M passos), sementes 0–2.

    python -m rl.comparar_dqn_ppo

Avalia em 420 partidas (FASES_TREINO × 300–319, gulosa) o melhor.pt de cada DQN, junta o
resultado final, o tempo de relógio (rl/logs/fila-dqn-2M.log) e os logs do PPO, e imprime a
tabela da seção "DQN comparável (3 sementes, 2M)" do rl/ppo_RELATORIO.md. Só usa
reset/step/observar/mascara (via rl.avaliar).
"""
import csv
import json
import math
import sys
from datetime import datetime

import numpy as np
import torch

from desafio3.modelo import criar, formato

from . import avaliar
from .dqn import PASTA_LOGS

ECON5 = 0.652
DQN = [f"dqn-cnn1d-k20-2M-s{s}" for s in range(3)]
PPO = ["ppo-fases-s0-cpu", "ppo-fases-s1", "ppo-fases-s2"]
PPO_8 = PPO + [f"ppo-fases-s{s}" for s in range(3, 8)]


def tempos_dqn():
    """Minutos de relógio de cada execução, pelos carimbos da fila (soma os trechos, se
    ela foi interrompida e retomada; o trecho entre o último checkpoint e a queda conta)."""
    ini, t = {}, {}
    for linha in (PASTA_LOGS / "fila-dqn-2M.log").read_text(encoding="utf-8").splitlines():
        p = linha.split()
        if len(p) < 5 or p[2] not in ("início", "fim"):   # anotações do supervisor
            continue
        quando = datetime.fromisoformat(f"{p[3]} {p[4]}")
        if p[2] == "início":
            ini[p[1]] = quando
        elif p[1] in ini:
            t[p[1]] = t.get(p[1], 0.0) + (quando - ini.pop(p[1])).total_seconds() / 60
    return t


def periodicas(id_):
    linhas = list(csv.DictReader(open(PASTA_LOGS / id_ / "avaliacoes.csv", encoding="utf-8")))
    return [(int(l["passo"]), float(l["vitorias"])) for l in linhas
            if l.get("modo", "gulosa") == "gulosa" and int(l["partidas"]) == 63]


def melhor_periodica(id_):
    """(passo, vitórias) da primeira avaliação periódica com o máximo (é a do melhor.pt)."""
    return max(periodicas(id_), key=lambda pv: (pv[1], -pv[0]))


def media_dp(xs):
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def main():
    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lista = avaliar.partidas(range(300, 320))
    t = tempos_dqn()
    linhas = []
    for id_ in DQN:
        pasta = PASTA_LOGS / id_
        cfg = json.loads((pasta / "config.json").read_text(encoding="utf-8"))["config"]
        assert (cfg["k_ticks"], cfg["gamma"], cfg["passos"]) == (20, 0.994, 2_000_000)
        m = criar(cfg["arquitetura"]).to(dev)
        m.load_state_dict(torch.load(pasta / "melhor.pt", map_location=dev))
        r = avaliar.avaliar_rede(m, lista, dev, k=cfg["k_ticks"], formato=formato(cfg["arquitetura"]))
        res = json.loads((pasta / "resultado.json").read_text(encoding="utf-8"))
        passo, v63 = melhor_periodica(id_)
        linhas.append(("DQN", cfg["semente"], res["vitorias_final"], r["vitorias"], passo, v63,
                       t.get(id_, math.nan)))
    for s, id_ in enumerate(PPO):
        res = json.loads((PASTA_LOGS / id_ / "resultado.json").read_text(encoding="utf-8"))
        passo, v63 = melhor_periodica(id_)
        # o melhor.pt do PPO não está neste computador; se a melhor periódica é a do fim,
        # o melhor checkpoint é o modelo final e vale a avaliação final de 420 partidas
        melhor420 = res["vitorias_final"] if passo == res["passos"] else math.nan
        linhas.append(("PPO", s, res["vitorias_final"], melhor420, passo, v63, res["minutos"]))

    print("| alg. | semente | final (420) | melhor ckpt (420) | melhor periódica (63) "
          "| passa do Econ5? | tempo (min) |")
    print("|---|---|---|---|---|---|---|")
    for alg, s, fin, mel, passo, v63, mins in linhas:
        print(f"| {alg} | {s} | {100 * fin:.1f}% | {100 * mel:.1f}% | {100 * v63:.1f}% "
              f"({passo / 1e6:.2f}M) | {'sim' if fin > ECON5 else 'não'} | {mins:.0f} |")
    fin = {a: [l[2] for l in linhas if l[0] == a] for a in ("DQN", "PPO")}
    mel = {a: [l[3] for l in linhas if l[0] == a] for a in ("DQN", "PPO")}
    for a in ("DQN", "PPO"):
        m, d = media_dp(fin[a])
        mm, dm = media_dp(mel[a])
        print(f"{a}: final {100 * m:.1f} ± {100 * d:.1f}; melhor ckpt {100 * mm:.1f} ± {100 * dm:.1f}; "
              f"passam do Econ5: {sum(x > ECON5 for x in fin[a])}/3")
    (ma, sa), (mb, sb) = media_dp(fin["DQN"]), media_dp(fin["PPO"])
    sc = math.sqrt((sa ** 2 + sb ** 2) / 2)
    delta = mb - ma
    print(f"Δ (PPO − DQN, final) = {100 * delta:.1f} pontos; s_c = {100 * sc:.1f}; 2 s_c = "
          f"{200 * sc:.1f} → {'vence' if abs(delta) > 2 * sc else 'empate formal'}")
    v8 = [json.loads((PASTA_LOGS / i / "resultado.json").read_text(encoding="utf-8"))["vitorias_final"]
          for i in PPO_8]
    m8, d8 = media_dp(v8)
    print("PPO, 8 sementes a 2M: " + " / ".join(f"{100 * v:.1f}" for v in v8)
          + f"; média {100 * m8:.1f} ± {100 * d8:.1f}; passam do Econ5: {sum(v > ECON5 for v in v8)}/8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
