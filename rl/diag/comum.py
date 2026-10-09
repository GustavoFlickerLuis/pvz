"""Peças comuns do diagnóstico: a política de coleta β, a observação v2 e o k fixo.

β = Econ5 com 15% de ações aleatórias (uniforme entre as válidas, "esperar" incluída), a
mesma regra de desafio3/dados.py. O Econ5 lê o estado interno (é um baseline escrito à mão,
exceção do CLAUDE.md); o resto só usa reset/step/observar/mascara (e amb.clonar() nas
provas, a exceção da análise de erros, CONCEITO 2.7).
"""
import random

import numpy as np
import torch

from pvz.agentes import Econ
from pvz.ambiente import ESPERAR, N_CANAIS, N_PLANTAS, SOL_REF_V2, SOL_TETO_V2
from pvz import config as C

K = 20          # escolhido em tools/medir_k.md
GAMMA = 0.994   # o par de k = 20
EPS_BETA = 0.15
ECON5 = Econ(5)


def acao_beta(amb, rng, m=None):
    m = amb.mascara() if m is None else m
    if rng.random() < EPS_BETA:
        return int(rng.choice(np.flatnonzero(m)))
    return int(ECON5.agir(amb))


def tipos_disponiveis(m):
    return [t for t in range(N_PLANTAS) if m[t * 45:(t + 1) * 45].any()]


def para_v2(obs, sol, mascara):
    """Observação v2 (25 canais) a partir da v1 (20), do sol e da máscara (arrays numpy ou
    tensores, com o lote na 1ª dimensão)."""
    if isinstance(obs, np.ndarray):
        obs, sol, mascara = map(torch.from_numpy, (obs, sol, mascara))
    x = torch.cat([obs.float(), mascara[:, :ESPERAR].reshape(-1, N_PLANTAS, C.FAIXAS,
                                                            C.COLUNAS).float()], dim=1)
    x[:, 11] = torch.clamp(sol.float()[:, None, None] / SOL_REF_V2, max=SOL_TETO_V2)
    return x


def entrada(formato, obs, sol, mascara):
    return obs.float() if formato == "v1" else para_v2(obs, sol, mascara)


assert N_CANAIS == 20
