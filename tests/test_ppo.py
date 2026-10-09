"""Testes do PPO (rl/ppo.py): máscara, formatos, GAE e um mini-treino de ponta a ponta."""
import json

import numpy as np
import pytest
import torch

from pvz.ambiente import ESPERAR, N_ACOES, Ambiente
from pvz.divisao import FASES_TREINO
from rl.ppo import Config, Politica, Treino, gae


def _cfg(**kw):
    base = dict(envs=8, rollout=32, passos=10 ** 9, minilote=64, aval_cada=10 ** 9,
                aval_sementes=(300, 301), aval_final_sementes=(300, 301))
    return Config(**{**base, **kw})


def test_formatos_e_mascara_nos_logits():
    torch.manual_seed(0)
    pol = Politica()
    amb = Ambiente(k=20, formato="v2")
    obs = [amb.reset(FASES_TREINO[i], i) for i in range(3)]
    x = torch.from_numpy(np.stack(obs))
    logits, v = pol(x)
    assert logits.shape == (3, N_ACOES) and v.shape == (3,)
    m = torch.from_numpy(np.stack([Ambiente(k=20).reset(FASES_TREINO[i], i) is not None and
                                   _mascara(FASES_TREINO[i], i) for i in range(3)]))
    probs = torch.softmax(logits, 1)
    assert torch.all(probs[~m] == 0) and torch.allclose(probs.sum(1), torch.ones(3))


def _mascara(fase, semente):
    amb = Ambiente(k=20)
    amb.reset(fase, semente)
    return amb.mascara()


def test_nenhuma_acao_invalida_em_milhares_de_passos(tmp_path):
    """O passo das partidas confere cada ação na máscara real (assert); aqui 8 × 400 = 3.200."""
    tr = Treino(_cfg(rollout=400), tmp_path)
    X, A, LP, adv, ret, fins = tr.coletar()
    assert tr.passo == 3200 and len(fins) > 0
    validas = torch.cat([X[:, :, 20:].flatten(2), torch.ones(*X.shape[:2], 1, device=X.device)], 2) > 0.5
    assert validas.gather(2, A[..., None]).all()
    assert (A != ESPERAR).any()          # também planta
    assert torch.isfinite(LP).all() and torch.isfinite(adv).all()


def test_gae():
    g, l = 0.9, 0.8
    r = np.array([[1.0], [0.0], [2.0]], np.float32)
    v = np.array([[0.5], [0.4], [0.3]], np.float32)
    fim = np.array([[0.0], [1.0], [0.0]], np.float32)
    adv, ret = gae(r, v, fim, np.array([0.7], np.float32), g, l)
    d2 = 2.0 + g * 0.7 - 0.3
    d1 = 0.0 - 0.4                        # acabou: sem bootstrap
    d0 = 1.0 + g * 0.4 - 0.5
    assert np.allclose(adv[:, 0], [d0 + g * l * d1, d1, d2])
    assert np.allclose(ret, adv + v)


def test_mini_treino_de_ponta_a_ponta(tmp_path):
    tr = Treino(_cfg(envs=4, rollout=32, passos=600, fases="calmo", dificuldade=0.5,
                     aval_cada=256), tmp_path)
    res = tr.rodar()
    assert res["passos"] >= 600 and res["partidas_final"] == 1
    for arq in ("registro.csv", "avaliacoes.csv", "resultado.json", "modelo.pt"):
        assert (tmp_path / arq).exists()
    cab = (tmp_path / "registro.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    for c in ("retorno_medio", "vitorias", "entropia", "kl", "cortada", "perda_v"):
        assert c in cab
    assert len(tr.avaliacoes) >= 3
