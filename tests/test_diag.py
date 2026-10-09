"""Testes do diagnóstico (rl/diag): nota(), observação v2 reconstruída e n passos off-line."""
import numpy as np
import pytest
import torch

from pvz.agentes import Econ
from pvz.ambiente import Ambiente
from pvz.divisao import FASES_TREINO
from rl.diag import prova
from rl.diag.comum import para_v2
from rl.diag.td import indices_n_passos


def _prova_sintetica():
    """3 estados, candidatas com Q̂ conhecido e EP pequeno; a do Econ5 é a 2ª."""
    q = np.full((3, prova.MAX_CAND, 2), np.nan, np.float32)
    cand = np.full((3, prova.MAX_CAND), -1, np.int16)
    for s, valores in enumerate(([0.1, 0.5, 0.3], [0.9, 0.2, 0.6, 0.0], [0.4, 0.41, 0.8])):
        q[s, :len(valores), 0] = valores
        q[s, :len(valores), 1] = 0.001
        cand[s, :len(valores)] = np.arange(len(valores)) + 10 * s
    return {"q": q, "cand": cand, "n_cand": np.array([3, 4, 3]), "econ5": np.array([1, 1, 1])}


class TestNota:
    def test_rede_que_devolve_o_proprio_q_da_100(self):
        p = _prova_sintetica()
        r = prova.nota(p["q"][:, :, 0], p)
        assert r["nota"] == 100.0 and r["arrependimento"] == pytest.approx(0)
        # pares claros: |Δ| > 0,02 → 3 + 6 + 2 (0,40 × 0,41 não é claro)
        assert r["pares"] == 11 and r["pares_i"] == 2 + 3 + 1

    def test_constante_da_50(self):
        p = _prova_sintetica()
        r = prova.nota(np.where(p["cand"] >= 0, 0.0, np.nan), p)
        assert r["nota"] == 50.0 and r["nota_i"] == 50.0 and r["nota_ii"] == 50.0
        # arrependimento com empate total = melhor − média
        esperado = np.mean([0.5 - 0.3, 0.9 - 0.425, 0.8 - (0.4 + 0.41 + 0.8) / 3])
        assert r["arrependimento"] == pytest.approx(esperado, abs=1e-6)

    def test_ordem_invertida_da_0(self):
        p = _prova_sintetica()
        assert prova.nota(-p["q"][:, :, 0], p)["nota"] == 0.0

    def test_arrependimento_do_econ5(self):
        p = _prova_sintetica()
        r = prova.nota(p["q"][:, :, 0], p)
        assert r["arrependimento_econ5"] == pytest.approx(np.mean([0.0, 0.7, 0.8 - 0.41]), abs=1e-6)


def test_v2_reconstruida_igual_a_do_ambiente():
    amb, ag, fim = Ambiente(k=20, formato="v2"), Econ(5), False
    amb.reset(FASES_TREINO[2], 1)
    for _ in range(40):
        v1, v2, m = amb.observar("v1"), amb.observar("v2"), amb.mascara()
        sol = round(float(v1[11, 0, 0]) * 1000)
        x = para_v2(v1[None].astype(np.float16), np.array([sol], np.int16), m[None])[0]
        assert torch.allclose(x, torch.from_numpy(v2), atol=2e-3)
        _, _, fim = amb.step(ag.agir(amb))
        if fim:
            break


def test_n_passos_offline():
    g = 0.9
    d = {"ep": np.array([0, 0, 0, 0, 1, 1]), "pos": np.array([0, 1, 2, 3, 0, 1]),
         "r": np.array([1., 2., 3., 4., 5., 6.]), "acao": np.zeros(6)}
    R, prox, tem = indices_n_passos(d, 2, g)
    assert np.allclose(R, [1 + g * 2, 2 + g * 3, 3 + g * 4, 4, 5 + g * 6, 6])
    assert list(tem) == [True, True, False, False, False, False]
    assert list(prox[:2]) == [2, 3]
