"""Testes do Double DQN (rl/dqn.py): n passos, máscara no alvo, Double DQN e exploração."""
import math

import numpy as np
import pytest
import torch

from pvz.ambiente import ESPERAR, N_ACOES
from rl.dqn import (RECEITAS, Config, NPassos, Replay, alvo_double, argmax_mascarado,
                    explorar)

G = 0.9


def _rodar(n, recompensas, fim_no_ultimo=True):
    np_ = NPassos(n, G)
    saida = []
    for i, r in enumerate(recompensas):
        fim = fim_no_ultimo and i == len(recompensas) - 1
        saida += np_.adicionar(f"s{i}", i, r, f"s{i + 1}", f"m{i + 1}", fim)
    return saida


class TestNPassos:
    def test_retorno_de_n_passos(self):
        rs = [1.0, 2.0, 3.0, 4.0, 5.0]
        saida = _rodar(3, rs, fim_no_ultimo=False)
        assert [t[1] for t in saida] == [0, 1, 2]           # sai quando a fila tem 3
        o, a, R, g, prox, m, fim = saida[0]
        assert (o, prox, m, fim) == ("s0", "s3", "m3", False)
        assert R == pytest.approx(1 + G * 2 + G ** 2 * 3)
        assert g == pytest.approx(G ** 3)
        assert saida[2][2] == pytest.approx(3 + G * 4 + G ** 2 * 5)

    def test_corta_no_terminal(self):
        rs = [1.0, 2.0, 3.0, 4.0, 5.0]
        saida = _rodar(3, rs)
        assert [t[1] for t in saida] == [0, 1, 2, 3, 4]     # o fim esvazia a fila
        for o, a, R, g, prox, m, fim in saida[2:]:
            assert fim and prox == "s5"
            assert R == pytest.approx(sum(G ** i * r for i, r in enumerate(rs[a:])))
        assert not saida[1][6]                              # a 2ª saiu antes do fim, com bootstrap

    def test_episodio_mais_curto_que_n(self):
        saida = _rodar(10, [0.5, -1.0])
        assert len(saida) == 2 and all(t[6] for t in saida)
        assert saida[0][2] == pytest.approx(0.5 - G)

    def test_n_igual_a_1(self):
        saida = _rodar(1, [1.0, 2.0])
        assert [(t[2], t[3], t[6]) for t in saida] == [(1.0, G, False), (2.0, G, True)]   # no fim o desconto não é usado


class TestAlvo:
    def test_mascara_no_argmax(self):
        q = torch.randn(64, N_ACOES)
        q[:, 7] = 100.0                                     # maior Q, mas inválida
        m = torch.rand(64, N_ACOES) < 0.3
        m[:, 7] = False
        m[:, ESPERAR] = True
        a = argmax_mascarado(q, m)
        assert m[torch.arange(64), a].all()
        assert (a != 7).all()

    def test_double_dqn(self):
        """argmax pela online, valor pela alvo; sem bootstrap no terminal."""
        on = torch.zeros(3, N_ACOES)
        al = torch.zeros(3, N_ACOES)
        on[:, 5] = 1.0                                      # online escolhe 5
        al[:, 5] = 0.25                                     # valor da alvo em 5
        al[:, 9] = 9.0                                      # a alvo prefere 9: não pode ser usada
        on[:, 9] = 50.0                                     # e 9 é inválida para a online
        m = torch.zeros(3, N_ACOES, dtype=torch.bool)
        m[:, [5, ESPERAR]] = True
        R = torch.tensor([1.0, 2.0, 3.0])
        d = torch.tensor([0.5, 0.5, 0.5])
        fim = torch.tensor([False, True, False])
        y = alvo_double(on, al, m, R, d, fim)
        assert torch.allclose(y, torch.tensor([1.0 + 0.5 * 0.25, 2.0, 3.0 + 0.5 * 0.25]))

    def test_terminal_ignora_q_infinito(self):
        on, al = torch.zeros(1, N_ACOES), torch.full((1, N_ACOES), math.inf)
        m = torch.ones(1, N_ACOES, dtype=torch.bool)
        y = alvo_double(on, al, m, torch.tensor([1.0]), torch.tensor([0.9]), torch.tensor([True]))
        assert y.item() == 1.0


class TestExploracao:
    def test_respeita_mascara(self):
        rng = np.random.default_rng(0)
        m = np.zeros(N_ACOES, bool)
        m[[3, 50, 120, ESPERAR]] = True
        acoes = [explorar(m, rng, 0.5) for _ in range(4000)]
        assert set(acoes) == {3, 50, 120, ESPERAR}
        assert abs(np.mean(np.array(acoes) == ESPERAR) - 0.5) < 0.03
        planta = [a for a in acoes if a != ESPERAR]
        assert all(abs(planta.count(a) / len(planta) - 1 / 3) < 0.04 for a in (3, 50, 120))

    def test_sem_planta_valida_espera(self):
        rng = np.random.default_rng(0)
        m = np.zeros(N_ACOES, bool)
        m[ESPERAR] = True
        assert {explorar(m, rng, 0.5) for _ in range(100)} == {ESPERAR}


class TestConfig:
    def test_receitas_do_proxy(self):
        assert RECEITAS["baseline"] == dict(otimizador="adam", lr=1e-4, agenda="constante")
        assert RECEITAS["vencedora"] == dict(otimizador="sgd", lr=3e-2, agenda="cosseno")
        assert RECEITAS["muon"]["lr_adamw"] == 3e-4

    def test_atualizacoes(self):
        c = Config(passos=100, inicio=10, freq_atualizacao=4)
        assert c.atualizacoes() == sum(1 for p in range(1, 101) if p >= 10 and p % 4 == 0)
        assert c.proxy().passos == c.atualizacoes()
        assert c.proxy().momento == 0.9 and c.proxy().clip == 10.0

    def test_replay_circular(self):
        r = Replay(3)
        o = np.zeros((20, 5, 9), np.float32)
        for i in range(5):
            r.guardar((o + i, i, float(i), 0.5, o, np.ones(N_ACOES, bool), False))
        assert r.tamanho == 3 and r.pos == 2
        assert sorted(r.acao.tolist()) == [2, 3, 4]
