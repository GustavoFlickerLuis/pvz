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


# ---------------------------------------------------------------- revisão (Etapa A): o laço real
from pvz.ambiente import Ambiente
from pvz.divisao import FASES_TREINO
from rl.dqn import Treino


def _treino(tmp_path, **kw):
    cfg = Config(**{"passos": 100_000, "inicio": 10 ** 6, "aval_cada": 10 ** 9, "replay": 5_000,
                    "n": 10, **kw})
    return Treino(cfg, tmp_path)


def _resimular(fase, semente, acoes, dificuldade=1.0):
    """Joga de novo as ações gravadas; devolve obs, máscaras e recompensas por passo."""
    amb = Ambiente()
    obs, mascaras, rs = [amb.reset(fase, semente, dificuldade)], [amb.mascara()], []
    for a in acoes:
        o, r, fim = amb.step(int(a))
        obs.append(o)
        mascaras.append(amb.mascara())
        rs.append(r)
    assert fim
    return obs, mascaras, rs


class TestLacoReal:
    def test_replay_confere_com_a_partida(self, tmp_path):
        """Cada transição do replay, gravada pelo laço de treino de verdade, confere com a
        partida re-simulada: s, a, R de n passos, γⁿ, s_{t+n}, MÁSCARA de s_{t+n}, fim e
        nenhuma transição atravessando dois episódios."""
        tr = _treino(tmp_path)
        n, g = tr.cfg.n, tr.cfg.gamma
        for semente in (1, 2, 3):
            assert tr.jogar_episodio(FASES_TREINO[semente], semente) == "fim"
        k = 0
        for ep in tr.episodios:
            T = ep["decisoes"]
            obs, mascaras, rs = _resimular(tuple(ep["fase"].split("+")), ep["semente"],
                                           tr.replay.acao[k:k + T])
            for i in range(T):
                j = min(i + n, T)
                fim = i + n >= T
                assert np.array_equal(tr.replay.obs[k + i], obs[i].astype(np.float16))
                assert np.array_equal(tr.replay.prox[k + i], obs[j].astype(np.float16))
                assert np.array_equal(tr.replay.mascara[k + i], mascaras[j])
                assert tr.replay.fim[k + i] == fim
                R = sum(g ** (l - i) * rs[l] for l in range(i, j))
                assert tr.replay.R[k + i] == pytest.approx(R, abs=1e-6)
                if not fim:
                    assert tr.replay.desconto[k + i] == pytest.approx(g ** n)
            assert rs[-1] > 0 if ep["vitoria"] else rs[-1] < 0
            k += T
        assert k == tr.replay.tamanho == tr.passo

    def test_exploracao_no_laco_so_usa_acoes_validas(self, tmp_path):
        """Com ε = 1 o laço só explora; o step levanta erro em ação inválida, e a fração de
        "esperar" fica perto de p_esperar (mais as vezes em que não há planta válida)."""
        tr = _treino(tmp_path)
        assert tr.epsilon() == 1.0
        tr.jogar_episodio(FASES_TREINO[0], 7)
        acoes = tr.replay.acao[:tr.replay.tamanho]
        assert np.mean(acoes == ESPERAR) >= 0.5 - 0.1

    def test_ator_igual_a_online_depois_de_cada_atualizacao(self, tmp_path):
        tr = _treino(tmp_path, inicio=50, freq_atualizacao=4, lote=32)
        tr.jogar_episodio(FASES_TREINO[0], 5)
        assert tr.atualizacoes > 0
        for a, p in zip(tr.ator.state_dict().values(), tr.online.state_dict().values()):
            assert torch.equal(a, p.cpu())
        tr.atualizar()
        for a, p in zip(tr.ator.state_dict().values(), tr.online.state_dict().values()):
            assert torch.equal(a, p.cpu())


class TestTruncagem:
    def test_max_ticks_e_derrota_terminal(self):
        """A trava MAX_TICKS é derrota (−1) e fim; o n passos não faz bootstrap depois dela."""
        from pvz import config as C
        from pvz.hordas import fase_manual
        amb = Ambiente(max_ticks=55)
        amb.reset(fase_manual([(10_000, C.COMUM, 0)]))
        np_ = NPassos(10, G)
        saida, fim, passos = [], False, 0
        while not fim:
            o, r, fim = amb.step(ESPERAR)
            saida += np_.adicionar(None, ESPERAR, r, o, amb.mascara(), fim)
            passos += 1
        assert passos == 6 and r == -1.0
        assert len(saida) == 6 and all(t[6] for t in saida)
        assert saida[0][2] == pytest.approx(-(G ** 5))
