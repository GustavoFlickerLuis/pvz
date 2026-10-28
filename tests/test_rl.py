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


# ---------------------------------------------------------------- arquiteturas e currículo (Etapa B)
from pvz.divisao import FASES_TESTE
from rl.dqn import CUR_PARTIDAS, NIVEIS, Curriculo, fases_do_nivel


class TestCurriculo:
    def test_niveis_so_geram_fases_de_treino(self):
        tamanhos = []
        for nivel, (seg, _) in enumerate(NIVEIS):
            fases = fases_do_nivel(nivel)
            tamanhos.append(len(fases))
            assert all(len(f) == seg for f in fases)
            assert all(any(t[:seg] == f for t in FASES_TREINO) for f in fases)   # prefixos do treino
            assert not set(fases) & set(FASES_TESTE)
        assert tamanhos == [3, 9, 21]
        assert fases_do_nivel(2) == list(FASES_TREINO)

    def test_sorteio_e_avaliacao(self):
        cur = Curriculo(Config(curriculo="sim"))
        rng = np.random.default_rng(0)
        for nivel in range(3):
            cur.nivel = nivel
            sorteios = [cur.sortear(rng) for _ in range(5000)]
            for fase, dif, n in sorteios:
                assert fase in fases_do_nivel(n) and dif == NIVEIS[n][1] and n <= nivel
                assert fase not in FASES_TESTE
            atual = np.mean([n == nivel for _, _, n in sorteios])
            assert atual == 1.0 if nivel == 0 else abs(atual - 0.8) < 0.02
            if nivel == 2:   # os anteriores, sorteados por igual
                assert abs(np.mean([n == 0 for _, _, n in sorteios]) - 0.1) < 0.015
            if nivel < 2:
                lista = cur.partidas()   # passa pelo assert de avaliar.partidas
                assert len(lista) == CUR_PARTIDAS
                assert {f for f, _ in lista} == set(fases_do_nivel(nivel))
                assert all(400 <= s < 1000 for _, s in lista)

    def test_quando_sobe(self):
        cur = Curriculo(Config(curriculo="sim"))
        assert cur.registrar(25_000, 0.49) == "" and cur.nivel == 0
        assert cur.registrar(50_000, 0.50) == "vitorias" and (cur.nivel, cur.inicio_nivel) == (1, 50_000)
        assert cur.registrar(175_000, 0.0) == ""
        assert cur.registrar(200_000, 0.0) == "tempo" and cur.nivel == 2 and cur.ultimo
        assert [h["nivel"] for h in cur.historico] == [1, 1, 2, 2]

    def test_config(self):
        with pytest.raises(ValueError):
            Config(arquitetura="cnn2", receita="vencedora")
        with pytest.raises(ValueError):
            Config(curriculo="sim", fases="calmo")
        with pytest.raises(ValueError):
            Config(arquitetura="cnn9")
        Config(arquitetura="cnn3", curriculo="sim")


class TestLacoNovo:
    @pytest.mark.parametrize("arq", ["cnn1d", "cnn2", "cnn3"])
    def test_replay_no_formato_da_arquitetura(self, tmp_path, arq):
        """O replay guarda 25 canais (v2), e o laço com currículo grava as transições certas,
        na dificuldade do nível."""
        tr = _treino(tmp_path, arquitetura=arq, curriculo="sim")
        assert tr.replay.obs.shape[1:] == (25, 5, 9)
        tr.curriculo.nivel = 1
        rng = np.random.default_rng(3)
        k = 0
        for _ in range(4):
            fase, dif, nivel = tr.curriculo.sortear(rng)
            assert tr.jogar_episodio(fase, 11 + k, dif, nivel) == "fim"
            ep = tr.episodios[-1]
            assert (ep["nivel"], ep["dificuldade"]) == (nivel + 1, dif)
            T = ep["decisoes"]
            amb = Ambiente(formato="v2")
            obs = [amb.reset(fase, ep["semente"], dif)]
            for a in tr.replay.acao[k:k + T]:
                obs.append(amb.step(int(a))[0])
            for i in range(T):
                assert np.array_equal(tr.replay.obs[k + i], obs[i].astype(np.float16))
                assert np.array_equal(tr.replay.prox[k + i], obs[min(i + tr.cfg.n, T)].astype(np.float16))
            k += T


# ---------------------------------------------------------------- exploração intencional
import hashlib

from pvz import config as C
from pvz.ambiente import N_CELULAS
from rl.dqn import CUSTOS, mascara_reserva, sol_da_obs


def _acoes_cenario(tmp_path, **kw):
    """4 episódios com atualizações e ε caindo até 0,05 no passo 400."""
    tr = _treino(tmp_path, inicio=300, lote=32, eps_frac=0.004, **kw)
    for s in (1, 2, 3, 4):
        tr.jogar_episodio(FASES_TREINO[s], s)
    return tr


class TestExploracaoIntencional:
    # (GPU, CPU): as contas da atualização mudam com o dispositivo (na cnn1 coincidem)
    REFS = {"cnn1": ("2aacddc64f15088f0636036b5a2a47d9212b5ea8d491cc0c6a365d62bb832832",) * 2,
            "cnn2": ("6e29469e167a86f1af01af885043f9ae3031e8d224e7c1a2491490fa692e359e",
                     "0b53d6382208e23cb4a775a7b4852a270e5f87ba05bd649b499ff02be33330b8")}

    @pytest.mark.parametrize("arq", ["cnn1", "cnn2"])
    def test_a_uniforme_gera_as_mesmas_acoes_de_antes(self, tmp_path, arq):
        """Hashes das ações gravados com o código de antes da opção (commit 9c1756f; os da
        CPU, com o código de antes do k configurável, commit 73279e8)."""
        ref = self.REFS[arq][0 if torch.cuda.is_available() else 1]
        tr = _acoes_cenario(tmp_path, arquitetura=arq)
        assert tr.cfg.exploracao == "uniforme"
        assert hashlib.sha256(tr.replay.acao[:tr.replay.tamanho].tobytes()).hexdigest() == ref

    @pytest.mark.parametrize("arq", ["cnn1", "cnn2"])
    def test_b_toda_acao_executada_e_valida(self, tmp_path, arq, monkeypatch):
        tr = _treino(tmp_path, inicio=300, lote=32, eps_ini=0.6, eps_fim=0.6,
                     exploracao="intencional", arquitetura=arq)
        agir, vistas = tr.agir, []

        def conferir(obs, mascara):
            acao, expl = agir(obs, mascara)
            assert mascara[acao]          # a máscara REAL (amb.mascara()), não a de reserva
            vistas.append(acao)
            return acao, expl
        monkeypatch.setattr(tr, "agir", conferir)
        for s in (1, 2, 3):
            tr.jogar_episodio(FASES_TREINO[s], s)
        assert len(vistas) == tr.passo > 300
        e = tr.exploracao
        assert sum(x["decisoes_com_alvo"] for x in e) > 0
        assert sum(x["alvos_cumpridos"] for x in e) > 0

    def test_c_com_eps_1_os_5_tipos_viram_alvo(self, tmp_path):
        tr = _treino(tmp_path, exploracao="intencional")
        assert tr.epsilon() == 1.0
        for s in range(6):
            tr.jogar_episodio(FASES_TREINO[s], s)
        alvos = [sum(x[f"alvos_{p.nome}"] for x in tr.exploracao) for p in C.PLANTAS]
        assert all(a > 0 for a in alvos), alvos
        assert tr.alvo_tipo is None and tr.espera == 0       # zeram no fim do episódio

    def test_d_reserva(self, tmp_path):
        """Sol 150 e alvo repetidora (200): nenhum plantio; sol 250: só os de custo ≤ 50."""
        assert list(CUSTOS) == [50, 100, 200, 50, 25]
        todas = np.ones(N_ACOES, bool)
        m = mascara_reserva(todas, 150, C.REPETIDORA)
        assert not m[:ESPERAR].any() and m[ESPERAR]
        m = mascara_reserva(todas, 250, C.REPETIDORA)
        for t in range(5):
            assert m[t * N_CELULAS:(t + 1) * N_CELULAS].all() == (CUSTOS[t] <= 50)
            assert m[t * N_CELULAS:(t + 1) * N_CELULAS].any() == (CUSTOS[t] <= 50)
        # e no ator: observação sintética, rede que só quer plantar (Q de esperar baixíssimo)
        for formato, arq in (("v1", "cnn1"), ("v2", "cnn2")):
            tr = _treino(tmp_path / formato, exploracao="intencional", arquitetura=arq)
            tr.ator = lambda x: torch.cat([torch.rand(1, ESPERAR), torch.full((1, 1), -1e9)], 1)
            amb = Ambiente(formato=formato)
            amb.reset(FASES_TREINO[0], 0)
            mascara = np.ones(N_ACOES, bool)
            mascara[C.REPETIDORA * N_CELULAS:(C.REPETIDORA + 1) * N_CELULAS] = False
            for sol, permitidos in ((150, set()), (250, {C.GIRASSOL, C.BATATA, C.MINA})):
                amb.sol = sol
                obs = amb.observar()
                assert sol_da_obs(obs, formato) == sol
                escolhidos = set()
                for _ in range(20):
                    tr.alvo_tipo, tr.espera = C.REPETIDORA, 0
                    acao, expl = tr.agir(obs, mascara)
                    assert not expl and tr.alvo_tipo == C.REPETIDORA
                    escolhidos.add("esperar" if acao == ESPERAR else acao // N_CELULAS)
                assert escolhidos == ({"esperar"} if not permitidos else permitidos), (formato, sol)

    def test_e_timeout_libera_o_alvo(self, tmp_path):
        tr = _treino(tmp_path, exploracao="intencional")
        amb = Ambiente()
        amb.reset(FASES_TREINO[0], 0)
        amb.sol = 0
        obs, mascara = amb.observar(), amb.mascara()
        tr.alvo_tipo, tr.espera = C.REPETIDORA, 0
        for i in range(tr.cfg.expl_espera_max):
            assert tr.alvo_tipo == C.REPETIDORA
            acao, expl = tr.agir(obs, mascara)
            assert acao == ESPERAR and not expl
        assert tr.alvo_tipo is None and tr.contagem["alvos_timeout"] == 1
        assert tr.contagem["decisoes_com_alvo"] == tr.cfg.expl_espera_max

    def test_alvo_disponivel_planta_e_libera(self, tmp_path):
        tr = _treino(tmp_path, exploracao="intencional")
        amb = Ambiente()
        amb.reset(FASES_TREINO[0], 0)
        amb.sol = 300
        tr.alvo_tipo = C.REPETIDORA
        acao, expl = tr.agir(amb.observar(), amb.mascara())
        assert expl and acao // N_CELULAS == C.REPETIDORA and amb.mascara()[acao]
        assert tr.alvo_tipo is None and tr.contagem["alvos_cumpridos"] == 1

    def test_ids(self):
        from rl.dqn import id_padrao
        from rl.rodar import ler_fila
        assert id_padrao("cnn1", "nao", 0) == "cnn1-s0"
        assert id_padrao("cnn2", "sim", 1) == "cnn2-cur-s1"
        assert id_padrao("cnn1", "nao", 2, exploracao="intencional") == "cnn1-s2-int"
        assert id_padrao("cnn1", "nao", 0, "vencedora") == "vencedora-s0"
        assert ler_fila(["cnn1:nao:0", "cnn2:sim:1:intencional"]) == \
            [("cnn1", "nao", 0, "uniforme"), ("cnn2", "sim", 1, "intencional")]
        with pytest.raises(SystemExit):
            ler_fila(["cnn1:nao:0:outra"])


class TestK:
    def test_k_padrao_e_ids(self):
        from rl.dqn import id_padrao
        assert Config().k_ticks == 10
        assert id_padrao("cnn1", "nao", 0) == "cnn1-s0"
        assert id_padrao("cnn1", "nao", 0, k_ticks=20) == "cnn1-s0-k20"
        assert id_padrao("cnn1", "nao", 1, exploracao="intencional", k_ticks=20) == "cnn1-s1-k20-int"

    def test_k20_treina_e_avalia_com_k20(self, tmp_path):
        """Com k = 20, o episódio de treino e a avaliação rodam 20 ticks por decisão: a
        mesma partida tem a metade das decisões da de k = 10 (com "esperar" sempre)."""
        from rl import avaliar as av
        tr = _treino(tmp_path, k_ticks=20, gamma=0.994, inicio=50, lote=16)
        tr.jogar_episodio(FASES_TREINO[0], 5)
        assert tr.atualizacoes > 0
        decisoes = {}
        for k in (10, 20):
            amb, fim, n = Ambiente(k=k), False, 0
            amb.reset(FASES_TREINO[0], 5)
            while not fim:
                _, _, fim = amb.step(ESPERAR)
                n += 1
            decisoes[k] = n
        assert decisoes[20] == pytest.approx(decisoes[10] / 2, abs=1)
        espera = lambda x: torch.cat([torch.zeros(len(x), ESPERAR), torch.ones(len(x), 1)], 1)
        espera.eval = espera.train = lambda: None
        import desafio3.modelo as dm
        orig = dm.formato_do_modelo
        av.formato_do_modelo = lambda m: "v1"
        try:
            r10 = av.avaliar_rede(espera, [(FASES_TREINO[0], 5)], "cpu")
            r20 = av.avaliar_rede(espera, [(FASES_TREINO[0], 5)], "cpu", k=20)
        finally:
            av.formato_do_modelo = orig
        assert (r10["decisoes"], r20["decisoes"]) == (decisoes[10], decisoes[20])

