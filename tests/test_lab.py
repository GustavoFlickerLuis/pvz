"""Testes do lab (LAB.md, Fase 0): formas e ordem da saída, parâmetros, cobertura das
colunas, C = cnn1d na inicialização, sensibilidade a 8 colunas, invariância à permutação
dos tokens, reprodução da ppo-fases-s0 (CUDA) e retomada exata do PPO."""
import csv
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from desafio3.modelo import (ARQUITETURAS, CNN1D, DILATACOES, AtencaoDepois, CNN1DDilatada,
                             Transformer, TransformerD128, mascara_da_entrada)
from pvz.ambiente import ESPERAR, Ambiente, codificar_acao
from pvz.divisao import FASES_TREINO
from rl.ppo import Config, Politica, PoliticaArq, Treino

NOVOS = ("atencao_antes", "atencao_depois", "transformer", "transformer_d128", "cnn1d_dilatada")
RAIZ = Path(__file__).resolve().parent.parent


def _obs(n=4, semente=0):
    obs = []
    for i in range(n):
        a = Ambiente(k=20, formato="v2")
        o = a.reset(FASES_TREINO[i], semente + i)
        for _ in range(15 + 5 * i):
            m = a.mascara()
            o, _, fim = a.step(int(np.flatnonzero(m)[i % m.sum()]))
            if fim:
                break
        obs.append(o)
    return torch.from_numpy(np.stack(obs))


def _x_cheio(n=2, semente=0):
    """Entrada aleatória com todas as ações válidas (canais 20–24 = 1)."""
    g = torch.Generator().manual_seed(semente)
    x = torch.rand(n, 25, 5, 9, generator=g)
    x[:, 20:] = 1.0
    return x


@pytest.mark.parametrize("nome", NOVOS)
def test_formas_e_ordem_da_saida(nome):
    torch.manual_seed(0)
    rede = ARQUITETURAS[nome][0]()
    x = _obs()
    mapa, esperar, v = rede.partes(x)
    assert mapa.shape == (4, 5, 5, 9) and esperar.shape == (4, 1) and v.shape == (4, 1)
    q = rede(x)
    assert q.shape == (4, 226)
    # ordem de codificar_acao: o índice (t, f, c) da saída é o mapa[t, f, c]; 225 = esperar
    a = torch.cat([mapa.flatten(1), esperar], 1)
    for t, f, c in [(0, 0, 0), (2, 3, 7), (4, 4, 8), (1, 2, 0)]:
        assert torch.equal(a[:, codificar_acao(t, f, c)], mapa[:, t, f, c])
    assert torch.equal(a[:, ESPERAR], esperar[:, 0])
    # na política: logits = A (antes da média do dueling), inválidas = −1e9, V separado
    pol = PoliticaArq(nome)
    pol.rede = rede
    logits, vv = pol(x)
    m = mascara_da_entrada(x)
    assert torch.equal(logits[m], a[m]) and torch.all(logits[~m] == -1e9)
    assert torch.equal(vv, v[:, 0])


def test_dilatada_tem_os_mesmos_parametros_da_cnn1d():
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(CNN1DDilatada()) == n(CNN1D()) == 88_775
    assert [tuple(p.shape) for p in CNN1DDilatada().parameters()] == \
           [tuple(p.shape) for p in CNN1D().parameters()]


def test_dilatada_cobre_as_9_colunas_sem_buracos():
    # colunas alcançáveis da saída para a entrada, só por posições dentro do tabuleiro
    for c in range(9):
        alc = {c}
        for d in reversed(DILATACOES):
            alc = {j + e for j in alc for e in (-d, 0, d) if 0 <= j + e < 9}
        assert alc == set(range(9)), c
    # empírico: cada célula do mapa depende de todas as colunas da própria faixa
    torch.manual_seed(0)
    rede = CNN1DDilatada()
    with torch.no_grad():
        for p in rede.parameters():
            p.abs_()              # pesos positivos e entrada positiva: nenhuma ReLU zera
    x = (torch.rand(1, 25, 5, 9) + 0.1).requires_grad_()
    mapa = rede.partes(x)[0]
    for f, c in itertools.product(range(5), range(9)):
        g, = torch.autograd.grad(mapa[0, 0, f, c], x, retain_graph=True)
        assert (g[0, :, f, :].abs().sum(0) > 0).all(), (f, c)


def test_atencao_depois_comeca_igual_a_cnn1d():
    torch.manual_seed(0)
    ref = CNN1D()
    c = AtencaoDepois()
    faltam = c.load_state_dict(ref.state_dict(), strict=False)
    assert not faltam.unexpected_keys
    assert all(k.startswith(("pos.", "atencao.")) for k in faltam.missing_keys)
    x = _obs()
    m = mascara_da_entrada(x)
    assert torch.equal(c(x, m), ref(x, m))
    pol_ref, pol_c = Politica(), PoliticaArq("atencao_depois")
    pol_ref.rede, pol_c.rede = ref, c
    for a, b in zip(pol_ref(x), pol_c(x)):
        assert torch.equal(a, b)


def _sensivel_a_8_colunas(rede, x):
    """O logit de plantio em (f, 0) muda quando só a célula (f, 8) muda?"""
    x2 = x.clone()
    x2[:, :20, 2, 8] += torch.rand(len(x), 20, generator=torch.Generator().manual_seed(1)) + 0.5
    with torch.no_grad():
        a, b = rede.partes(x)[0], rede.partes(x2)[0]
    return float((a[:, :, 2, 0] - b[:, :, 2, 0]).abs().max())


@pytest.mark.parametrize("nome", NOVOS)
def test_novos_sao_sensiveis_a_8_colunas(nome):
    torch.manual_seed(0)
    rede = ARQUITETURAS[nome][0]()
    if nome == "atencao_depois":     # Wo = 0 na inicialização (= cnn1d); dá um valor qualquer
        torch.nn.init.normal_(rede.atencao.mha.out_proj.weight, std=0.1)
    assert _sensivel_a_8_colunas(rede, _x_cheio()) > 1e-5


def test_cnn1d_nao_e_sensivel_a_8_colunas():
    torch.manual_seed(0)
    rede = CNN1D()
    rede.partes = lambda x: (rede.cabeca_q(rede.tronco(x)[-1]),)
    assert _sensivel_a_8_colunas(rede, _x_cheio()) == 0.0


@pytest.mark.parametrize("classe", [Transformer, TransformerD128])
def test_transformer_invariante_a_ordem_dos_tokens(classe):
    torch.manual_seed(0)
    rede = classe().eval()
    x = _obs()
    with torch.no_grad():
        t = rede.tokens(x)                       # conteúdo + posição de cada célula
        perm = torch.randperm(t.shape[1], generator=torch.Generator().manual_seed(3))
        cel, cls = rede.codificar(t)
        cel_p, cls_p = rede.codificar(t[:, perm])
    assert torch.allclose(cel[:, perm], cel_p, atol=1e-5)
    assert torch.allclose(cls, cls_p, atol=1e-5)
    # e sem a posição o token não sabe onde está: trocar duas células troca as saídas
    p1, e1, v1 = rede.saidas(cel, cls)
    p2, e2, v2 = rede.saidas(cel_p, cls_p)
    assert torch.allclose(p1[:, perm], p2, atol=1e-5) and torch.allclose(e1, e2, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a ppo-fases-s0 rodou em CUDA")
def test_receita_atual_reproduz_a_ppo_fases_s0(tmp_path):
    """As 4 primeiras atualizações da ppo-fases-s0 (CUDA, cnn1d, receita atual), bit a bit."""
    ref = list(csv.DictReader(open(RAIZ / "rl/logs/ppo-fases-s0/registro.csv", encoding="utf-8")))
    tr = Treino(Config(semente=0, passos=2_000_000, aval_cada=250_000), tmp_path)
    for u in range(4):
        X, A, LP, adv, ret, fins = tr.coletar()
        s = tr.atualizar(X, A, LP, adv, ret)
        assert len(fins) == int(ref[u]["episodios"])
        for k in ("entropia", "kl", "cortada", "perda_v", "perda_pi"):
            assert s[k] == float(ref[u][k]), (u, k, s[k], ref[u][k])


class _ParaDepois(Treino):
    """Cria o arquivo de parada depois de n atualizações."""
    n = 3

    def atualizar(self, *a):
        s = super().atualizar(*a)
        if len(self.linhas) + 1 == self.n:
            Path(self.cfg.arquivo_parar).write_text("")
        return s


@pytest.mark.parametrize("arquitetura", ["cnn1d", "transformer"])
def test_parada_e_retomada_exatas(tmp_path, arquitetura):
    base = dict(envs=4, rollout=32, passos=4 * 32 * 6, minilote=64, aval_cada=10 ** 9,
                aval_sementes=(300, 301), aval_final_sementes=(300, 301),
                arquitetura=arquitetura, continuar=True)
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    parar = tmp_path / "PARAR"
    ra = Treino(Config(**base), tmp_path / "a").rodar()
    tb = _ParaDepois(Config(**base, arquivo_parar=str(parar)), tmp_path / "b")
    r1 = tb.rodar()
    assert r1["motivo"] == "parada" and (tmp_path / "b" / "ckpt.pt").exists()
    assert len(tb.linhas) == 3
    parar.unlink()
    tb2 = Treino(Config(**base, arquivo_parar=str(parar)), tmp_path / "b")
    rb = tb2.rodar()
    assert rb["passos"] == ra["passos"] and rb["vitorias_final"] == ra["vitorias_final"]
    la = (tmp_path / "a" / "registro.csv").read_text(encoding="utf-8").splitlines()
    lb = (tmp_path / "b" / "registro.csv").read_text(encoding="utf-8").splitlines()
    corta = lambda l: [x.rsplit(",", 1)[0] for x in l]     # tira passos_por_s
    assert corta(la) == corta(lb)
    wa = torch.load(tmp_path / "a" / "modelo.pt")
    wb = torch.load(tmp_path / "b" / "modelo.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
