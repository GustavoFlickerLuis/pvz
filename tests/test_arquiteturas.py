"""Testes das arquiteturas do rl/ (desafio3/modelo.py): registry, índice da saída, cnn1 bit a
bit, cabeça dueling com máscara, bloco de contexto e a máscara tirada da entrada."""
import hashlib

import numpy as np
import pytest
import torch

from desafio3.modelo import (ARQUITETURAS, CNN, CNN1D, CNN2, CNN3, contexto, criar, formato,
                             formato_do_modelo, mascara_da_entrada)
from pvz import config as C
from pvz.agentes import Econ
from pvz.ambiente import ESPERAR, FORMATOS, N_ACOES, N_PLANTAS, Ambiente, codificar_acao
from pvz.divisao import FASES_TREINO

NOVAS = ("cnn1d", "cnn2", "cnn3")


def _entrada(nome, b=6, semente=1, p_valida=0.4):
    """Entrada aleatória no formato da arquitetura; no v2, canais 20–24 de 0/1 de verdade."""
    g = torch.Generator().manual_seed(semente)
    x = torch.rand(b, FORMATOS[formato(nome)], C.FAIXAS, C.COLUNAS, generator=g)
    if formato(nome) == "v2":
        x[:, 20:] = (torch.rand(b, 5, C.FAIXAS, C.COLUNAS, generator=g) < p_valida).float()
    return x


@pytest.fixture(autouse=True)
def _uma_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


class TestRegistry:
    def test_nomes_classes_e_formatos(self):
        # lab: os braços novos vêm depois das quatro de sempre (tests/test_lab.py)
        assert list(ARQUITETURAS)[:4] == ["cnn1", "cnn1d", "cnn2", "cnn3"]
        assert [type(criar(n)) for n in list(ARQUITETURAS)[:4]] == [CNN, CNN1D, CNN2, CNN3]
        assert [formato(n) for n in ARQUITETURAS] == ["v1"] + ["v2"] * (len(ARQUITETURAS) - 1)
        assert all(formato_do_modelo(criar(n)) == formato(n) for n in ARQUITETURAS)

    @pytest.mark.parametrize("nome", list(ARQUITETURAS))
    def test_saida_226(self, nome):
        q = criar(nome)(_entrada(nome))
        assert q.shape == (6, N_ACOES) and torch.isfinite(q).all()

    @pytest.mark.parametrize("nome", list(ARQUITETURAS))
    def test_camadas_cobrem_todos_os_parametros(self, nome):
        """O Registro mede por camada: as CAMADAS declaradas têm de cobrir a rede toda."""
        m = criar(nome)
        camadas = getattr(m, "CAMADAS", ("conv1", "conv2", "conv3", "cabeca_q", "cabeca_esperar"))
        n = sum(p.numel() for c in camadas for p in getattr(m, c).parameters())
        assert n == sum(p.numel() for p in m.parameters())

    def test_cnn3_tem_8_convs_3x3_de_64_em_4_blocos(self):
        m = CNN3()
        blocos = [getattr(m, f"bloco{i}{l}") for i in range(1, 5) for l in "ab"]
        assert len(blocos) == 8
        assert all(c.kernel_size == (3, 3) and c.in_channels == c.out_channels == 64 for c in blocos)
        assert m.entrada.in_channels == 25 and m.entrada.out_channels == 64


class TestCNN1BitABit:
    def test_pesos_e_saida_iguais_aos_de_ontem(self):
        """Hashes gravados com o desafio3/modelo.py de antes desta mudança (commit 5c21f56)."""
        torch.manual_seed(0)
        m = criar("cnn1")
        x = torch.rand(4, 20, 5, 9, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            y = m(x)
        pesos = hashlib.sha256(b"".join(p.numpy().tobytes() for p in m.state_dict().values()))
        assert pesos.hexdigest() == "b4269ef0dce667c5d4334dc18c8e6b6b7d13285b5528d9706bcb665d2fe22363"
        assert hashlib.sha256(y.numpy().tobytes()).hexdigest() == \
            "da81716b7e7d98853a26fedeea993039d628928e03f16257db8e21fcaf641b3e"


def _partes(m, x):
    """V, mapa A (B, 5, 5, 9), A(esperar) e máscara, calculados à mão pelas camadas."""
    h = m.tronco(x)[-1]
    g = h.mean(dim=(2, 3))
    return m.cabeca_v(g)[:, 0], m.cabeca_q(h), m.cabeca_esperar(g)[:, 0], mascara_da_entrada(x)


@pytest.mark.parametrize("nome", NOVAS)
class TestDueling:
    def test_indice_da_acao_e_a_celula_do_mapa(self, nome):
        """Q(s, a) = V + A(a) − média das válidas, com A(planta p em (f, c)) = mapa[p, f, c]."""
        torch.manual_seed(0)
        m, x = criar(nome), _entrada(nome)
        with torch.no_grad():
            q = m(x)
            v, mapa, esp, mask = _partes(m, x)
        a = torch.zeros(len(x), N_ACOES)
        for p in range(N_PLANTAS):
            for f in range(C.FAIXAS):
                for c in range(C.COLUNAS):
                    a[:, codificar_acao(p, f, c)] = mapa[:, p, f, c]
        a[:, ESPERAR] = esp
        media = torch.stack([a[i][mask[i]].mean() for i in range(len(x))])
        assert torch.allclose(q, v[:, None] + a - media[:, None], atol=1e-5)

    def test_media_das_validas_e_v(self, nome):
        torch.manual_seed(0)
        m, x = criar(nome), _entrada(nome)
        with torch.no_grad():
            q = m(x)
            v, *_, mask = _partes(m, x)
        for i in range(len(x)):
            assert q[i][mask[i]].mean() == pytest.approx(float(v[i]), abs=1e-5)
            assert q[i].mean() != pytest.approx(float(v[i]), abs=1e-4)   # a média de todas não é

    def test_media_ignora_as_invalidas(self, nome):
        """Mudar A só nas ações inválidas não muda o Q de nenhuma ação válida."""
        torch.manual_seed(0)
        m, x = criar(nome), _entrada(nome)
        mask_mapa = x[:, 20:] > 0.5
        with torch.no_grad():
            q0 = m(x)
            gancho = m.cabeca_q.register_forward_hook(
                lambda mod, ent, saida: saida + 1000.0 * torch.randn_like(saida) * ~mask_mapa)
            q1 = m(x)
            gancho.remove()
        valida = mascara_da_entrada(x)
        assert torch.allclose(q0[valida], q1[valida], atol=1e-3)
        assert not torch.allclose(q0[~valida], q1[~valida], atol=1.0)

    def test_mascara_explicita_igual_a_da_entrada(self, nome):
        torch.manual_seed(0)
        m, x = criar(nome), _entrada(nome)
        with torch.no_grad():
            assert torch.equal(m(x), m(x, mascara_da_entrada(x)))

    def test_mascara_da_entrada_e_a_do_ambiente(self, nome):
        """Em estados reais (Econ5), os canais 20–24 + esperar reproduzem amb.mascara()."""
        amb, ag = Ambiente(formato="v2"), Econ(5)
        obs, fim, n = amb.reset(FASES_TREINO[0], 3), False, 0
        while not fim:
            m = mascara_da_entrada(torch.from_numpy(obs)[None])[0].numpy()
            assert np.array_equal(m, amb.mascara())
            obs, _, fim = amb.step(ag.agir(amb))
            n += 1
        assert n > 50


class TestContexto:
    def test_a_mao_num_tabuleiro_pequeno(self):
        """1 canal, 2 faixas × 4 colunas."""
        h = torch.tensor([[[[3., 1., 4., 2.],
                            [0., 5., 1., 1.]]]])
        z = contexto(h)
        assert z.shape == (1, 7, 2, 4)
        h_, faixa_media, faixa_max, direita, esquerda, glob_media, glob_max = z[0]
        assert torch.equal(h_, h[0, 0])
        assert torch.allclose(faixa_media, torch.tensor([[2.5] * 4, [1.75] * 4]))
        assert torch.equal(faixa_max, torch.tensor([[4.] * 4, [5.] * 4]))
        # máximo das colunas ≥ c (à direita, de onde vêm os zumbis) e ≤ c (à esquerda)
        assert torch.equal(direita, torch.tensor([[4., 4., 4., 2.], [5., 5., 1., 1.]]))
        assert torch.equal(esquerda, torch.tensor([[3., 3., 4., 4.], [0., 5., 5., 5.]]))
        assert torch.allclose(glob_media, torch.full((2, 4), 17 / 8))
        assert torch.equal(glob_max, torch.full((2, 4), 5.))

    def test_max_acumulado_igual_ao_cummax(self):
        from desafio3.modelo import _max_acumulado
        h = torch.randn(4, 8, 5, 9)
        assert torch.equal(_max_acumulado(h), torch.cummax(h, dim=3).values)

    def test_canais_separados(self):
        """Cada canal tem o seu contexto: o bloco não mistura canais antes da conv 1×1."""
        h = torch.rand(2, 3, 5, 9)
        z = contexto(h)
        for k in range(3):
            assert torch.equal(contexto(h[:, k:k + 1]), z[:, k::3])

    def test_cnn2_ve_a_faixa_inteira(self):
        """Com o contexto, uma mudança na coluna 0 chega ao mapa A da coluna 8 da mesma faixa
        (as 3 convs 3×3 sozinhas só alcançam 3 colunas; a cnn1d não alcança)."""
        torch.manual_seed(0)
        for nome, alcanca in (("cnn2", True), ("cnn1d", False), ("cnn3", True)):
            m, x = criar(nome), _entrada(nome, b=1)
            x2 = x.clone()
            x2[0, :20, 2, 0] += 1.0
            with torch.no_grad():
                d = (m.cabeca_q(m.tronco(x)[-1]) - m.cabeca_q(m.tronco(x2)[-1]))[0, :, 2, 8]
            assert bool(d.abs().max() > 0) == alcanca, nome
