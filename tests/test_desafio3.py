"""Testes do Desafio 3: a saída da CNN, o registry e as agendas."""
import math
import unittest

import torch

from desafio3 import muon, treino
from desafio3.modelo import CNN
from pvz import config as C
from pvz.ambiente import ESPERAR, N_ACOES, N_CANAIS, N_PLANTAS, codificar_acao


class TestModelo(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.modelo = CNN()
        self.x = torch.randn(4, N_CANAIS, C.FAIXAS, C.COLUNAS)

    def test_indice_da_acao_e_a_celula_do_mapa(self):
        q = self.modelo(self.x)
        self.assertEqual(q.shape, (4, N_ACOES))
        mapa, esperar = self.modelo.cabecas(self.modelo.tronco(self.x)[-1])
        for p in range(N_PLANTAS):
            for f in range(C.FAIXAS):
                for c in range(C.COLUNAS):
                    self.assertTrue(torch.equal(q[:, codificar_acao(p, f, c)], mapa[:, p, f, c]))
        self.assertTrue(torch.equal(q[:, ESPERAR], esperar[:, 0]))

    def test_esperar_vem_so_da_cabeca_esperar(self):
        q = self.modelo(self.x)
        q[:, ESPERAR].sum().backward()
        g = self.modelo.cabeca_q.weight.grad
        self.assertTrue(g is None or not g.any())
        self.assertGreater(self.modelo.cabeca_esperar.weight.grad.abs().sum(), 0)


class TestRegistries(unittest.TestCase):
    def test_otimizadores_devolvem_lista(self):
        for nome, fabrica in treino.OTIMIZADORES.items():
            opts = fabrica(CNN(), treino.Config(otimizador=nome, lr_adamw=1e-3))
            self.assertIsInstance(opts, list, nome)
            self.assertTrue(all(isinstance(o, torch.optim.Optimizer) for o in opts), nome)

    def test_muon_exige_lr_adamw(self):
        with self.assertRaises(ValueError):
            treino.OTIMIZADORES["muon"](CNN(), treino.Config(otimizador="muon"))

    def test_agendas_entre_0_e_1(self):
        total = 1000
        for nome, f in treino.AGENDAS.items():
            valores = [f(t, total) for t in range(total)]
            self.assertTrue(all(0 <= v <= 1 for v in valores), nome)
        self.assertEqual(treino.AGENDAS["constante"](500, total), 1)
        self.assertTrue(math.isclose(treino.AGENDAS["cosseno"](500, total), 0.5))
        # wsd: estável até 80%, decai linearmente nos últimos 20%
        wsd = treino.AGENDAS["wsd"]
        self.assertEqual(wsd(799, total), 1)
        self.assertTrue(math.isclose(wsd(900, total), 0.5))
        # warmup em 5% dos passos
        self.assertLess(treino.AGENDAS["warmup_cosseno"](10, total), 0.25)
        self.assertEqual(treino.AGENDAS["warmup_cosseno"](49, total), 1)


# faixa dos valores singulares depois de 5 iterações de Newton–Schulz com os coeficientes do
# enunciado: eles não convergem para 1 exatamente, só para uma faixa em torno de 1
FAIXA_NS = (0.6, 1.25)


class TestMuon(unittest.TestCase):
    def test_ortogonaliza_valores_singulares_perto_de_1(self):
        g = torch.Generator().manual_seed(0)
        for forma in ((64, 180), (64, 576), (576, 64), (32, 32)):
            s = torch.linalg.svdvals(muon.ortogonaliza(torch.randn(*forma, generator=g)))
            self.assertGreater(s.min().item(), FAIXA_NS[0], forma)
            self.assertLess(s.max().item(), FAIXA_NS[1], forma)

    def test_cada_parametro_em_exatamente_um_otimizador(self):
        modelo = CNN()
        opts = muon.criar(modelo, treino.Config(otimizador="muon", lr=0.02, lr_adamw=1e-3))
        self.assertEqual(len(opts), 2)
        self.assertIsInstance(opts[0], muon.Muon)
        contagem = {}
        for o in opts:
            for grupo in o.param_groups:
                for p in grupo["params"]:
                    contagem[id(p)] = contagem.get(id(p), 0) + 1
        self.assertEqual(set(contagem), {id(p) for p in modelo.parameters()})
        self.assertTrue(all(n == 1 for n in contagem.values()))
        no_muon = {id(p) for p in opts[0].param_groups[0]["params"]}
        self.assertEqual(no_muon, {id(getattr(modelo, n).weight) for n in ("conv1", "conv2", "conv3")})

    def test_passo_preserva_formato_dos_kernels(self):
        modelo = CNN()
        formas = {n: p.shape for n, p in modelo.named_parameters()}
        opts = muon.criar(modelo, treino.Config(otimizador="muon", lr=0.02, lr_adamw=1e-3))
        antes = {n: p.detach().clone() for n, p in modelo.named_parameters()}
        modelo(torch.randn(8, N_CANAIS, C.FAIXAS, C.COLUNAS)).pow(2).mean().backward()
        for o in opts:
            o.step()
        for n, p in modelo.named_parameters():
            self.assertEqual(p.shape, formas[n], n)
            self.assertFalse(torch.equal(p, antes[n]), n)

    def test_muon_deterministico(self):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def rodar():
            treino.fixar_sementes(0)   # os mesmos modos determinísticos do treino real
            modelo = CNN().to(dev)
            opts = muon.criar(modelo, treino.Config(otimizador="muon", lr=0.02, lr_adamw=1e-3))
            g = torch.Generator(device=dev).manual_seed(1)
            for _ in range(20):
                x = torch.randn(64, N_CANAIS, C.FAIXAS, C.COLUNAS, generator=g, device=dev)
                a = torch.randint(N_ACOES, (64,), generator=g, device=dev)
                y = torch.randn(64, generator=g, device=dev)
                perda = treino.perda_q(modelo(x), a, y, 1.0)
                for o in opts:
                    o.zero_grad()
                perda.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 10.0)
                for o in opts:
                    o.step()
            return [p.detach().cpu() for p in modelo.parameters()]

        for p1, p2 in zip(rodar(), rodar()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
