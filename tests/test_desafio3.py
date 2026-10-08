"""Testes do Desafio 3: a saída da CNN, o registry e as agendas."""
import math
import unittest

import torch

from desafio3 import treino
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
        cfg = treino.Config()
        for nome, fabrica in treino.OTIMIZADORES.items():
            if nome == "muon":
                continue
            opts = fabrica(CNN(), cfg)
            self.assertIsInstance(opts, list, nome)
            self.assertTrue(all(isinstance(o, torch.optim.Optimizer) for o in opts), nome)

    def test_muon_e_do_pablo(self):
        with self.assertRaisesRegex(NotImplementedError, "parte do Pablo"):
            treino.OTIMIZADORES["muon"](CNN(), treino.Config())

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


if __name__ == "__main__":
    unittest.main()
