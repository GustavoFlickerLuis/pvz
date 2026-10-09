"""Testes das regras do simulador.  Rodar com:  python -m unittest -v"""
import hashlib
import itertools
import random
import unittest

import numpy as np

from pvz import config as C
from pvz.agentes import Econ, Guloso
from pvz.ambiente import ESPERAR, N_ACOES, N_CANAIS, Ambiente, codificar_acao
from pvz.divisao import FASES_TESTE, FASES_TREINO, SEMENTES_TESTE, SEMENTES_TREINO, VITRINE
from pvz.hordas import (INICIO, MEIO, PESO_SEGMENTO, SEGMENTOS, VAR_QUANTIDADE, fase_manual,
                        montar_fase, todas_as_fases)


def ambiente_vazio():
    """Gramado sem horda (um zumbi bem no futuro só para a fase não acabar)."""
    amb = Ambiente()
    amb.reset(fase_manual([(10_000, C.COMUM, 0)]))
    return amb


def ticks(amb, n):
    for _ in range(n):
        amb._tick()


def partida_aleatoria(fase, semente_fase, semente_agente):
    amb = Ambiente()
    amb.reset(fase, semente_fase)
    rng = random.Random(semente_agente)
    historico = []
    fim = False
    while not fim:
        acao = int(rng.choice(np.flatnonzero(amb.mascara())))
        _, r, fim = amb.step(acao)
        historico.append((acao, r))
    return amb, historico


class TestDeterminismo(unittest.TestCase):
    def test_mesma_semente_mesma_partida(self):
        a, ha = partida_aleatoria("calmo,misto,blindados", 3, 7)
        b, hb = partida_aleatoria("calmo,misto,blindados", 3, 7)
        self.assertEqual(ha, hb)
        self.assertEqual(a.tick, b.tick)

    def test_clonar_e_independente(self):
        amb, _ = partida_aleatoria("calmo", 0, 0)
        amb.reset("calmo,misto,blindados", 0)
        for _ in range(50):
            amb.step(ESPERAR)
        copia = amb.clonar()
        r1 = [amb.step(ESPERAR)[1] for _ in range(30)]
        r2 = [copia.step(ESPERAR)[1] for _ in range(30)]
        self.assertEqual(r1, r2)


class TestSol(unittest.TestCase):
    def test_ceu(self):
        amb = ambiente_vazio()
        ticks(amb, C.INTERVALO_CEU)
        self.assertEqual(amb.sol, C.SOL_INICIAL)
        ticks(amb, 1)
        self.assertEqual(amb.sol, C.SOL_INICIAL + C.SOL_CEU)

    def test_girassol(self):
        amb = ambiente_vazio()
        amb._colocar(C.GIRASSOL, 0, 0)
        ticks(amb, C.INTERVALO_GIRASSOL - 1)
        self.assertEqual(amb.sol, C.SOL_INICIAL)
        ticks(amb, 1)
        self.assertEqual(amb.sol, C.SOL_INICIAL + C.SOL_GIRASSOL)


class TestTiro(unittest.TestCase):
    def _cena(self, x_zumbi):
        amb = ambiente_vazio()
        amb._colocar(C.ERVILHA, 2, 5)
        z = amb._adicionar_zumbi(C.COMUM, 2, x_zumbi)
        ticks(amb, 1)
        return z

    def test_nao_atira_para_tras(self):
        self.assertEqual(self._cena(3.5).vida, C.ZUMBIS[C.COMUM].vida)

    def test_atira_para_frente_e_na_propria_celula(self):
        self.assertEqual(self._cena(7.5).vida, C.ZUMBIS[C.COMUM].vida - C.DANO_TIRO)
        self.assertEqual(self._cena(5.5).vida, C.ZUMBIS[C.COMUM].vida - C.DANO_TIRO)

    def test_nao_atinge_outra_faixa(self):
        amb = ambiente_vazio()
        amb._colocar(C.ERVILHA, 2, 0)
        z = amb._adicionar_zumbi(C.COMUM, 3, 7.5)
        ticks(amb, 100)
        self.assertEqual(z.vida, C.ZUMBIS[C.COMUM].vida)

    def test_cadencia_e_repetidora(self):
        amb = ambiente_vazio()
        amb._colocar(C.ERVILHA, 0, 0)
        amb._colocar(C.REPETIDORA, 1, 0)
        ze = amb._adicionar_zumbi(C.BLINDADO, 0, 8.5)
        zr = amb._adicionar_zumbi(C.BLINDADO, 1, 8.5)
        ticks(amb, 1 + C.INTERVALO_TIRO * 3)   # 4 disparos
        self.assertEqual(ze.vida, C.ZUMBIS[C.BLINDADO].vida - 4 * C.DANO_TIRO)
        self.assertEqual(zr.vida, C.ZUMBIS[C.BLINDADO].vida - 8 * C.DANO_TIRO)


class TestBloqueioEMina(unittest.TestCase):
    def test_batata_segura(self):
        amb = ambiente_vazio()
        b = amb._colocar(C.BATATA, 0, 3)
        z = amb._adicionar_zumbi(C.COMUM, 0, 3.5)
        ticks(amb, 100)
        self.assertEqual(z.x, 3.5)
        self.assertEqual(b.vida, C.PLANTAS[C.BATATA].vida - 100 * C.ZUMBIS[C.COMUM].mordida)

    def test_zumbi_anda_quando_planta_morre(self):
        amb = ambiente_vazio()
        amb._colocar(C.GIRASSOL, 0, 3)
        z = amb._adicionar_zumbi(C.COMUM, 0, 3.5)
        ticks(amb, C.PLANTAS[C.GIRASSOL].vida // C.ZUMBIS[C.COMUM].mordida + 1)
        self.assertIsNone(amb.grade[0][3])
        ticks(amb, 5)
        self.assertLess(z.x, 3.5)

    def test_mina_desarmada_e_comida(self):
        amb = ambiente_vazio()
        amb._colocar(C.MINA, 0, 4)
        z = amb._adicionar_zumbi(C.COMUM, 0, 4.5)
        ticks(amb, 61)
        self.assertIsNone(amb.grade[0][4])
        self.assertEqual(z.vida, C.ZUMBIS[C.COMUM].vida)

    def test_mina_armada_explode(self):
        amb = ambiente_vazio()
        amb._colocar(C.MINA, 0, 4)
        amb._colocar(C.MINA, 1, 4)
        ticks(amb, C.MINA_ARMAR)
        z1 = amb._adicionar_zumbi(C.BLINDADO, 0, 4.5)
        z2 = amb._adicionar_zumbi(C.COMUM, 0, 4.9)
        z3 = amb._adicionar_zumbi(C.COMUM, 0, 5.5)   # célula vizinha: escapa
        ticks(amb, 1)
        self.assertNotIn(z1, amb.zumbis[0])
        self.assertNotIn(z2, amb.zumbis[0])
        self.assertEqual(z3.vida, C.ZUMBIS[C.COMUM].vida)
        self.assertIsNone(amb.grade[0][4])        # uso único
        self.assertIsNotNone(amb.grade[1][4])     # a outra faixa não foi tocada

    def test_mina_espera_zumbi_chegar(self):
        amb = ambiente_vazio()
        amb._colocar(C.MINA, 0, 4)
        z = amb._adicionar_zumbi(C.COMUM, 0, 8.9)
        ticks(amb, 250)   # chega na coluna 4 depois de ~195 ticks, mina já armada
        self.assertNotIn(z, amb.zumbis[0])
        self.assertIsNone(amb.grade[0][4])


class TestFimERecompensa(unittest.TestCase):
    def test_derrota_sem_defesa(self):
        amb = Ambiente()
        amb.reset(fase_manual([(0, C.RAPIDO, 2)]))
        total, fim = 0.0, False
        while not fim:
            _, r, fim = amb.step(ESPERAR)
            total += r
        self.assertFalse(amb.venceu)
        self.assertAlmostEqual(total, -1.0)
        self.assertAlmostEqual(amb.tick, C.COLUNAS * C.ZUMBIS[C.RAPIDO].ticks_por_coluna, delta=2)

    def test_vitoria_e_dano_soma_um(self):
        amb = Ambiente()
        amb.reset(fase_manual([(0, C.COMUM, 0), (50, C.COMUM, 0)]))
        _, total, fim = amb.step(codificar_acao(C.ERVILHA, 0, 0))
        while not fim:
            _, r, fim = amb.step(ESPERAR)
            total += r
        self.assertTrue(amb.venceu)
        self.assertAlmostEqual(total, 2.0)   # dano (1) + vitória (1)


class TestMascaraEEstado(unittest.TestCase):
    def test_mascara(self):
        amb = ambiente_vazio()
        m = amb.mascara()
        self.assertEqual(m.shape, (N_ACOES,))
        self.assertTrue(m[ESPERAR])
        self.assertTrue(m[codificar_acao(C.ERVILHA, 0, 0)])        # 100 de sol
        self.assertFalse(m[codificar_acao(C.REPETIDORA, 0, 0)])    # custa 200
        amb.step(codificar_acao(C.GIRASSOL, 0, 0))
        m = amb.mascara()
        self.assertFalse(m[codificar_acao(C.ERVILHA, 0, 0)])       # célula ocupada
        self.assertFalse(m[codificar_acao(C.GIRASSOL, 1, 0)])      # recarga
        self.assertFalse(m[codificar_acao(C.ERVILHA, 1, 0)])       # sobrou só 50 de sol
        self.assertTrue(m[codificar_acao(C.MINA, 1, 0)])

    def test_acao_invalida_levanta_erro(self):
        amb = ambiente_vazio()
        with self.assertRaises(ValueError):
            amb.step(codificar_acao(C.REPETIDORA, 0, 0))

    def test_estado(self):
        amb = ambiente_vazio()
        amb.step(codificar_acao(C.MINA, 3, 2))
        e = amb.observar()
        self.assertEqual(e.shape, (N_CANAIS, C.FAIXAS, C.COLUNAS))
        self.assertEqual(e[C.MINA, 3, 2], 1.0)
        self.assertEqual(e[6, 3, 2], 0.0)   # ainda não armada
        self.assertEqual(e[18, 0, 8], 1.0)


class TestFases(unittest.TestCase):
    def test_27_fases_montam(self):
        fases = todas_as_fases()
        self.assertEqual(len(fases), 27)
        for f in fases:
            fase = montar_fase(f, 0)
            self.assertGreater(len(fase.spawns), 0)
            self.assertTrue(all(0 <= s.faixa < C.FAIXAS for s in fase.spawns))

    def test_mesma_semente_mesma_fase(self):
        for f in todas_as_fases():
            self.assertEqual(montar_fase(f, 5, 1.3).spawns, montar_fase(f, 5, 1.3).spawns)

    def test_forma_das_classes_nao_muda_a_fase(self):
        """String, string com "+", tupla ou lista: a chave do rng é a mesma."""
        for s in range(5):
            ref = montar_fase(("calmo", "misto", "blindados"), s).spawns
            for forma in ("calmo,misto,blindados", "calmo+misto+blindados",
                          " calmo, misto ,blindados", ["calmo", "misto", "blindados"]):
                self.assertEqual(montar_fase(forma, s).spawns, ref, forma)

    def test_semente_muda_a_historia(self):
        """A semente sorteia quantidade, ticks e composição, não só a faixa."""
        fases = [montar_fase("calmo,misto,blindados", s) for s in range(10)]
        self.assertGreater(len({len(f.spawns) for f in fases}), 1)
        self.assertGreater(len({tuple(s.tick for s in f.spawns) for f in fases}), 1)
        self.assertGreater(len({tuple(s.tipo for s in f.spawns) for f in fases}), 1)

    def test_fases_diferentes_nao_compartilham_sorteio(self):
        """calmo e so_rapidos têm o mesmo número de zumbis; o meio não pode sair igual."""
        a = montar_fase("calmo,misto,blindados", 0)
        b = montar_fase("so_rapidos,misto,blindados", 0)
        meio = lambda f: [(s.tick, s.tipo, s.faixa) for s in f.spawns if s.segmento == 1]
        self.assertNotEqual(meio(a), meio(b))

    def _segmentos(self, f, semente, dificuldade=1.0):
        """Para cada segmento: (classe, escala, início absoluto, spawns)."""
        fase = montar_fase(f, semente, dificuldade)
        inicio, saida = 0, []
        for i, (nome, tabela) in enumerate(zip(f, SEGMENTOS)):
            classe = tabela[nome]
            spawns = [s for s in fase.spawns if s.segmento == i]
            saida.append((classe, dificuldade * PESO_SEGMENTO[i], inicio, spawns))
            inicio += classe.duracao
        return saida

    def test_spawns_dentro_do_segmento(self):
        for f in todas_as_fases():
            for semente in range(5):
                for dificuldade in (0.5, 1.0, 2.0):
                    for classe, _, inicio, spawns in self._segmentos(f, semente, dificuldade):
                        for s in spawns:
                            self.assertGreaterEqual(s.tick, inicio)
                            self.assertLess(s.tick, inicio + classe.duracao)

    def test_quantidade_e_composicao(self):
        for f in todas_as_fases():
            for semente in range(5):
                for classe, escala, _, spawns in self._segmentos(f, semente):
                    minimo = sum(max(1, round(q.n * escala * (1 - VAR_QUANTIDADE))) for q in classe.filas)
                    maximo = sum(max(1, round(q.n * escala * (1 + VAR_QUANTIDADE))) for q in classe.filas)
                    self.assertTrue(minimo <= len(spawns) <= maximo, (f, semente, len(spawns)))
                    tipos = {t for q in classe.filas for t in q.padrao}
                    self.assertTrue({s.tipo for s in spawns} <= tipos)

    def test_composicao_segue_proporcoes(self):
        """Em muitas sementes, a proporção de cada tipo se aproxima da do padrão."""
        contagem, total = [0] * len(C.ZUMBIS), 0
        for semente in range(200):
            for s in montar_fase("calmo,misto", semente).spawns:
                if s.segmento == 1:
                    contagem[s.tipo] += 1
                    total += 1
        padrao = MEIO["misto"].filas[0].padrao
        for tipo in set(padrao):
            self.assertAlmostEqual(contagem[tipo] / total, padrao.count(tipo) / len(padrao), delta=0.04)

    def test_uma_faixa_pesada_concentra(self):
        fila = MEIO["uma_faixa_pesada"].filas[0]
        for semente in range(20):
            spawns = [s for s in montar_fase("calmo,uma_faixa_pesada", semente).spawns if s.segmento == 1]
            maior = max(sum(s.faixa == f for s in spawns) for f in range(C.FAIXAS))
            self.assertGreaterEqual(maior, round(fila.n * PESO_SEGMENTO[1] * (1 - VAR_QUANTIDADE)))

    def test_pesos_por_argumento(self):
        """Os pesos podem ser passados sem mexer no módulo, e o padrão não muda."""
        leve = montar_fase("calmo,misto,blindados", 0, pesos=(1.0, 1.0, 1.0))
        normal = montar_fase("calmo,misto,blindados", 0)
        self.assertLess(len(leve.spawns), len(normal.spawns))
        self.assertEqual(normal.spawns, montar_fase("calmo,misto,blindados", 0).spawns)


class TestDivisao(unittest.TestCase):
    def test_teste_e_treino_disjuntos_e_completos(self):
        self.assertEqual(len(FASES_TESTE), 6)
        self.assertEqual(len(FASES_TREINO), 21)
        self.assertFalse(set(FASES_TESTE) & set(FASES_TREINO))
        self.assertEqual(set(FASES_TESTE) | set(FASES_TREINO), set(todas_as_fases()))

    def test_treino_ve_todas_as_classes_e_pares(self):
        for i, tabela in enumerate(SEGMENTOS):
            self.assertEqual({f[i] for f in FASES_TREINO}, set(tabela))
        self.assertEqual({f[:2] for f in FASES_TREINO}, set(itertools.product(INICIO, MEIO)))

    def test_cada_classe_duas_vezes_no_teste(self):
        for i, tabela in enumerate(SEGMENTOS):
            for nome in tabela:
                self.assertEqual(sum(f[i] == nome for f in FASES_TESTE), 2, nome)

    def test_vitrine_e_sementes(self):
        self.assertEqual(len(VITRINE), 3)
        for fase, semente in VITRINE:
            self.assertIn(fase, FASES_TESTE)
            self.assertNotIn(semente, SEMENTES_TREINO)
            self.assertNotIn(semente, SEMENTES_TESTE)
        self.assertFalse(set(SEMENTES_TREINO) & set(SEMENTES_TESTE))


class TestAgentes(unittest.TestCase):
    @staticmethod
    def impressao(agente):
        """Hash das ações do agente em 14 partidas (7 fases × 2 sementes)."""
        h = hashlib.sha256()
        for fase in todas_as_fases()[::4]:
            for semente in (0, 1):
                amb = Ambiente()
                amb.reset(fase, semente)
                fim = False
                while not fim:
                    a = agente.agir(amb)
                    h.update(a.to_bytes(2, "little"))
                    _, _, fim = amb.step(a)
        return h.hexdigest()[:16]

    def test_jogadas_nao_mudam(self):
        # Valores gravados antes de Guloso._ervilha passar a devolver None.
        # Mudar as regras do jogo ou dos agentes muda estes hashes.
        self.assertEqual(self.impressao(Guloso()), "38376f9c6a6b707d")
        self.assertEqual(self.impressao(Econ(5)), "182e2523db60775d")


if __name__ == "__main__":
    unittest.main()


class TestCanais(unittest.TestCase):
    def test_os_20_canais_do_conceito_2_3(self):
        """Cada canal da tabela do CONCEITO 2.3 num tabuleiro montado à mão."""
        from pvz.ambiente import SOL_REF, SURGIDOS_REF, VIDA_ZUMBI_REF
        amb = Ambiente()
        amb.reset(fase_manual([(0, C.RAPIDO, 4), (0, C.RAPIDO, 4), (5_000, C.COMUM, 0)]))
        amb._tick()                                       # surgem os 2 rápidos na faixa 4
        g = amb._colocar(C.GIRASSOL, 0, 0)
        g.vida = 150
        amb._colocar(C.ERVILHA, 1, 1)
        amb._colocar(C.REPETIDORA, 2, 2)
        amb._colocar(C.BATATA, 3, 3)
        m = amb._colocar(C.MINA, 4, 4)
        m.timer = 0                                       # armada
        amb._colocar(C.MINA, 0, 5)                        # desarmada
        amb._adicionar_zumbi(C.COMUM, 1, 6.5)
        b = amb._adicionar_zumbi(C.BLINDADO, 1, 6.2)
        b.vida = 550
        amb.sol = 375
        amb.recargas = [10, 0, 25, 200, 50]
        e = amb.observar()
        self.assertEqual(e.shape, (20, 5, 9))
        for tipo, (f, c) in enumerate([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]):
            self.assertEqual(e[tipo, f, c], 1.0)
            self.assertEqual(e[tipo].sum(), 2.0 if tipo == C.MINA else 1.0)   # 0–4 one-hot
        self.assertEqual(e[5, 0, 0], 0.5)                                    # 5 vida
        self.assertEqual(e[5, 3, 3], 1.0)
        self.assertEqual((e[6, 4, 4], e[6, 0, 5], e[6].sum()), (1.0, 0.0, 1.0))   # 6 mina armada
        self.assertEqual((e[7, 1, 6], e[8, 1, 6], e[7].sum(), e[8].sum()), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual((e[9, 4, 8], e[9].sum()), (2.0, 2.0))                # 7–9 contagem
        self.assertAlmostEqual(e[10, 1, 6], (200 + 550) / VIDA_ZUMBI_REF, places=6)   # 10
        self.assertAlmostEqual(e[10, 4, 8], 300 / VIDA_ZUMBI_REF, places=6)
        self.assertTrue(np.all(e[11] == np.float32(375 / SOL_REF)))           # 11 sol
        for tipo, r in enumerate([10, 0, 25, 200, 50]):                       # 12–16 recargas
            self.assertTrue(np.all(e[12 + tipo] == np.float32(r / C.PLANTAS[tipo].recarga)))
        self.assertTrue(np.all(e[17] == np.float32(1 / amb.fase.duracao)))   # 17 progresso
        self.assertTrue(np.allclose(e[18], np.arange(9) / 8))                 # 18 coluna
        self.assertTrue(np.allclose(e[19][:, 0], [0, 0, 0, 0, 2 / SURGIDOS_REF]))   # 19
        self.assertTrue(np.all(e[19] == e[19][:, :1]))
