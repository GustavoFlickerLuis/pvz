"""O simulador (seção 2.2): física pura e determinística, interface reset/step.

    amb = Ambiente()
    estado = amb.reset(("calmo", "misto", "blindados"), semente=0)
    while True:
        mascara = amb.mascara()                 # 226 bools, quais ações são válidas
        estado, recompensa, fim = amb.step(acao)
        if fim: break
"""
import copy

import numpy as np

from . import config as C
from .hordas import Fase, montar_fase

N_PLANTAS = len(C.PLANTAS)
N_CELULAS = C.FAIXAS * C.COLUNAS
N_ACOES = N_PLANTAS * N_CELULAS + 1   # 226
ESPERAR = N_ACOES - 1
N_CANAIS = 20

# Referências de normalização do estado
SOL_REF = 1000
VIDA_ZUMBI_REF = C.ZUMBIS[C.BLINDADO].vida
SURGIDOS_REF = 10


def codificar_acao(planta, faixa, coluna):
    return planta * N_CELULAS + faixa * C.COLUNAS + coluna


def decodificar_acao(acao):
    """Devolve (planta, faixa, coluna), ou None para "esperar"."""
    if acao == ESPERAR:
        return None
    planta, celula = divmod(acao, N_CELULAS)
    faixa, coluna = divmod(celula, C.COLUNAS)
    return planta, faixa, coluna


class Planta:
    __slots__ = ("tipo", "faixa", "coluna", "vida", "timer")

    def __init__(self, tipo, faixa, coluna):
        self.tipo = tipo
        self.faixa = faixa
        self.coluna = coluna
        self.vida = C.PLANTAS[tipo].vida
        # girassol: ticks até o próximo sol; atirador: até poder atirar; mina: até armar
        if tipo == C.GIRASSOL:
            self.timer = C.INTERVALO_GIRASSOL
        elif tipo == C.MINA:
            self.timer = C.MINA_ARMAR
        else:
            self.timer = 0

    @property
    def armada(self):
        return self.tipo == C.MINA and self.timer == 0


class Zumbi:
    __slots__ = ("tipo", "faixa", "x", "vida", "velocidade", "mordida")

    def __init__(self, tipo, faixa, x=float(C.COLUNAS)):
        t = C.ZUMBIS[tipo]
        self.tipo = tipo
        self.faixa = faixa
        self.x = x                     # posição contínua; entra em x = 9 (coluna 8), perde se x < 0
        self.vida = t.vida
        self.velocidade = 1.0 / t.ticks_por_coluna
        self.mordida = t.mordida

    @property
    def coluna(self):
        return min(int(self.x), C.COLUNAS - 1)


class Ambiente:
    def __init__(self, k=C.K_TICKS, max_ticks=C.MAX_TICKS):
        self.k = k
        self.max_ticks = max_ticks
        self.fase = None

    # ------------------------------------------------------------ interface de RL
    def reset(self, fase, semente=0, dificuldade=1.0):
        """`fase`: um objeto Fase, ou as classes de segmento (tupla ou "calmo,misto,blindados")."""
        if not isinstance(fase, Fase):
            fase = montar_fase(fase, semente, dificuldade)
        self.fase = fase
        self.tick = 0
        self.sol = C.SOL_INICIAL
        self.recargas = [0] * N_PLANTAS
        self.grade = [[None] * C.COLUNAS for _ in range(C.FAIXAS)]
        self.plantas = []
        self.zumbis = [[] for _ in range(C.FAIXAS)]
        self.prox_spawn = 0
        self.segmento = 0
        self.surgidos = [0] * C.FAIXAS      # zumbis surgidos por faixa no segmento atual (canal 19)
        self.dano_total = 0.0
        self.plantios = [0] * N_PLANTAS
        self.fim = False
        self.venceu = None
        return self.observar()

    def step(self, acao):
        """Aplica a ação e roda k ticks. Devolve (estado, recompensa, fim)."""
        if self.fim:
            raise RuntimeError("a partida já acabou; chame reset()")
        acao = int(acao)
        if not self.acao_valida(acao):
            raise ValueError(f"ação inválida: {acao} {decodificar_acao(acao)}")
        if acao != ESPERAR:
            self._plantar(*decodificar_acao(acao))

        dano_antes = self.dano_total
        for _ in range(self.k):
            self._tick()
            if self.fim:
                break

        recompensa = (self.dano_total - dano_antes) / self.fase.vida_total
        if self.fim:
            recompensa += 1.0 if self.venceu else -1.0
        return self.observar(), recompensa, self.fim

    def mascara(self):
        """Vetor de 226 bools: ações válidas agora. "Esperar" é sempre válida."""
        m = np.zeros(N_ACOES, dtype=bool)
        livres = np.array([[p is None for p in linha] for linha in self.grade]).ravel()
        for tipo, t in enumerate(C.PLANTAS):
            if self.sol >= t.custo and self.recargas[tipo] == 0:
                m[tipo * N_CELULAS:(tipo + 1) * N_CELULAS] = livres
        m[ESPERAR] = True
        return m

    def acao_valida(self, acao):
        if acao == ESPERAR:
            return True
        if not 0 <= acao < N_ACOES:
            return False
        tipo, faixa, coluna = decodificar_acao(acao)
        return (self.sol >= C.PLANTAS[tipo].custo
                and self.recargas[tipo] == 0
                and self.grade[faixa][coluna] is None)

    def observar(self):
        """Tensor (20, 5, 9) descrito na seção 2.3."""
        e = np.zeros((N_CANAIS, C.FAIXAS, C.COLUNAS), dtype=np.float32)
        for p in self.plantas:
            e[p.tipo, p.faixa, p.coluna] = 1.0
            e[5, p.faixa, p.coluna] = p.vida / C.PLANTAS[p.tipo].vida
            if p.armada:
                e[6, p.faixa, p.coluna] = 1.0
        for faixa, lista in enumerate(self.zumbis):
            for z in lista:
                e[7 + z.tipo, faixa, z.coluna] += 1.0
                e[10, faixa, z.coluna] += z.vida / VIDA_ZUMBI_REF
        e[11] = self.sol / SOL_REF
        for tipo, t in enumerate(C.PLANTAS):
            e[12 + tipo] = self.recargas[tipo] / t.recarga
        e[17] = min(1.0, self.tick / max(1, self.fase.duracao))
        e[18] = np.arange(C.COLUNAS) / (C.COLUNAS - 1)
        e[19] = np.array(self.surgidos, dtype=np.float32)[:, None] / SURGIDOS_REF
        return e

    def clonar(self):
        """Cópia independente do jogo neste instante (para a análise de erros, seção 2.7)."""
        return copy.deepcopy(self)

    def info(self):
        return {
            "tick": self.tick,
            "sol": self.sol,
            "venceu": self.venceu,
            "dano_frac": self.dano_total / self.fase.vida_total,
            "zumbis_vivos": sum(len(l) for l in self.zumbis),
            "zumbis_por_vir": len(self.fase.spawns) - self.prox_spawn,
            "plantios": {C.PLANTAS[i].nome: n for i, n in enumerate(self.plantios)},
        }

    # ------------------------------------------------------------ mecânica
    def _plantar(self, tipo, faixa, coluna):
        self.sol -= C.PLANTAS[tipo].custo
        self.recargas[tipo] = C.PLANTAS[tipo].recarga
        self.plantios[tipo] += 1
        self._colocar(tipo, faixa, coluna)

    def _colocar(self, tipo, faixa, coluna):
        """Põe uma planta sem custo nem recarga (usado pelos testes)."""
        p = Planta(tipo, faixa, coluna)
        self.grade[faixa][coluna] = p
        self.plantas.append(p)
        return p

    def _adicionar_zumbi(self, tipo, faixa, x=float(C.COLUNAS)):
        """Põe um zumbi fora da horda (usado pelos testes)."""
        z = Zumbi(tipo, faixa, x)
        self.zumbis[faixa].append(z)
        return z

    def _ferir(self, z, dano):
        self.dano_total += min(dano, z.vida)   # dano além da vida não conta na recompensa
        z.vida -= dano

    def _alvo(self, p):
        """Zumbi vivo mais próximo à direita da planta (ou na própria célula), na mesma faixa."""
        alvo = None
        for z in self.zumbis[p.faixa]:
            if z.vida > 0 and z.coluna >= p.coluna and (alvo is None or z.x < alvo.x):
                alvo = z
        return alvo

    def _tick(self):
        t = self.tick

        # 1. Sol
        if t > 0 and t % C.INTERVALO_CEU == 0:
            self.sol += C.SOL_CEU
        for p in self.plantas:
            if p.tipo == C.GIRASSOL:
                p.timer -= 1
                if p.timer == 0:
                    self.sol += C.SOL_GIRASSOL
                    p.timer = C.INTERVALO_GIRASSOL

        # 2. Spawn
        spawns = self.fase.spawns
        while self.prox_spawn < len(spawns) and spawns[self.prox_spawn].tick <= t:
            s = spawns[self.prox_spawn]
            if s.segmento != self.segmento:
                self.segmento = s.segmento
                self.surgidos = [0] * C.FAIXAS
            self.zumbis[s.faixa].append(Zumbi(s.tipo, s.faixa))
            self.surgidos[s.faixa] += 1
            self.prox_spawn += 1

        # 3. Plantas agem
        for p in self.plantas:
            if p.tipo in C.TIROS:
                if p.timer > 0:
                    p.timer -= 1
                if p.timer == 0:
                    atirou = False
                    for _ in range(C.TIROS[p.tipo]):
                        alvo = self._alvo(p)
                        if alvo is None:
                            break
                        self._ferir(alvo, C.DANO_TIRO)
                        atirou = True
                    if atirou:
                        p.timer = C.INTERVALO_TIRO
            elif p.tipo == C.MINA and p.vida > 0:
                if p.timer > 0:
                    p.timer -= 1
                if p.timer == 0:
                    na_celula = [z for z in self.zumbis[p.faixa] if z.vida > 0 and z.coluna == p.coluna]
                    if na_celula:
                        for z in na_celula:
                            self._ferir(z, C.MINA_DANO)
                        p.vida = 0   # uso único

        # 4. Zumbis bloqueados mordem (e não andam)  /  5. Zumbis livres andam
        livres = []
        for faixa, lista in enumerate(self.zumbis):
            linha = self.grade[faixa]
            for z in lista:
                if z.vida <= 0:
                    continue
                p = linha[z.coluna]
                if p is not None and p.vida > 0:
                    p.vida -= z.mordida
                else:
                    livres.append(z)
        for z in livres:
            z.x -= z.velocidade

        # 6. Remoção
        for faixa in range(C.FAIXAS):
            self.zumbis[faixa] = [z for z in self.zumbis[faixa] if z.vida > 0]
        if any(p.vida <= 0 for p in self.plantas):
            for p in self.plantas:
                if p.vida <= 0:
                    self.grade[p.faixa][p.coluna] = None
            self.plantas = [p for p in self.plantas if p.vida > 0]

        # 7. Fim?
        if any(z.x < 0 for lista in self.zumbis for z in lista):
            self.fim, self.venceu = True, False
        elif self.prox_spawn == len(spawns) and not any(self.zumbis):
            self.fim, self.venceu = True, True

        self.recargas = [max(0, r - 1) for r in self.recargas]
        self.tick += 1
        if not self.fim and self.tick >= self.max_ticks:
            self.fim, self.venceu = True, False
