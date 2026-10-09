"""A CNN totalmente convolucional do CONCEITO 2.5.

    estado (20 × 5 × 9)
       → conv1 → conv2 → conv3          (3×3, 64 canais, padding 1, ReLU)
       ├→ cabeca_q: conv 1×1 → mapa Q (5 plantas × 5 faixas × 9 colunas)
       └→ média global → cabeca_esperar: Linear → Q(esperar)

Saída (B, 226): o mapa achatado em ordem C (planta, faixa, coluna), que é a ordem
de `codificar_acao`, seguido de Q(esperar) no índice 225 (ESPERAR).

Essa é a "cnn1" do rl/ (intacta). As do rl/ ficam no fim do arquivo, no registry
ARQUITETURAS: "cnn1d" (o tronco da cnn1 com entrada de 25 canais e cabeça dueling),
"cnn2" (com bloco de contexto da faixa) e "cnn3" (4 blocos residuais). Todas devolvem
(B, 226) no mesmo índice.
"""
import torch
from torch import nn

from pvz import config as C
from pvz.ambiente import FORMATOS, N_CANAIS, N_PLANTAS

CANAIS = 64
CAMADAS = ("conv1", "conv2", "conv3", "cabeca_q", "cabeca_esperar")
CONVS = ("conv1", "conv2", "conv3")


class CNN(nn.Module):
    def __init__(self, canais=CANAIS):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CANAIS, canais, 3, padding=1)
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=1)
        self.conv3 = nn.Conv2d(canais, canais, 3, padding=1)
        self.cabeca_q = nn.Conv2d(canais, N_PLANTAS, 1)
        self.cabeca_esperar = nn.Linear(canais, 1)

    def tronco(self, x):
        """Devolve as ativações (pós-ReLU) das três convs."""
        a1 = torch.relu(self.conv1(x))
        a2 = torch.relu(self.conv2(a1))
        a3 = torch.relu(self.conv3(a2))
        return a1, a2, a3

    def cabecas(self, h):
        """(mapa (B, 5, 5, 9), esperar (B, 1)) a partir da saída do tronco."""
        # média em vez de AdaptiveAvgPool2d: o backward dela é determinístico na GPU
        return self.cabeca_q(h), self.cabeca_esperar(h.mean(dim=(2, 3)))

    def forward(self, x):
        mapa, esperar = self.cabecas(self.tronco(x)[-1])
        return torch.cat([mapa.flatten(1), esperar], dim=1)


assert N_PLANTAS * C.FAIXAS * C.COLUNAS + 1 == 226


# ---------------------------------------------------------------- CNNs novas do rl/
# Todas recebem a observação "v2" (25 canais: os 20 da seção 2.3, sol/200, e a máscara
# de plantio nos canais 20–24) e terminam na mesma cabeça dueling. A saída continua
# (B, 226) na ordem de codificar_acao. A máscara das 226 ações sai da própria entrada
# (canais 20–24 + "esperar", sempre válida), então forward(x) basta; dá para passá-la
# explicitamente também.
N_CANAIS_V2 = FORMATOS["v2"]
CABECA = ("cabeca_q", "cabeca_esperar", "cabeca_v")


def mascara_da_entrada(x):
    """(B, 226) bool: canais 20–24 achatados em (planta, faixa, coluna) + esperar."""
    plantio = x[:, N_CANAIS:N_CANAIS_V2].flatten(1) > 0.5
    return torch.cat([plantio, plantio.new_ones(len(x), 1)], dim=1)


def _max_acumulado(h):
    """Máximo das colunas 0..c, para cada coluna c (= torch.cummax(h, 3).values). Feito com
    torch.maximum coluna a coluna: o backward do cummax determinístico na GPU leva ~70 ms
    num lote de 256 (mais que a cnn3 inteira); este é elementar."""
    colunas = [h[..., 0]]
    for c in range(1, h.shape[-1]):
        colunas.append(torch.maximum(colunas[-1], h[..., c]))
    return torch.stack(colunas, dim=-1)


def contexto(h):
    """Bloco de contexto da cnn2, antes da conv 1×1: concatena a h, para cada célula (f, c),
    a média e o máximo da faixa f (iguais em toda a faixa), o máximo à direita (colunas
    ≥ c) e à esquerda (colunas ≤ c) na faixa, e a média e o máximo globais. (B, 7C, F, W)."""
    forma = h.shape
    faixa_media = h.mean(dim=3, keepdim=True).expand(forma)
    faixa_max = h.amax(dim=3, keepdim=True).expand(forma)
    esquerda = _max_acumulado(h)
    direita = _max_acumulado(h.flip(3)).flip(3)
    global_media = h.mean(dim=(2, 3), keepdim=True).expand(forma)
    global_max = h.amax(dim=(2, 3), keepdim=True).expand(forma)
    return torch.cat([h, faixa_media, faixa_max, direita, esquerda, global_media, global_max], 1)


class _Dueling(nn.Module):
    """Q(s, a) = V(s) + A(s, a) − média de A sobre as ações VÁLIDAS.

    V e A(esperar) vêm da média global da última camada; A das plantas é o mapa
    5 × 5 × 9 da conv 1×1, alinhado com as células.
    """

    def _criar_cabeca(self, canais):
        self.cabeca_q = nn.Conv2d(canais, N_PLANTAS, 1)
        self.cabeca_esperar = nn.Linear(canais, 1)
        self.cabeca_v = nn.Linear(canais, 1)

    def cabeca(self, h, mascara):
        g = h.mean(dim=(2, 3))
        a = torch.cat([self.cabeca_q(h).flatten(1), self.cabeca_esperar(g)], dim=1)
        m = mascara.to(a.dtype)
        media = (a * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)
        return self.cabeca_v(g) + a - media

    def forward(self, x, mascara=None):
        if mascara is None:
            mascara = mascara_da_entrada(x)
        return self.cabeca(self.tronco(x)[-1], mascara)


class CNN1D(_Dueling):
    """O tronco da cnn1 (conv1 → conv2 → conv3, 3×3, 64), com a entrada v2 e a cabeça dueling."""

    CAMADAS = ("conv1", "conv2", "conv3") + CABECA

    def __init__(self, canais=CANAIS):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CANAIS_V2, canais, 3, padding=1)
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=1)
        self.conv3 = nn.Conv2d(canais, canais, 3, padding=1)
        self._criar_cabeca(canais)

    def tronco(self, x):
        a1 = torch.relu(self.conv1(x))
        a2 = torch.relu(self.conv2(a1))
        a3 = torch.relu(self.conv3(a2))
        return a1, a2, a3


class CNN2(_Dueling):
    """conv1 → conv2 (3×3, 64) → contexto (concat + conv 1×1 → 64) → conv3 → dueling."""

    CAMADAS = ("conv1", "conv2", "contexto", "conv3") + CABECA

    def __init__(self, canais=CANAIS):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CANAIS_V2, canais, 3, padding=1)
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=1)
        self.contexto = nn.Conv2d(7 * canais, canais, 1)
        self.conv3 = nn.Conv2d(canais, canais, 3, padding=1)
        self._criar_cabeca(canais)

    def tronco(self, x):
        a1 = torch.relu(self.conv1(x))
        a2 = torch.relu(self.conv2(a1))
        ac = torch.relu(self.contexto(contexto(a2)))
        a3 = torch.relu(self.conv3(ac))
        return a1, a2, ac, a3


class CNN3(_Dueling):
    """Entrada (conv 3×3, 25 → 64) e 4 blocos residuais h ← ReLU(h + conv(ReLU(conv(h)))),
    8 convs 3×3 de 64 canais; depois a cabeça dueling. Campo receptivo: 19 colunas."""

    BLOCOS = 4
    CAMADAS = (("entrada",) + tuple(f"bloco{i}{l}" for i in range(1, BLOCOS + 1) for l in "ab")
               + CABECA)

    def __init__(self, canais=CANAIS):
        super().__init__()
        self.entrada = nn.Conv2d(N_CANAIS_V2, canais, 3, padding=1)
        for i in range(1, self.BLOCOS + 1):
            setattr(self, f"bloco{i}a", nn.Conv2d(canais, canais, 3, padding=1))
            setattr(self, f"bloco{i}b", nn.Conv2d(canais, canais, 3, padding=1))
        self._criar_cabeca(canais)

    def tronco(self, x):
        h = torch.relu(self.entrada(x))
        saidas = [h]
        for i in range(1, self.BLOCOS + 1):
            a, b = getattr(self, f"bloco{i}a"), getattr(self, f"bloco{i}b")
            h = torch.relu(h + b(torch.relu(a(h))))
            saidas.append(h)
        return tuple(saidas)


# Registry, como OTIMIZADORES e AGENDAS: nome → (classe, formato da observação).
ARQUITETURAS = {
    "cnn1": (CNN, "v1"),      # a de ontem, intacta
    "cnn1d": (CNN1D, "v2"),
    "cnn2": (CNN2, "v2"),
    "cnn3": (CNN3, "v2"),
}


def criar(nome):
    return ARQUITETURAS[nome][0]()


def formato(nome):
    return ARQUITETURAS[nome][1]


def formato_do_modelo(modelo):
    for classe, f in ARQUITETURAS.values():
        if type(modelo) is classe:
            return f
    raise ValueError(f"modelo fora do registry: {type(modelo).__name__}")
