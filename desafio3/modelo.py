"""A CNN totalmente convolucional do CONCEITO 2.5.

    estado (20 × 5 × 9)
       → conv1 → conv2 → conv3          (3×3, 64 canais, padding 1, ReLU)
       ├→ cabeca_q: conv 1×1 → mapa Q (5 plantas × 5 faixas × 9 colunas)
       └→ média global → cabeca_esperar: Linear → Q(esperar)

Saída (B, 226): o mapa achatado em ordem C (planta, faixa, coluna), que é a ordem
de `codificar_acao`, seguido de Q(esperar) no índice 225 (ESPERAR).
"""
import torch
from torch import nn

from pvz import config as C
from pvz.ambiente import N_CANAIS, N_PLANTAS

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
