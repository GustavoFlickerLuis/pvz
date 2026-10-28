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


# ---------------------------------------------------------------- braços do lab (LAB.md)
# Todas: entrada v2 (B, 25, 5, 9); partes(x) → (mapa A dos plantios (B, 5, 5, 9) na ordem
# (planta, faixa, coluna) de codificar_acao, A(esperar) (B, 1), V (B, 1)). forward(x) é o
# Q dueling (para o teto do diagnóstico); o PPO lê as partes direto (logits = A, crítico = V).
N_CEL = C.FAIXAS * C.COLUNAS


class _Partes(nn.Module):
    def forward(self, x, mascara=None):
        if mascara is None:
            mascara = mascara_da_entrada(x)
        mapa, esperar, v = self.partes(x)
        a = torch.cat([mapa.flatten(1), esperar], dim=1)
        m = mascara.to(a.dtype)
        media = (a * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)
        return v + a - media


class Posicoes(nn.Module):
    """Embeddings aprendidos de faixa e de coluna, somados: (45, d) na ordem faixa × coluna."""

    def __init__(self, d):
        super().__init__()
        self.faixa = nn.Parameter(torch.randn(C.FAIXAS, d) * 0.02)
        self.coluna = nn.Parameter(torch.randn(C.COLUNAS, d) * 0.02)

    def forward(self):
        return (self.faixa[:, None, :] + self.coluna[None, :, :]).reshape(N_CEL, -1)


class Atencao(nn.Module):
    """Self-attention pre-LN, residual, sem MLP: t ← t + Wo · MHA(LN(t + pos)).

    A posição entra só na entrada do ramo (q, k, v), não no resíduo: com Wo = 0 a camada
    é exatamente a identidade (é o que faz o braço C começar igual à cnn1d)."""

    def __init__(self, d, cabecas, zerar_saida=False):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, cabecas, batch_first=True)
        if zerar_saida:
            nn.init.zeros_(self.mha.out_proj.weight)
            nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, t, pos):
        u = self.ln(t + pos)
        return t + self.mha(u, u, u, need_weights=False)[0]


def _grade_para_tokens(h):
    """(B, d, 5, 9) → (B, 45, d), células na ordem faixa × coluna."""
    return h.flatten(2).transpose(1, 2)


def _tokens_para_grade(t):
    return t.transpose(1, 2).reshape(len(t), -1, C.FAIXAS, C.COLUNAS)


class _CabecaCNN1D(nn.Module):
    """As cabeças da cnn1d: mapa = conv 1×1; esperar e V = Linear na média global."""

    def _criar_cabeca(self, canais):
        self.cabeca_q = nn.Conv2d(canais, N_PLANTAS, 1)
        self.cabeca_esperar = nn.Linear(canais, 1)
        self.cabeca_v = nn.Linear(canais, 1)

    def _cabecas(self, h):
        g = h.mean(dim=(2, 3))
        return self.cabeca_q(h), self.cabeca_esperar(g), self.cabeca_v(g)


class AtencaoAntes(_CabecaCNN1D, _Partes):
    """B: projeção por célula 25 → 64 + posições, 1 camada de atenção (4 × 16), depois o
    tronco da cnn1d (conv1 com 64 canais de entrada, não 25) e as cabeças dela."""

    CAMADAS = ("proj", "pos", "atencao", "conv1", "conv2", "conv3") + CABECA

    def __init__(self, canais=CANAIS, cabecas=4):
        super().__init__()
        self.proj = nn.Linear(N_CANAIS_V2, canais)
        self.pos = Posicoes(canais)
        self.atencao = Atencao(canais, cabecas)
        self.conv1 = nn.Conv2d(canais, canais, 3, padding=1)
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=1)
        self.conv3 = nn.Conv2d(canais, canais, 3, padding=1)
        self._criar_cabeca(canais)

    def partes(self, x):
        t = self.atencao(self.proj(_grade_para_tokens(x)), self.pos())
        h = torch.relu(self.conv1(_tokens_para_grade(t)))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        return self._cabecas(h)


class AtencaoDepois(_CabecaCNN1D, _Partes):
    """C: o tronco da cnn1d (mesmos nomes de camada), posições e 1 camada de atenção igual à
    do B com a projeção de saída zerada (começa igual à cnn1d), depois as cabeças."""

    CAMADAS = ("conv1", "conv2", "conv3", "pos", "atencao") + CABECA

    def __init__(self, canais=CANAIS, cabecas=4):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CANAIS_V2, canais, 3, padding=1)
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=1)
        self.conv3 = nn.Conv2d(canais, canais, 3, padding=1)
        self.pos = Posicoes(canais)
        self.atencao = Atencao(canais, cabecas, zerar_saida=True)
        self._criar_cabeca(canais)

    def partes(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        h = _tokens_para_grade(self.atencao(_grade_para_tokens(h), self.pos()))
        return self._cabecas(h)


class Token(nn.Module):
    """Um vetor aprendido (o CLS), como módulo para entrar nas CAMADAS."""

    def __init__(self, d):
        super().__init__()
        self.valor = nn.Parameter(torch.randn(1, 1, d) * 0.02)


class Bloco(nn.Module):
    """Bloco pre-LN do encoder: t ← t + MHA(LN(t)); t ← t + MLP(LN(t)), MLP 4× com GELU."""

    def __init__(self, d, cabecas):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, cabecas, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, t):
        u = self.ln1(t)
        t = t + self.mha(u, u, u, need_weights=False)[0]
        return t + self.mlp(self.ln2(t))


class Transformer(_Partes):
    """D (d = 64) e D2 (d = 128): encoder sem máscara causal. 45 tokens (célula 25 → d +
    posições) + um token CLS; 2 blocos pre-LN; LN final. Plantios = Linear d → 5 em cada
    token de célula; esperar e V a partir do CLS."""

    CAMADAS = ("proj", "pos", "cls", "blocos", "ln", "cabeca_plantio", "cabeca_esperar", "cabeca_v")

    def __init__(self, d=64, cabecas=4, blocos=2):
        super().__init__()
        self.proj = nn.Linear(N_CANAIS_V2, d)
        self.pos = Posicoes(d)
        self.cls = Token(d)
        self.blocos = nn.ModuleList(Bloco(d, cabecas) for _ in range(blocos))
        self.ln = nn.LayerNorm(d)
        self.cabeca_plantio = nn.Linear(d, N_PLANTAS)
        self.cabeca_esperar = nn.Linear(d, 1)
        self.cabeca_v = nn.Linear(d, 1)

    def tokens(self, x):
        """(B, 45, d): conteúdo da célula + posição (o teste de permutação entra aqui)."""
        return self.proj(_grade_para_tokens(x)) + self.pos()

    def codificar(self, t):
        """tokens de célula (B, n, d) → (células (B, n, d), CLS (B, d)) depois do encoder."""
        t = torch.cat([self.cls.valor.expand(len(t), -1, -1), t], dim=1)
        for b in self.blocos:
            t = b(t)
        t = self.ln(t)
        return t[:, 1:], t[:, 0]

    def saidas(self, celulas, cls):
        """(plantios (B, n, 5), esperar (B, 1), V (B, 1))."""
        return self.cabeca_plantio(celulas), self.cabeca_esperar(cls), self.cabeca_v(cls)

    def partes(self, x):
        plantio, esperar, v = self.saidas(*self.codificar(self.tokens(x)))
        mapa = plantio.transpose(1, 2).reshape(len(x), N_PLANTAS, C.FAIXAS, C.COLUNAS)
        return mapa, esperar, v


class TransformerD128(Transformer):
    def __init__(self):
        super().__init__(d=128, cabecas=4)


# Só nas colunas. Com padding de zeros, um caminho que passa por uma coluna fora do
# tabuleiro não conta: (1, 2, 5) alcança −8..8 em tese, mas a célula 4 não chega às
# colunas 0 e 8. (1, 5, 2) é a única com a 1ª camada local (d = 1) em que toda célula
# alcança as 9 colunas da faixa por caminhos dentro do tabuleiro (tests/test_lab.py).
DILATACOES = (1, 5, 2)


class CNN1DDilatada(_CabecaCNN1D, _Partes):
    """E: a cnn1d com dilatação (1, d) nas colunas, d = 1, 5, 2. Mesmos parâmetros exatos."""

    CAMADAS = CNN1D.CAMADAS

    def __init__(self, canais=CANAIS):
        super().__init__()
        d1, d2, d3 = DILATACOES
        self.conv1 = nn.Conv2d(N_CANAIS_V2, canais, 3, padding=(1, d1), dilation=(1, d1))
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=(1, d2), dilation=(1, d2))
        self.conv3 = nn.Conv2d(canais, canais, 3, padding=(1, d3), dilation=(1, d3))
        self._criar_cabeca(canais)

    def partes(self, x):
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        return self._cabecas(h)


# Registry, como OTIMIZADORES e AGENDAS: nome → (classe, formato da observação).
ARQUITETURAS = {
    "cnn1": (CNN, "v1"),      # a de ontem, intacta
    "cnn1d": (CNN1D, "v2"),
    "cnn2": (CNN2, "v2"),
    "cnn3": (CNN3, "v2"),
    "atencao_antes": (AtencaoAntes, "v2"),        # B
    "atencao_depois": (AtencaoDepois, "v2"),      # C
    "transformer": (Transformer, "v2"),           # D
    "transformer_d128": (TransformerD128, "v2"),  # D2
    "cnn1d_dilatada": (CNN1DDilatada, "v2"),      # E
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
