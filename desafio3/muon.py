"""Muon (Jordan, 2024), implementação própria, com a divisão Muon + AdamW do enunciado.

Para cada matriz de pesos, o Muon acumula o momento M e, em vez de usar M como passo,
usa a versão ortogonalizada U·Vᵀ (se M = U·Σ·Vᵀ), calculada com 5 iterações de
Newton–Schulz. Todas as direções singulares recebem o mesmo tamanho de passo.

Divisão dos parâmetros (enunciado, seção 2: Muon só nas matrizes das camadas ocultas;
vieses e camada de saída continuam com AdamW):
- Muon: pesos de conv1, conv2 e conv3, cada kernel achatado em (C_out, C_in·k·k).
- AdamW: os vieses de todas as camadas, cabeca_q (é a camada de saída: o mapa Q) e
  cabeca_esperar (a outra saída).
"""
import torch

COEFICIENTES = (3.4445, -4.7750, 2.0315)   # os do enunciado
CAMADAS_MUON = ("conv1", "conv2", "conv3")


def ortogonaliza(G, passos=5):
    """≈ U·Vᵀ de G = U·Σ·Vᵀ, por Newton–Schulz (em float32). G é 2-D."""
    a, b, c = COEFICIENTES
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposta = X.size(0) > X.size(1)
    if transposta:
        X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposta else X


class Muon(torch.optim.Optimizer):
    """Momento estilo Nesterov, passo ortogonalizado e weight decay desacoplado.

    Para cada parâmetro W (qualquer formato; é achatado para (linhas, resto)):
        M ← β·M + (1 − β)·g
        U ← (1 − β)·g + β·M                         (Nesterov)
        W ← W·(1 − η·λ) − η·√max(1, linhas/colunas)·ortogonaliza(U)
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, passos_ns=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      passos_ns=passos_ns))

    @torch.no_grad()
    def step(self, closure=None):
        perda = None
        if closure is not None:
            with torch.enable_grad():
                perda = closure()
        for grupo in self.param_groups:
            lr, beta, wd = grupo["lr"], grupo["momentum"], grupo["weight_decay"]
            for p in grupo["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                estado = self.state[p]
                if "momento" not in estado:
                    estado["momento"] = torch.zeros_like(g)
                m = estado["momento"]
                m.lerp_(g, 1 - beta)
                u = g.lerp(m, beta)
                matriz = u.reshape(u.shape[0], -1)
                passo = ortogonaliza(matriz, grupo["passos_ns"]).reshape(p.shape).to(p.dtype)
                escala = max(1.0, matriz.shape[0] / matriz.shape[1]) ** 0.5
                p.mul_(1 - lr * wd)
                p.add_(passo, alpha=-lr * escala)
        return perda


def dividir(modelo):
    """(parâmetros do Muon, parâmetros do AdamW). Cada parâmetro cai em exatamente um."""
    muon = [getattr(modelo, n).weight for n in CAMADAS_MUON]
    ids = {id(p) for p in muon}
    return muon, [p for p in modelo.parameters() if id(p) not in ids]


def criar(modelo, cfg):
    """Fábrica do registry: [Muon (η = cfg.lr), AdamW (η = cfg.lr_adamw)], ambos com cfg.weight_decay."""
    if cfg.lr_adamw is None:
        raise ValueError("muon precisa de lr_adamw (o η da parte AdamW)")
    p_muon, p_adamw = dividir(modelo)
    return [Muon(p_muon, lr=cfg.lr, momentum=0.95, weight_decay=cfg.weight_decay),
            torch.optim.AdamW(p_adamw, lr=cfg.lr_adamw, weight_decay=cfg.weight_decay)]
