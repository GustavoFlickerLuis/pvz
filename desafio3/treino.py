"""Treino instrumentado do proxy supervisionado (Desafio 3).

Regressão de Q(s, a) para o retorno de Monte Carlo G: huber(q[ação], G), só na ação
tomada. O laço é o mesmo para todos os otimizadores; eles e as agendas entram por
registries (OTIMIZADORES, AGENDAS) e a instrumentação fica em Registro.

    python -m desafio3.treino --otimizador adam --lr 1e-4 --agenda constante --semente 0 \\
        --passos 10000 --etapa baseline
    python -m desafio3.treino --config cfg.json [--rascunho]

Sem --rascunho, recusa rodar com alterações não commitadas (fora de desafio3/logs/ e
desafio3/experimentos.csv). Rascunhos vão para desafio3/logs/rascunho/ e não entram
no experimentos.csv.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # antes do torch, para o determinismo

import argparse
import csv
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pvz.ambiente import ESPERAR

from . import muon
from .dados import PASTA as PASTA_DADOS
from .dados import carregar, commit_atual, hash_dataset
from .modelo import CAMADAS, CONVS, CNN

RAIZ = Path(__file__).parent
PASTA_LOGS = RAIZ / "logs"
PASTA_PESOS = RAIZ / "pesos"   # no .gitignore
EXPERIMENTOS = RAIZ / "experimentos.csv"
ETAPAS = ("baseline", "varredura_lr", "agendas", "confirmacao", "extra")
WD_PADRAO = {"adamw": 0.01, "muon": 0.01}


# ---------------------------------------------------------------- otimizadores
# Fábrica: (modelo, cfg) -> lista de torch.optim.Optimizer (o Muon vai devolver 2).
def _sgd(modelo, cfg, nesterov=False):
    return [torch.optim.SGD(modelo.parameters(), lr=cfg.lr, momentum=cfg.momento,
                            nesterov=nesterov, weight_decay=cfg.weight_decay)]


def _adam(modelo, cfg):
    return [torch.optim.Adam(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)]


def _adamw(modelo, cfg):
    return [torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)]


def _rmsprop(modelo, cfg):
    return [torch.optim.RMSprop(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)]


OTIMIZADORES = {
    "sgd": _sgd,
    "nesterov": partial(_sgd, nesterov=True),
    "adam": _adam,
    "adamw": _adamw,
    "rmsprop": _rmsprop,
    "muon": muon.criar,   # devolve [Muon, AdamW]
}


# ---------------------------------------------------------------- agendas
# fator(t, total) ∈ [0, 1], multiplicado pelo η base (LambdaLR). t vai de 0 a total − 1.
WARMUP = 0.05       # fração dos passos
DECAIMENTO = 0.20   # wsd: fração final com decaimento linear


def _passos_warmup(total):
    return max(1, round(WARMUP * total))


def constante(t, total):
    return 1.0


def cosseno(t, total):
    return 0.5 * (1 + math.cos(math.pi * t / total))


def warmup_cosseno(t, total):
    w = _passos_warmup(total)
    if t < w:
        return (t + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (t - w) / (total - w)))


def wsd(t, total):
    """Warmup, estável, decaimento linear até 0 nos últimos 20%."""
    w = _passos_warmup(total)
    d = max(1, round(DECAIMENTO * total))
    if t < w:
        return (t + 1) / w
    if t >= total - d:
        return (total - t) / d
    return 1.0


AGENDAS = {
    "constante": constante,
    "cosseno": cosseno,
    "warmup_cosseno": warmup_cosseno,
    "wsd": wsd,
}


# ---------------------------------------------------------------- configuração
@dataclass
class Config:
    otimizador: str = "adam"
    lr: float = 1e-4             # no muon, o η das matrizes
    lr_adamw: float = None       # muon: η da parte AdamW (vieses e cabeças)
    agenda: str = "constante"
    semente: int = 0
    passos: int = 10_000         # orçamento fixo de todas as comparações (CLAUDE.md)
    etapa: str = "baseline"
    lote: int = 256
    weight_decay: float = None   # None: padrão do otimizador (0,01 no adamw e no muon, 0 nos outros)
    momento: float = 0.9         # sgd e nesterov
    clip: float = 10.0           # norma global do gradiente
    delta_huber: float = 1.0
    k_registro: int = 25
    k_val: int = 250
    lote_fixo: int = 1024        # amostras fixas para ReLUs mortas (val) e huber_treino
    amostras: int = None         # usa só um subconjunto do treino (verificação de sobreajuste)
    rascunho: bool = False
    salvar_pesos: bool = False   # grava o state_dict final em desafio3/pesos/<id>.pt
    dados: str = str(PASTA_DADOS)

    def __post_init__(self):
        if self.weight_decay is None:
            self.weight_decay = WD_PADRAO.get(self.otimizador, 0.0)
        if self.otimizador not in OTIMIZADORES:
            raise ValueError(f"otimizador desconhecido: {self.otimizador}")
        if self.agenda not in AGENDAS:
            raise ValueError(f"agenda desconhecida: {self.agenda}")
        if self.etapa not in ETAPAS:
            raise ValueError(f"etapa desconhecida: {self.etapa}")


# ---------------------------------------------------------------- perda
def perda_q(q, acao, alvo, delta):
    """Huber só na ação tomada. One-hot em vez de gather: o backward é determinístico na GPU."""
    qa = (q * F.one_hot(acao, q.shape[1])).sum(dim=1)
    return F.huber_loss(qa, alvo, delta=delta)


@torch.no_grad()
def huber_medio(modelo, obs, acao, alvo, delta, bloco=8192, grupo=None):
    """Huber médio. Com `grupo` (bool por amostra), devolve (total, dentro, fora do grupo).

    O total é sempre somado do mesmo jeito, para o val_final não mudar entre versões.
    """
    soma = soma_dentro = soma_fora = 0.0
    for i in range(0, len(alvo), bloco):
        q = modelo(obs[i:i + bloco].float())
        qa = q.gather(1, acao[i:i + bloco, None]).squeeze(1)
        soma += F.huber_loss(qa, alvo[i:i + bloco], delta=delta, reduction="sum").item()
        if grupo is not None:
            perdas = F.huber_loss(qa, alvo[i:i + bloco], delta=delta, reduction="none")
            g = grupo[i:i + bloco]
            soma_dentro += (perdas * g).sum().item()
            soma_fora += (perdas * ~g).sum().item()
    if grupo is None:
        return soma / len(alvo)
    n = int(grupo.sum())
    return soma / len(alvo), soma_dentro / n, soma_fora / (len(alvo) - n)


# ---------------------------------------------------------------- instrumentação
def _norma(tensores):
    return math.sqrt(sum(float(t.pow(2).sum()) for t in tensores))


class Registro:
    """Mede o treino. O laço só chama antes_do_passo() e depois_do_passo(t, perda, norma_g).

    A cada k_registro passos: perda do lote; ‖g‖ global antes do corte e se houve
    corte; ‖g‖ por camada (antes do corte); ‖θ‖ global e por camada;
    ρ = ‖θ_{t+1} − θ_t‖ / ‖θ_t‖ global e por camada; η de cada otimizador.
    A cada k_val passos (e depois do último): Huber de validação (total e separada
    entre amostras com ação ESPERAR e de plantio), Huber num lote fixo do treino e, num lote fixo da validação, a fração de filtros mortos (nunca ativam)
    e de ativações zero de cada conv.
    """

    def __init__(self, modelo, opts, cfg, dados):
        self.modelo, self.opts, self.cfg, self.d = modelo, opts, cfg, dados
        self.camadas = {n: list(getattr(modelo, n).parameters()) for n in CAMADAS}
        self.linhas = []
        self.t = 0
        self.linha = {}
        self.perda_inicial = None
        self.val = []          # (passo, huber de validação)
        self.status = None

    def colunas(self):
        c = ["passo", "perda_lote", "huber_treino", "huber_val", "huber_val_esperar",
             "huber_val_plantio", "norma_g", "cortou", "norma_theta", "rho_global"]
        c += [f"g_{n}" for n in CAMADAS] + [f"theta_{n}" for n in CAMADAS]
        c += [f"rho_{n}" for n in CAMADAS] + [f"eta_{i}" for i in range(len(self.opts))]
        c += [f"mortos_{n}" for n in CONVS] + [f"zeros_{n}" for n in CONVS]
        return c

    def antes_do_passo(self):
        t, cfg = self.t, self.cfg
        self.linha = {}
        if t % cfg.k_val == 0:
            self._validar()
        if t % cfg.k_registro == 0:
            self.antes = {n: [p.detach().clone() for p in ps] for n, ps in self.camadas.items()}
            for i, o in enumerate(self.opts):
                self.linha[f"eta_{i}"] = o.param_groups[0]["lr"]

    def depois_do_passo(self, t, perda, norma_g):
        """Devolve False se o treino deve parar (divergiu)."""
        assert t == self.t
        self.t += 1
        cfg = self.cfg
        if t % cfg.k_registro == 0:
            if not self._medir(t, perda, norma_g):
                self.status = "divergiu"
                self._fechar_linha(t)
                return False
        self._fechar_linha(t)
        if self.t == cfg.passos:   # validação final, com os pesos depois do último passo
            self.linha = {}
            self._validar()
            self._fechar_linha(self.t)
        return True

    def _medir(self, t, perda, norma_g):
        perda, norma_g = float(perda), float(norma_g)
        if self.perda_inicial is None:
            self.perda_inicial = perda
        coef = min(1.0, self.cfg.clip / (norma_g + 1e-6))   # o mesmo do clip_grad_norm_
        l = self.linha
        l.update(passo=t, perda_lote=perda, norma_g=norma_g, cortou=int(norma_g > self.cfg.clip))
        dif2 = base2 = 0.0
        for n, ps in self.camadas.items():
            # os gradientes já foram cortados; divide pelo fator para ter a norma de antes
            l[f"g_{n}"] = _norma(p.grad for p in ps if p.grad is not None) / coef
            d2 = sum(float((p.detach() - a).pow(2).sum()) for p, a in zip(ps, self.antes[n]))
            b2 = sum(float(a.pow(2).sum()) for a in self.antes[n])
            l[f"theta_{n}"] = _norma(p.detach() for p in ps)
            l[f"rho_{n}"] = math.sqrt(d2) / math.sqrt(b2)
            dif2, base2 = dif2 + d2, base2 + b2
        l["norma_theta"] = _norma(p.detach() for ps in self.camadas.values() for p in ps)
        l["rho_global"] = math.sqrt(dif2) / math.sqrt(base2)
        return math.isfinite(perda) and perda <= 10 * self.perda_inicial

    @torch.no_grad()
    def _validar(self):
        d, delta = self.d, self.cfg.delta_huber
        v, v_esperar, v_plantio = huber_medio(self.modelo, d["val_obs"], d["val_acao"], d["val_G"],
                                              delta, grupo=d["val_esperar"])
        self.linha.update(huber_val=v, huber_val_esperar=v_esperar, huber_val_plantio=v_plantio)
        self.linha["huber_treino"] = huber_medio(self.modelo, d["fixo_treino_obs"],
                                                 d["fixo_treino_acao"], d["fixo_treino_G"], delta)
        for n, a in zip(CONVS, self.modelo.tronco(d["fixo_val_obs"].float())):
            self.linha[f"mortos_{n}"] = float(((a > 0).sum(dim=(0, 2, 3)) == 0).float().mean())
            self.linha[f"zeros_{n}"] = float((a == 0).float().mean())
        self.val.append((self.t, v, v_esperar, v_plantio))

    def _fechar_linha(self, passo):
        if self.linha:
            self.linha["passo"] = passo
            self.linhas.append(self.linha)
        self.linha = {}

    def resultado(self):
        passo_min, val_min, *_ = min(self.val, key=lambda pv: pv[1])
        val_0 = self.val[0][1]
        _, val_final, val_esperar, val_plantio = self.val[-1]
        status = self.status
        if status is None:
            status = "divergiu" if not math.isfinite(val_final) else (
                "estagnou" if val_final > 0.95 * val_0 else "ok")
        return {"val_final": val_final, "val_esperar": val_esperar, "val_plantio": val_plantio,
                "val_min": val_min, "passo_val_min": passo_min, "status": status}

    def salvar(self, caminho):
        colunas = self.colunas()
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colunas)
            w.writeheader()
            for l in self.linhas:
                w.writerow({c: (repr(l[c]) if isinstance(l.get(c), float) else l.get(c, ""))
                            for c in colunas})


# ---------------------------------------------------------------- dados na GPU
def preparar_dados(cfg, dev):
    treino, val = carregar("treino", cfg.dados), carregar("val", cfg.dados)
    if cfg.amostras is not None:
        idx = np.sort(np.random.default_rng(cfg.semente).choice(len(treino["G"]), cfg.amostras,
                                                                replace=False))
        treino = {c: v[idx] for c, v in treino.items()}
    # lotes fixos: a mesma escolha em todas as execuções (não depende da semente)
    rng = np.random.default_rng(12345)
    fixo_t = np.sort(rng.choice(len(treino["G"]), min(cfg.lote_fixo, len(treino["G"])), replace=False))
    fixo_v = np.sort(rng.choice(len(val["G"]), min(cfg.lote_fixo, len(val["G"])), replace=False))

    def gpu(a, tipo=None):
        t = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        return t if tipo is None else t.to(tipo)

    d = {}
    for nome, arr, idx in (("treino", treino, None), ("val", val, None),
                           ("fixo_treino", treino, fixo_t), ("fixo_val", val, fixo_v)):
        sel = (lambda a: a) if idx is None else (lambda a, i=idx: a[i])
        d[f"{nome}_obs"] = gpu(sel(arr["obs"]))                    # float16
        d[f"{nome}_acao"] = gpu(sel(arr["acao"]), torch.long)
        d[f"{nome}_G"] = gpu(sel(arr["G"]))
    d["val_esperar"] = d["val_acao"] == ESPERAR
    return d


# ---------------------------------------------------------------- execução
def fixar_sementes(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def alteracoes_pendentes():
    raiz = RAIZ.parent
    saida = subprocess.run(["git", "status", "--porcelain"], cwd=raiz, capture_output=True,
                           text=True, check=True).stdout
    ignorar = ("desafio3/logs/", "desafio3/experimentos.csv")
    return [l for l in saida.splitlines() if not l[3:].strip('"').startswith(ignorar)]


def treinar(cfg):
    pendentes = alteracoes_pendentes()
    if pendentes and not cfg.rascunho:
        raise SystemExit("há alterações não commitadas (use --rascunho para testar):\n"
                         + "\n".join(pendentes))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixar_sementes(cfg.semente)
    d = preparar_dados(cfg, dev)
    modelo = CNN().to(dev)
    opts = OTIMIZADORES[cfg.otimizador](modelo, cfg)
    agenda = AGENDAS[cfg.agenda]
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda t: agenda(t, cfg.passos)) for o in opts]
    gerador = torch.Generator(device=dev).manual_seed(cfg.semente)
    registro = Registro(modelo, opts, cfg, d)
    obs, acao, alvo = d["treino_obs"], d["treino_acao"], d["treino_G"]

    t0 = time.time()
    for t in range(cfg.passos):
        idx = torch.randint(len(alvo), (cfg.lote,), generator=gerador, device=dev)
        registro.antes_do_passo()
        perda = perda_q(modelo(obs[idx].float()), acao[idx], alvo[idx], cfg.delta_huber)
        for o in opts:
            o.zero_grad(set_to_none=True)
        perda.backward()
        norma_g = torch.nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip)   # norma antes do corte
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        if not registro.depois_do_passo(t, perda.detach(), norma_g):
            break
    if dev.type == "cuda":
        torch.cuda.synchronize()
    tempo = time.time() - t0
    return registro, tempo, dev, modelo


def anotar_experimento(linha):
    """Acrescenta uma linha ao experimentos.csv. Se as colunas mudaram, reescreve o arquivo
    com as colunas novas (vazias nas linhas antigas)."""
    linhas, colunas = [], list(linha)
    if EXPERIMENTOS.exists():
        with open(EXPERIMENTOS, newline="", encoding="utf-8") as f:
            leitor = csv.DictReader(f)
            linhas = list(leitor)
            colunas += [c for c in leitor.fieldnames if c not in colunas]
    with open(EXPERIMENTOS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colunas, restval="")
        w.writeheader()
        w.writerows(linhas + [linha])


def executar(cfg):
    agora = datetime.now()
    commit, sujo = commit_atual()
    aw = "" if cfg.lr_adamw is None else f"-aw{cfg.lr_adamw:g}"
    id_ = f"{agora:%Y%m%d-%H%M%S}-{cfg.otimizador}-lr{cfg.lr:g}{aw}-{cfg.agenda}-s{cfg.semente}"
    pasta = PASTA_LOGS / ("rascunho" if cfg.rascunho else "") / id_
    registro, tempo, dev, modelo = treinar(cfg)
    if cfg.salvar_pesos:
        PASTA_PESOS.mkdir(exist_ok=True)
        torch.save(modelo.state_dict(), PASTA_PESOS / f"{id_}.pt")
    res = {**registro.resultado(), "tempo_s": round(tempo, 1)}

    pasta.mkdir(parents=True, exist_ok=True)
    registro.salvar(pasta / "registro.csv")
    info = {"id": id_, "data": agora.isoformat(timespec="seconds"), "commit": commit,
            "commit_sujo": sujo, "hash_dataset": hash_dataset(cfg.dados), "config": asdict(cfg),
            "torch": torch.__version__,
            "dispositivo": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "resultado": res}
    with open(pasta / "config.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    if not cfg.rascunho:
        anotar_experimento({"id": id_, "data": info["data"], "commit": commit[:7],
                            "etapa": cfg.etapa, "otimizador": cfg.otimizador, "lr": cfg.lr,
                            "lr_adamw": "" if cfg.lr_adamw is None else cfg.lr_adamw,
                            "agenda": cfg.agenda, "semente": cfg.semente, "passos": cfg.passos,
                            "lote": cfg.lote, "weight_decay": cfg.weight_decay, **res})
    print(json.dumps({"pasta": str(pasta), **res}, ensure_ascii=False))
    return pasta, res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", help="json com campos da Config (os argumentos abaixo sobrepõem)")
    p.add_argument("--otimizador", choices=list(OTIMIZADORES))
    p.add_argument("--lr", type=float)
    p.add_argument("--lr-adamw", type=float, help="muon: η da parte AdamW")
    p.add_argument("--agenda", choices=list(AGENDAS))
    p.add_argument("--semente", type=int)
    p.add_argument("--passos", type=int)
    p.add_argument("--etapa", choices=ETAPAS)
    p.add_argument("--amostras", type=int, help="usa só N amostras do treino")
    p.add_argument("--rascunho", action="store_true", default=None)
    p.add_argument("--salvar-pesos", action="store_true", default=None)
    args = vars(p.parse_args())
    campos = {}
    caminho = args.pop("config")
    if caminho:
        with open(caminho, encoding="utf-8") as f:
            campos = json.load(f)
        campos = campos.get("config", campos)   # aceita também o config.json de um log
    nomes = {f.name for f in fields(Config)}
    campos = {k: v for k, v in campos.items() if k in nomes}
    campos.update({k: v for k, v in args.items() if v is not None})
    executar(Config(**campos))


if __name__ == "__main__":
    main()
