"""PPO com máscara de ações (clip, crítico V, GAE), k = 20, γ = 0,994.

    python -m rl.ppo --id ppo-sanidade --fases calmo --dificuldade 0.5 --passos 300000 --ate 15:55

Rede: o tronco e as cabeças da cnn1d (desafio3/modelo.py, observação v2): o mapa A (5 × 5 × 9)
+ A(esperar) são os logits da política; cabeca_v é o crítico V. Ações inválidas recebem logit
−1e9 (nunca sorteadas; a máscara sai dos canais 20–24 da observação, como na cnn1d).
Várias partidas em paralelo na CPU (passo a passo, inferência em lote). Só usa reset, step,
observar e mascara do Ambiente.

Registro por atualização em rl/logs/<id>/registro.csv: retorno médio e vitórias dos episódios
que acabaram na coleta, entropia, KL aproximada, fração cortada, perda do V, passos/s.
Avaliação gulosa (a ação mais provável) com rl.avaliar, nas sementes do DQN.
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from desafio3.modelo import ARQUITETURAS, CNN1D, mascara_da_entrada
from desafio3.treino import AGENDAS, OTIMIZADORES
from desafio3.treino import Config as ConfigProxy
from lab.processos import rss_mb
from pvz.ambiente import Ambiente
from pvz.divisao import FASES_TESTE, FASES_TREINO

from . import avaliar

PASTA_LOGS = Path(__file__).parent / "logs"

# As receitas vêm do proxy supervisionado (desafio3/tabelas.md). "atual" é o que o PPO
# usava antes de existir esta opção: Adam 2,5e-4 constante.
RECEITAS = {
    "atual":     dict(otimizador="adam",    lr=2.5e-4, agenda="constante"),
    "vencedora": dict(otimizador="sgd",     lr=3e-2,   agenda="cosseno"),
    "rmsprop":   dict(otimizador="rmsprop", lr=3e-4,   agenda="warmup_cosseno"),
    "muon":      dict(otimizador="muon",    lr=1e-3, lr_adamw=3e-4, agenda="warmup_cosseno"),
    # lab (LAB.md, v2b): a "atual" com η decaindo linearmente até 0 no orçamento todo
    "atual_linear": dict(otimizador="adam", lr=2.5e-4, agenda="constante"),
}
# lab: as execuções de referência da receita atual (ppo-fases-s0 em CUDA e as sementes 1–7
# em CPU) rodaram ANTES do mecanismo RECEITAS, com Adam ε = 1e-5. O _adam do desafio3 usa o
# ε padrão (1e-8), e com ele o início da ppo-fases-s0 não se reproduz. Nas receitas "atual*"
# o lab volta ao ε = 1e-5 (reproduz a ppo-fases-s0 bit a bit; teste em lab/repro.py).
EPS_ADAM_ATUAL = 1e-5
AGENDA_LAB = {"atual_linear": lambda t, total: max(0.0, 1.0 - t / total)}


@dataclass
class Config:
    semente: int = 0
    passos: int = 300_000
    k_ticks: int = 20
    gamma: float = 0.994
    lam: float = 0.95              # GAE
    envs: int = 16                 # partidas em paralelo
    rollout: int = 128             # passos por partida em cada coleta (16 × 128 = 2.048)
    epocas: int = 4
    minilote: int = 512
    lr: float = 2.5e-4
    clip: float = 0.2
    coef_v: float = 0.5
    coef_ent: float = 0.01
    max_norma: float = 0.5
    fases: str = "treino"          # "treino" ou classes separadas por vírgula ("calmo")
    dificuldade: float = 1.0
    sementes_treino: tuple = (1000, 10_000)
    aval_cada: int = 20_000
    aval_sementes: tuple = (300, 303)
    aval_final_sementes: tuple = (300, 320)
    ate: str = ""                  # "HH:MM": para de coletar nesse horário
    aval_sorteio: bool = False     # avaliar também com sorteio (e contar as ações)
    continuar: bool = False        # retoma do ckpt.pt da mesma pasta, se existir
    receita: str = "atual"         # otimizador + agenda (RECEITAS); "atual" = Adam 2,5e-4 constante
    arquitetura: str = "cnn1d"     # lab: braço (desafio3.modelo.ARQUITETURAS); cnn1d = Politica intacta
    arquivo_parar: str = ""        # lab: se esse arquivo existir, para no fim da atualização (com ckpt)
    ckpt_cada: int = 250_000       # lab: checkpoint automático a cada N passos (0 = só no PARAR e no fim)

    def atualizacoes(self):
        """Passos do otimizador no orçamento todo (para a agenda saber o total)."""
        coletas = self.passos // (self.envs * self.rollout)
        por_coleta = self.epocas * -(-(self.envs * self.rollout) // self.minilote)
        return max(1, coletas * por_coleta)

    def proxy(self):
        """Config do desafio3 com a receita: alimenta OTIMIZADORES e AGENDAS."""
        return ConfigProxy(**RECEITAS[self.receita], semente=self.semente,
                           passos=self.atualizacoes(), k_val=10 ** 12, rascunho=True)

    def lista_fases(self):
        if self.fases == "treino":
            return list(FASES_TREINO)
        fase = tuple(c.strip() for c in self.fases.split(","))
        assert fase not in FASES_TESTE
        return [fase]


class Politica(nn.Module):
    """A cnn1d com as cabeças lidas como política (logits = A) e crítico (V)."""

    def __init__(self):
        super().__init__()
        self.rede = CNN1D()

    def forward(self, x):
        """(logits mascarados (B, 226), V (B,))."""
        r = self.rede
        h = r.tronco(x)[-1]
        g = h.mean(dim=(2, 3))
        logits = torch.cat([r.cabeca_q(h).flatten(1), r.cabeca_esperar(g)], dim=1)
        logits = logits.masked_fill(~mascara_da_entrada(x), -1e9)
        return logits, r.cabeca_v(g)[:, 0]


class PoliticaArq(nn.Module):
    """Lab: um braço novo (partes(x) → mapa A, A(esperar), V) lido como política e crítico."""

    def __init__(self, nome):
        super().__init__()
        self.rede = ARQUITETURAS[nome][0]()

    def forward(self, x):
        mapa, esperar, v = self.rede.partes(x)
        logits = torch.cat([mapa.flatten(1), esperar], dim=1)
        logits = logits.masked_fill(~mascara_da_entrada(x), -1e9)
        return logits, v[:, 0]


def criar_politica(arquitetura):
    return Politica() if arquitetura == "cnn1d" else PoliticaArq(arquitetura)


class Gulosa(nn.Module):
    """Para a avaliação: devolve os logits (o argmax mascarado é a ação mais provável)."""

    def __init__(self, pol):
        super().__init__()
        self.pol = pol

    def forward(self, x):
        return self.pol(x)[0]


def gae(r, v, fim, v_ult, gamma, lam):
    """Vantagens e retornos-alvo. r, v, fim: (T, N); v_ult: (N,) o V do estado seguinte ao
    último. fim[t] = o passo t encerrou a partida (sem bootstrap; a seguinte é outra)."""
    T = len(r)
    adv = np.zeros_like(r)
    ult = np.zeros_like(v_ult)
    for t in range(T - 1, -1, -1):
        v_prox = v_ult if t == T - 1 else v[t + 1]
        nao_fim = 1.0 - fim[t]
        delta = r[t] + gamma * v_prox * nao_fim - v[t]
        ult = delta + gamma * lam * nao_fim * ult
        adv[t] = ult
    return adv, adv + v


@torch.no_grad()
def avaliar_politica(pol, lista, dev, dificuldade, k, sorteio, semente=0):
    """Joga as partidas de `lista` em lote: gulosa (argmax dos logits mascarados) ou com
    sorteio (amostra da política; gerador semeado). Devolve vitórias, dano, decisões e a
    contagem de ações por tipo (esperar e as 5 plantas)."""
    pol.eval()
    gen = torch.Generator(device=dev).manual_seed(semente)
    ambs, obs = [], []
    for fase, s in lista:
        a = Ambiente(k=k, formato="v2")
        obs.append(a.reset(fase, s, dificuldade))
        ambs.append(a)
    vivos, dano, venceu, dec = list(range(len(ambs))), [0.0] * len(ambs), [False] * len(ambs), [0] * len(ambs)
    cont = np.zeros(6, np.int64)   # esperar, girassol, ervilha, repetidora, batata, mina
    while vivos:
        logits, _ = pol(torch.from_numpy(np.stack([obs[i] for i in vivos])).to(dev))
        if sorteio:
            acoes = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen)[:, 0].tolist()
        else:
            acoes = logits.argmax(1).tolist()
        seguem = []
        for i, a in zip(vivos, acoes):
            cont[0 if a == 225 else 1 + a // 45] += 1
            obs[i], r, fim = ambs[i].step(a)
            dec[i] += 1
            dano[i] += r
            if fim:
                venceu[i] = r > 0
                dano[i] -= 1.0 if venceu[i] else -1.0
            else:
                seguem.append(i)
        vivos = seguem
    pol.train()
    return {"partidas": len(lista), "vitorias": float(np.mean(venceu)), "dano": float(np.mean(dano)),
            "decisoes": float(np.mean(dec)), "acoes": (cont / cont.sum()).tolist()}


class Partidas:
    """N partidas em paralelo; reinicia cada uma ao acabar."""

    def __init__(self, cfg, rng):
        self.cfg, self.rng, self.fases = cfg, rng, cfg.lista_fases()
        self.ambs = [Ambiente(k=cfg.k_ticks, formato="v2") for _ in range(cfg.envs)]
        self.obs = [self._novo(a) for a in self.ambs]
        self.ret = np.zeros(cfg.envs)

    def _novo(self, amb):
        fase = self.fases[self.rng.integers(len(self.fases))]
        return amb.reset(fase, int(self.rng.integers(*self.cfg.sementes_treino)), self.cfg.dificuldade)

    def passo(self, acoes):
        r, fim, terminados = np.zeros(len(acoes), np.float32), np.zeros(len(acoes), np.float32), []
        for i, (amb, a) in enumerate(zip(self.ambs, acoes)):
            assert amb.mascara()[a], "ação inválida sorteada"
            o, ri, f = amb.step(int(a))
            r[i], fim[i] = ri, f
            self.ret[i] += ri
            if f:
                terminados.append((self.ret[i], ri > 0))
                self.ret[i] = 0.0
                o = self._novo(amb)
            self.obs[i] = o
        return r, fim, terminados


class Treino:
    def __init__(self, cfg, pasta):
        self.cfg, self.pasta = cfg, pasta
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.semente)
        torch.set_num_threads(1)
        self.rng = np.random.default_rng(cfg.semente)
        self.pol = criar_politica(cfg.arquitetura).to(self.dev)
        # a receita vem do proxy; self.pol.rede tem os mesmos parâmetros de self.pol e
        # expõe conv1..conv3, que é o que o Muon precisa para separar as matrizes
        cp = cfg.proxy()
        if cfg.receita.startswith("atual"):
            self.opts = [torch.optim.Adam(self.pol.rede.parameters(), lr=cp.lr, eps=EPS_ADAM_ATUAL)]
        else:
            self.opts = OTIMIZADORES[cp.otimizador](self.pol.rede, cp)
        agenda = AGENDA_LAB.get(cfg.receita, AGENDAS[cp.agenda])
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda t: agenda(t, cp.passos))
                       for o in self.opts]
        self.partidas = Partidas(cfg, self.rng)
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.semente)
        self.passo, self.linhas, self.avaliacoes = 0, [], []
        self.melhor = -1.0          # melhor vitória gulosa vista nas avaliações periódicas
        self.t0 = time.time()

    # ---------------- checkpoint: parar e retomar (PASSAGEM 5.2.1)
    def salvar_ckpt(self):
        torch.save({"pol": self.pol.state_dict(), "opt": [o.state_dict() for o in self.opts],
                    "sched": [s.state_dict() for s in self.scheds],
                    "passo": self.passo, "melhor": self.melhor,
                    "linhas": self.linhas, "avaliacoes": self.avaliacoes,
                    "rng": self.rng.bit_generator.state, "gen": self.gen.get_state(),
                    # lab: retomada exata (partidas em voo e geradores globais do torch)
                    "partidas": {"ambs": self.partidas.ambs, "obs": self.partidas.obs,
                                 "ret": self.partidas.ret},
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state() if self.dev.type == "cuda" else None,
                    "segundos": time.time() - self.t0},
                   self.pasta / "ckpt.tmp")
        # lab: troca atômica, para uma morte no meio da gravação não estragar o ckpt anterior
        os.replace(self.pasta / "ckpt.tmp", self.pasta / "ckpt.pt")
        (self.pasta / "ckpt.json").write_text(json.dumps({"passo": self.passo}), encoding="utf-8")

    def carregar_ckpt(self):
        """Retoma de onde parou. Devolve False se não houver checkpoint."""
        caminho = self.pasta / "ckpt.pt"
        if not caminho.exists():
            return False
        c = torch.load(caminho, map_location=self.dev, weights_only=False)
        self.pol.load_state_dict(c["pol"])
        for o, e in zip(self.opts, c["opt"]): o.load_state_dict(e)
        for s, e in zip(self.scheds, c.get("sched", [])): s.load_state_dict(e)
        self.passo, self.melhor = c["passo"], c["melhor"]
        self.linhas, self.avaliacoes = c["linhas"], c["avaliacoes"]
        self.rng.bit_generator.state = c["rng"]
        self.gen.set_state(c["gen"].cpu())   # lab: o map_location leva o estado para a GPU
        if "partidas" in c:      # lab: as partidas em voo continuam de onde estavam
            p = c["partidas"]
            self.partidas.ambs, self.partidas.obs, self.partidas.ret = p["ambs"], p["obs"], p["ret"]
            torch.set_rng_state(c["torch_cpu"].cpu())
            if c["torch_cuda"] is not None and self.dev.type == "cuda":
                torch.cuda.set_rng_state(c["torch_cuda"].cpu())
            self.t0 = time.time() - c["segundos"]
        else:
            self.partidas = Partidas(self.cfg, self.rng)   # as 16 partidas em voo recomeçam
        print(f"retomando de {caminho} no passo {self.passo}", flush=True)
        return True

    @torch.no_grad()
    def coletar(self):
        c, P = self.cfg, self.partidas
        T, N = c.rollout, c.envs
        X = torch.zeros((T, N, 25, 5, 9), device=self.dev)
        A = torch.zeros((T, N), dtype=torch.long, device=self.dev)
        LP, V = torch.zeros((T, N), device=self.dev), np.zeros((T, N), np.float32)
        R, F = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
        terminados = []
        for t in range(T):
            x = torch.from_numpy(np.stack(P.obs)).to(self.dev)
            logits, v = self.pol(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            X[t], A[t], LP[t], V[t] = x, a, dist.log_prob(a), v.cpu().numpy()
            R[t], F[t], fins = P.passo(a.cpu().numpy())
            terminados += fins
            self.passo += N
        _, v_ult = self.pol(torch.from_numpy(np.stack(P.obs)).to(self.dev))
        adv, ret = gae(R, V, F, v_ult.cpu().numpy(), c.gamma, c.lam)
        return X, A, LP, torch.from_numpy(adv).to(self.dev), torch.from_numpy(ret).to(self.dev), terminados

    def atualizar(self, X, A, LP, adv, ret):
        c = self.cfg
        X, A, LP = X.flatten(0, 1), A.flatten(), LP.flatten()
        adv, ret = adv.flatten(), ret.flatten()
        stats = {"entropia": [], "kl": [], "cortada": [], "perda_v": [], "perda_pi": []}
        for _ in range(c.epocas):
            perm = torch.randperm(len(A), device=self.dev, generator=self.gen)
            for i in range(0, len(A), c.minilote):
                idx = perm[i:i + c.minilote]
                logits, v = self.pol(X[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[idx])
                razao = torch.exp(lp - LP[idx])
                ad = adv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                perda_pi = -torch.min(razao * ad, razao.clamp(1 - c.clip, 1 + c.clip) * ad).mean()
                perda_v = 0.5 * (v - ret[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                perda = perda_pi + c.coef_v * perda_v - c.coef_ent * ent
                for o in self.opts: o.zero_grad(set_to_none=True)
                perda.backward()
                nn.utils.clip_grad_norm_(self.pol.parameters(), c.max_norma)
                for o in self.opts: o.step()
                for s in self.scheds: s.step()
                with torch.no_grad():
                    log_r = lp - LP[idx]
                    stats["kl"].append(float(((razao - 1) - log_r).mean()))
                    stats["cortada"].append(float(((razao - 1).abs() > c.clip).float().mean()))
                    stats["entropia"].append(float(ent))
                    stats["perda_v"].append(float(perda_v))
                    stats["perda_pi"].append(float(perda_pi))
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def avaliar(self, sementes, sorteio=False):
        lista = avaliar.partidas(range(*sementes), self.cfg.lista_fases())
        if self.cfg.aval_sorteio or sorteio:
            return avaliar_politica(self.pol, lista, self.dev, self.cfg.dificuldade,
                                    self.cfg.k_ticks, sorteio)
        return avaliar.avaliar_rede(Gulosa(self.pol), lista, self.dev, self.cfg.dificuldade,
                                    k=self.cfg.k_ticks, formato="v2")

    def avaliacao(self, sementes=None, final=False):
        modos = (False, True) if self.cfg.aval_sorteio else (False,)
        for sorteio in modos:
            r = self.avaliar(sementes or self.cfg.aval_sementes, sorteio)
            linha = {"passo": self.passo, "modo": "sorteio" if sorteio else "gulosa",
                     "final": int(final), "partidas": r["partidas"], "vitorias": r["vitorias"],
                     "dano": r["dano"], "decisoes": r.get("decisoes", "")}
            for nome, f in zip(("esperar", "girassol", "ervilha", "repetidora", "batata", "mina"),
                               r.get("acoes", [""] * 6)):
                linha[f"acao_{nome}"] = f
            self.avaliacoes.append(linha)
            if not final and not sorteio and r["vitorias"] > self.melhor:
                self.melhor = r["vitorias"]
                torch.save(self.pol.state_dict(), self.pasta / "melhor.pt")
            print(f"[{'final' if final else 'aval'}] passo {self.passo} {linha['modo']}: vitórias "
                  f"{100 * r['vitorias']:.1f}% ({r['partidas']} partidas), dano {r['dano']:.3f}"
                  + (f", ações {[round(100 * x, 1) for x in r['acoes']]}" if "acoes" in r else ""),
                  flush=True)
        self.gravar()
        return r

    def gravar(self):
        for nome, linhas in (("registro.csv", self.linhas), ("avaliacoes.csv", self.avaliacoes)):
            if linhas:
                with open(self.pasta / nome, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(linhas[0]))
                    w.writeheader()
                    w.writerows(linhas)

    def acabou_o_tempo(self):
        if not self.cfg.ate:
            return False
        h, m = map(int, self.cfg.ate.split(":"))
        agora = datetime.now()
        return (agora.hour, agora.minute) >= (h, m)

    def rodar(self):
        c = self.cfg
        retomou = c.continuar and self.carregar_ckpt()
        if not retomou:
            self.avaliacao()
        prox_aval = (self.passo // c.aval_cada + 1) * c.aval_cada
        prox_ckpt = (self.passo // c.ckpt_cada + 1) * c.ckpt_cada if c.ckpt_cada else float("inf")
        self.rss_inicio = rss_mb()
        motivo = "orçamento"
        while self.passo < c.passos:
            if self.acabou_o_tempo():
                motivo = f"horário ({c.ate})"
                break
            if c.arquivo_parar and Path(c.arquivo_parar).exists():
                self.salvar_ckpt()
                self.gravar()
                print(f"[parada] {c.arquivo_parar} existe: checkpoint no passo {self.passo}; "
                      f"para retomar, apague o arquivo e chame de novo", flush=True)
                return {"passos": self.passo, "motivo": "parada"}
            X, A, LP, adv, ret, fins = self.coletar()
            s = self.atualizar(X, A, LP, adv, ret)
            linha = {"atualizacao": len(self.linhas), "passo": self.passo,
                     "episodios": len(fins),
                     "retorno_medio": float(np.mean([f[0] for f in fins])) if fins else "",
                     "vitorias": float(np.mean([f[1] for f in fins])) if fins else "",
                     **s, "passos_por_s": self.passo / (time.time() - self.t0)}
            self.linhas.append(linha)
            if len(self.linhas) % 10 == 0:
                print(f"atualização {len(self.linhas)} passo {self.passo} vitórias(coleta) "
                      f"{linha['vitorias']} entropia {s['entropia']:.3f} kl {s['kl']:.4f} "
                      f"cortada {s['cortada']:.3f} perda_v {s['perda_v']:.4f} "
                      f"{linha['passos_por_s']:.0f} passos/s, RAM {rss_mb():.0f} MB", flush=True)
            if self.passo >= prox_aval:
                self.avaliacao()
                prox_aval += c.aval_cada
            if self.passo >= prox_ckpt:    # lab: depois da avaliação do mesmo passo, se houver
                self.salvar_ckpt()
                print(f"[ckpt] checkpoint automático no passo {self.passo}", flush=True)
                prox_ckpt += c.ckpt_cada
        if c.aval_sorteio:
            self.avaliacao(c.aval_final_sementes, final=True)
            r = next(a for a in self.avaliacoes if a["final"] and a["modo"] == "gulosa")
        else:
            r = self.avaliar(c.aval_final_sementes)
        print(f"[final] gulosa: vitórias {100 * r['vitorias']:.1f}% em {r['partidas']} partidas "
              f"(parou por {motivo}, passo {self.passo})", flush=True)
        self.gravar()
        torch.save(self.pol.state_dict(), self.pasta / "modelo.pt")
        self.salvar_ckpt()
        res = {"passos": self.passo, "motivo": motivo, "vitorias_final": r["vitorias"],
               "melhor_periodica": self.melhor,
               "dano_final": r["dano"], "partidas_final": r["partidas"],
               "minutos": round((time.time() - self.t0) / 60, 1),
               "passos_por_s": self.passo / (time.time() - self.t0),
               # lab: RAM (RSS, MB) deste processo no começo e no fim desta tentativa, e o pico
               "rss_inicio_mb": self.rss_inicio, "rss_fim_mb": rss_mb(), "rss_pico_mb": rss_mb(pico=True)}
        (self.pasta / "resultado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
        return res


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--id", required=True)
    p.add_argument("--json", help="campos extras da Config, em JSON")
    for nome, tipo in (("semente", int), ("passos", int), ("envs", int), ("rollout", int),
                       ("fases", str), ("dificuldade", float), ("aval_cada", int), ("ate", str)):
        p.add_argument("--" + nome.replace("_", "-"), type=tipo)
    p.add_argument("--continuar", action="store_true", default=None,
                   help="retoma do ckpt.pt da pasta do log")
    p.add_argument("--arquitetura", choices=list(ARQUITETURAS))
    p.add_argument("--arquivo-parar")
    p.add_argument("--receita", choices=list(RECEITAS),
                   help="otimizador + agenda vindos do proxy (padrão: atual)")
    a = vars(p.parse_args())
    id_, extra = a.pop("id"), json.loads(a.pop("json") or "{}")
    cfg = Config(**{k: v for k, v in a.items() if v is not None}, **extra)
    cfg.aval_sementes, cfg.aval_final_sementes = tuple(cfg.aval_sementes), tuple(cfg.aval_final_sementes)
    pasta = PASTA_LOGS / id_
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "config.json").write_text(json.dumps({"id": id_, "config": asdict(cfg)}, indent=2,
                                                  ensure_ascii=False), encoding="utf-8")
    Treino(cfg, pasta).rodar()


if __name__ == "__main__":
    main()
