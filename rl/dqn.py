"""Double DQN com máscara (CONCEITO 2.5), retorno de n passos e receitas do proxy.

    python -m rl.dqn --receita baseline --semente 0 --passos 600000 [--id baseline-s0]

Uma execução vive em rl/logs/<id>/: config.json, registro.csv (instrumentação, a cada
50 atualizações), episodios.csv, avaliacoes.csv, velocidade.csv, progresso.json e, no
fim, resultado.json e modelo.pt. Checkpoints em rl/logs/<id>/ckpt/ (no .gitignore).

Parar e continuar: no fim de cada episódio, se existir rl/PARAR, grava um checkpoint
completo e sai com código 3. Chamar de novo com o mesmo --id retoma do checkpoint.
Também grava checkpoint a cada ~20 min. A retomada é exata (registro.csv idêntico).

Regra do CLAUDE.md: o agente só usa reset, step, observar e mascara do Ambiente.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # antes do torch, para o determinismo

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from desafio3.modelo import CNN
from desafio3.treino import AGENDAS, OTIMIZADORES, Registro, fixar_sementes, perda_q
from desafio3.treino import Config as ConfigProxy
from pvz.ambiente import ESPERAR, N_ACOES, N_CANAIS, Ambiente
from pvz import config as C
from pvz.divisao import FASES_TESTE, FASES_TREINO

from . import avaliar, status

RAIZ = Path(__file__).parent
PASTA_LOGS = RAIZ / "logs"
ARQ_PARAR = RAIZ / "PARAR"
CODIGO_PARADO = 3

# As receitas vêm prontas do proxy (desafio3/tabelas.md); nenhum ajuste novo.
RECEITAS = {
    "baseline": dict(otimizador="adam", lr=1e-4, agenda="constante"),
    "vencedora": dict(otimizador="sgd", lr=3e-2, agenda="cosseno"),
    "muon": dict(otimizador="muon", lr=1e-3, lr_adamw=3e-4, agenda="warmup_cosseno"),
}


@dataclass
class Config:
    receita: str = "baseline"
    semente: int = 0
    passos: int = 1_000_000          # orçamento em passos de jogo (decisões)
    gamma: float = 0.997
    n: int = 10                      # retorno de n passos
    lote: int = 256
    replay: int = 100_000
    inicio: int = 10_000             # passos de jogo antes da primeira atualização
    freq_atualizacao: int = 4        # 1 atualização a cada 4 passos de jogo
    alvo_cada: int = 2_000           # atualizações entre cópias da rede alvo
    eps_ini: float = 1.0
    eps_fim: float = 0.05
    eps_frac: float = 0.4            # fração do orçamento com ε decaindo
    p_esperar: float = 0.5           # na exploração: "esperar" com prob. 0,5, senão planta válida uniforme
    clip: float = 10.0
    delta_huber: float = 1.0
    k_registro: int = 50             # atualizações entre linhas do registro.csv
    aval_cada: int = 50_000          # passos de jogo entre avaliações periódicas
    aval_sementes: tuple = (300, 303)        # periódica: FASES_TREINO × 300–302 = 63 partidas
    aval_final_sementes: tuple = (300, 320)  # final: FASES_TREINO × 300–319 = 420 partidas
    sementes_treino: tuple = (1000, 10_000)
    fases: str = "treino"            # "treino" (FASES_TREINO) ou classes separadas por vírgula ("calmo")
    dificuldade: float = 1.0
    ckpt_minutos: float = 20.0

    def __post_init__(self):
        if self.receita not in RECEITAS:
            raise ValueError(f"receita desconhecida: {self.receita}")
        self.aval_sementes = tuple(self.aval_sementes)
        self.aval_final_sementes = tuple(self.aval_final_sementes)
        self.sementes_treino = tuple(self.sementes_treino)

    def lista_fases(self):
        if self.fases == "treino":
            return list(FASES_TREINO)
        fase = tuple(c.strip() for c in self.fases.split(","))
        assert fase not in FASES_TESTE
        return [fase]

    def atualizacoes(self):
        """Total de atualizações no orçamento: passos p ∈ [inicio, passos] com p % freq == 0."""
        f = self.freq_atualizacao
        return self.passos // f - (self.inicio - 1) // f

    def proxy(self):
        """Config do desafio3 com a receita: alimenta OTIMIZADORES e Registro."""
        return ConfigProxy(**RECEITAS[self.receita], semente=self.semente,
                           passos=self.atualizacoes(), clip=self.clip,
                           delta_huber=self.delta_huber, k_registro=self.k_registro,
                           k_val=10 ** 12, rascunho=True)


# ---------------------------------------------------------------- peças testáveis
class NPassos:
    """Junta as transições de 1 passo em transições de n passos.

    Cada (s, a, r) entra na fila. Com n itens, sai a mais antiga com
    R = Σ γ^i r_i, desconto γ^n e s' = estado atual. No fim do episódio, saem todas,
    com o retorno cortado no terminal e fim = True (sem bootstrap).
    Saída: tuplas (obs, acao, R, desconto, obs_prox, mascara_prox, fim).
    """

    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.fila = deque()

    def limpar(self):
        self.fila.clear()

    def _retorno(self):
        R, g = 0.0, 1.0
        for _, _, r in self.fila:
            R += g * r
            g *= self.gamma
        return R, g

    def adicionar(self, obs, acao, r, obs_prox, mascara_prox, fim):
        self.fila.append((obs, acao, r))
        saida = []
        if fim:
            while self.fila:
                R, g = self._retorno()
                o, a, _ = self.fila.popleft()
                saida.append((o, a, R, g, obs_prox, mascara_prox, True))
        elif len(self.fila) == self.n:
            R, g = self._retorno()
            o, a, _ = self.fila.popleft()
            saida.append((o, a, R, g, obs_prox, mascara_prox, False))
        return saida


def explorar(mascara, rng, p_esperar):
    """Ação de exploração: "esperar" com prob. p_esperar; senão uniforme entre as plantas válidas."""
    if rng.random() < p_esperar:
        return ESPERAR
    validas = np.flatnonzero(mascara[:ESPERAR])
    if len(validas) == 0:
        return ESPERAR
    return int(validas[rng.integers(len(validas))])


def argmax_mascarado(q, mascara):
    return q.masked_fill(~mascara, -math.inf).argmax(dim=-1)


def alvo_double(q_online_prox, q_alvo_prox, mascara_prox, R, desconto, fim):
    """y = R + desconto · Q_alvo(s', argmax_{a' válida} Q_online(s', a')); y = R se fim."""
    a = argmax_mascarado(q_online_prox, mascara_prox)
    v = q_alvo_prox.gather(1, a[:, None]).squeeze(1)
    return R + torch.where(fim, torch.zeros_like(v), desconto * v)


class Replay:
    """Memória circular. Observações em float16; guarda a máscara de s'."""

    CAMPOS = ("obs", "prox", "acao", "R", "desconto", "fim", "mascara")

    def __init__(self, capacidade):
        forma = (capacidade, N_CANAIS, C.FAIXAS, C.COLUNAS)
        self.obs = np.zeros(forma, np.float16)
        self.prox = np.zeros(forma, np.float16)
        self.acao = np.zeros(capacidade, np.int64)
        self.R = np.zeros(capacidade, np.float32)
        self.desconto = np.zeros(capacidade, np.float32)
        self.fim = np.zeros(capacidade, bool)
        self.mascara = np.zeros((capacidade, N_ACOES), bool)
        self.capacidade, self.pos, self.tamanho = capacidade, 0, 0

    def guardar(self, t):
        o, a, R, g, p, m, fim = t
        i = self.pos
        self.obs[i], self.acao[i], self.R[i], self.desconto[i] = o, a, R, g
        self.prox[i], self.mascara[i], self.fim[i] = p, m, fim
        self.pos = (i + 1) % self.capacidade
        self.tamanho = min(self.tamanho + 1, self.capacidade)

    def amostrar(self, rng, lote, dev):
        idx = rng.integers(self.tamanho, size=lote)
        return {c: torch.from_numpy(getattr(self, c)[idx]).to(dev) for c in self.CAMPOS}

    def salvar(self, pasta):
        np.savez(pasta / "replay.npz", pos=self.pos, tamanho=self.tamanho,
                 **{c: getattr(self, c)[:self.tamanho] for c in self.CAMPOS})

    def carregar(self, pasta):
        with np.load(pasta / "replay.npz") as z:
            self.pos, self.tamanho = int(z["pos"]), int(z["tamanho"])
            for c in self.CAMPOS:
                getattr(self, c)[:self.tamanho] = z[c]


class RegistroRL(Registro):
    """O Registro do desafio3, sem a validação supervisionada. Acrescenta, nas mesmas
    linhas: passo de jogo, ε e Q médio previsto (na ação tomada, no lote). A divergência é
    só perda não finita (no RL a perda TD cresce naturalmente com o alvo)."""

    EXTRAS = ("passo_jogo", "epsilon", "q_medio")

    def __init__(self, modelo, opts, cfg):
        super().__init__(modelo, opts, cfg, dados=None)

    def colunas(self):
        tira = {"huber_treino", "huber_val", "huber_val_esperar", "huber_val_plantio"}
        tira |= {c for c in super().colunas() if c.startswith(("mortos_", "zeros_"))}
        c = [c for c in super().colunas() if c not in tira]
        return c[:1] + list(self.EXTRAS) + c[1:]

    def _validar(self):
        pass

    def _medir(self, t, perda, norma_g):
        super()._medir(t, perda, norma_g)
        return math.isfinite(float(perda))

    def depois(self, t, perda, norma_g, extras):
        mede = t % self.cfg.k_registro == 0
        ok = self.depois_do_passo(t, perda, norma_g)
        if mede:
            self.linhas[-1].update(extras)
        return ok

    def estado(self):
        return {"linhas": self.linhas, "t": self.t, "perda_inicial": self.perda_inicial,
                "status": self.status}

    def restaurar(self, e):
        self.linhas, self.t = e["linhas"], e["t"]
        self.perda_inicial, self.status = e["perda_inicial"], e["status"]


# ---------------------------------------------------------------- treino
def _escrever_csv(caminho, linhas, colunas=None):
    colunas = colunas or (list(linhas[0]) if linhas else [])
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colunas)
        w.writeheader()
        w.writerows(linhas)


class Treino:
    def __init__(self, cfg, pasta):
        self.cfg, self.pasta = cfg, pasta
        self.ckpt = pasta / "ckpt"
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fixar_sementes(cfg.semente)
        torch.set_num_threads(1)   # inferência do ator na CPU: rápida e determinística
        self.rng = np.random.default_rng(cfg.semente)
        self.cproxy = cfg.proxy()
        self.online = CNN().to(self.dev)
        self.alvo = CNN().to(self.dev)
        self.alvo.load_state_dict(self.online.state_dict())
        self.alvo.requires_grad_(False)
        # cópia da online na CPU só para agir: uma inferência de 1 estado é ~2× mais rápida
        # na CPU do que na GPU (sem sincronização); é ressincronizada depois de cada atualização
        self.ator = CNN().requires_grad_(False)
        self.sincronizar_ator()
        self.opts = OTIMIZADORES[self.cproxy.otimizador](self.online, self.cproxy)
        agenda, total = AGENDAS[self.cproxy.agenda], cfg.atualizacoes()
        self.scheds = [torch.optim.lr_scheduler.LambdaLR(o, lambda t: agenda(t, total))
                       for o in self.opts]
        self.registro = RegistroRL(self.online, self.opts, self.cproxy)
        self.replay = Replay(cfg.replay)
        self.npassos = NPassos(cfg.n, cfg.gamma)
        self.fases = cfg.lista_fases()
        self.passo = 0          # passos de jogo
        self.atualizacoes = 0
        self.episodios = []     # por episódio
        self.avaliacoes = []    # (passo, vitórias, dano)
        self.velocidade = []
        self.status = "rodando"
        self.inicio_proc = time.time()
        self.passo_proc = 0     # passo no início deste processo (para passos/s)

    # -------------------------------------------------------------- política
    def epsilon(self):
        c = self.cfg
        frac = min(1.0, self.passo / (c.eps_frac * c.passos))
        return c.eps_ini + frac * (c.eps_fim - c.eps_ini)

    @torch.no_grad()
    def sincronizar_ator(self):
        for a, p in zip(self.ator.parameters(), self.online.parameters()):
            a.copy_(p)

    @torch.no_grad()
    def agir(self, obs, mascara):
        if self.rng.random() < self.epsilon():
            return explorar(mascara, self.rng, self.cfg.p_esperar)
        q = self.ator(torch.from_numpy(obs)[None])[0]
        return int(argmax_mascarado(q, torch.from_numpy(mascara)))

    # -------------------------------------------------------------- atualização
    def atualizar(self):
        c, t = self.cfg, self.atualizacoes
        b = self.replay.amostrar(self.rng, c.lote, self.dev)
        with torch.no_grad():
            prox = b["prox"].float()
            y = alvo_double(self.online(prox), self.alvo(prox), b["mascara"], b["R"],
                            b["desconto"], b["fim"])
        self.registro.antes_do_passo()
        q = self.online(b["obs"].float())
        perda = perda_q(q, b["acao"], y, c.delta_huber)
        for o in self.opts:
            o.zero_grad(set_to_none=True)
        perda.backward()
        norma_g = torch.nn.utils.clip_grad_norm_(self.online.parameters(), c.clip)
        for o in self.opts:
            o.step()
        for s in self.scheds:
            s.step()
        extras = {}
        if t % c.k_registro == 0:
            qa = q.detach().gather(1, b["acao"][:, None]).mean()
            extras = {"passo_jogo": self.passo, "epsilon": self.epsilon(), "q_medio": float(qa)}
            agora = time.time()
            self.velocidade.append({"atualizacao": t, "passo_jogo": self.passo,
                                    "passos_por_s": (self.passo - self.passo_proc)
                                    / max(1e-9, agora - self.inicio_proc)})
        ok = self.registro.depois(t, perda.detach(), norma_g, extras)
        self.sincronizar_ator()
        self.atualizacoes += 1
        if self.atualizacoes % c.alvo_cada == 0:
            self.alvo.load_state_dict(self.online.state_dict())
        return ok

    # -------------------------------------------------------------- avaliação
    def avaliar(self, sementes):
        lista = avaliar.partidas(range(*sementes), self.fases)
        return avaliar.avaliar_rede(self.online, lista, self.dev, self.cfg.dificuldade)

    def avaliacao_periodica(self):
        r = self.avaliar(self.cfg.aval_sementes)
        self.avaliacoes.append({"passo": self.passo, "partidas": r["partidas"],
                                "vitorias": r["vitorias"], "dano": r["dano"],
                                "decisoes": r["decisoes"]})
        print(f"[aval] passo {self.passo}: vitórias {100 * r['vitorias']:.1f}% "
              f"({r['partidas']} partidas), dano {r['dano']:.3f}", flush=True)
        self.gravar_progresso()

    # -------------------------------------------------------------- laço
    def jogar_episodio(self, fase, semente):
        """Um episódio de treino. Devolve "fim", "orcamento" (acabou no meio) ou "divergiu"."""
        c = self.cfg
        amb = Ambiente()
        obs = amb.reset(fase, semente, c.dificuldade)
        mascara = amb.mascara()
        self.npassos.limpar()
        retorno, decisoes, fim = 0.0, 0, False
        while not fim and self.passo < c.passos:
            acao = self.agir(obs, mascara)
            obs_p, r, fim = amb.step(acao)
            mascara_p = amb.mascara()
            for t in self.npassos.adicionar(obs, acao, r, obs_p, mascara_p, fim):
                self.replay.guardar(t)
            obs, mascara = obs_p, mascara_p
            retorno += r
            decisoes += 1
            self.passo += 1
            if self.passo >= c.inicio and self.passo % c.freq_atualizacao == 0:
                if not self.atualizar():
                    return "divergiu"
            if self.passo % c.aval_cada == 0:
                self.avaliacao_periodica()
        if not fim:
            return "orcamento"   # o episódio não conta
        self.episodios.append({"episodio": len(self.episodios), "passo_jogo": self.passo,
                               "fase": "+".join(fase), "semente": semente,
                               "retorno": retorno, "vitoria": int(r > 0),
                               "decisoes": decisoes, "epsilon": self.epsilon()})
        return "fim"

    def rodar(self):
        c = self.cfg
        ultimo_ckpt = time.time()
        if self.passo == 0:
            self.avaliacao_periodica()
        while self.passo < c.passos:
            fase = self.fases[self.rng.integers(len(self.fases))]
            semente = int(self.rng.integers(*c.sementes_treino))
            saida = self.jogar_episodio(fase, semente)
            if saida == "divergiu":
                self.status = "divergiu"
                print(f"perda não finita no passo {self.passo}", flush=True)
                return self.terminar()
            if saida == "orcamento":
                break
            if len(self.episodios) % 100 == 0:
                ult = self.episodios[-100:]
                print(f"episódio {len(self.episodios)} passo {self.passo} ε {self.epsilon():.3f} "
                      f"vitórias(100) {np.mean([e['vitoria'] for e in ult]):.2f}", flush=True)
            if ARQ_PARAR.exists():
                self.salvar_ckpt()
                self.status = "parado"
                self.gravar_progresso()
                print(f"rl/PARAR encontrado: checkpoint no passo {self.passo}; saindo", flush=True)
                return CODIGO_PARADO
            if time.time() - ultimo_ckpt > 60 * c.ckpt_minutos:
                self.salvar_ckpt()
                ultimo_ckpt = time.time()
        return self.terminar()

    def terminar(self):
        c = self.cfg
        if self.status == "rodando":
            self.status = "terminado"
        r = self.avaliar(c.aval_final_sementes)
        print(f"[final] vitórias {100 * r['vitorias']:.1f}% em {r['partidas']} partidas", flush=True)
        torch.save(self.online.state_dict(), self.pasta / "modelo.pt")
        _escrever_csv(self.pasta / "avaliacao_final.csv",
                      [dict(zip(("fase", "semente", "vitoria", "dano", "decisoes"), d))
                       for d in r["detalhe"]])
        self.gravar_arquivos()
        res = {"status": self.status, "passos": self.passo, "atualizacoes": self.atualizacoes,
               "episodios": len(self.episodios), "vitorias_final": r["vitorias"],
               "dano_final": r["dano"], "decisoes_final": r["decisoes"],
               "partidas_final": r["partidas"]}
        with open(self.pasta / "resultado.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        self.gravar_progresso()
        if self.ckpt.exists():
            shutil.rmtree(self.ckpt)   # o replay ocupa centenas de MB
        return 0 if self.status == "terminado" else 1

    # -------------------------------------------------------------- arquivos
    def gravar_arquivos(self):
        self.registro.salvar(self.pasta / "registro.csv")
        _escrever_csv(self.pasta / "episodios.csv", self.episodios,
                      ["episodio", "passo_jogo", "fase", "semente", "retorno", "vitoria",
                       "decisoes", "epsilon"])
        _escrever_csv(self.pasta / "avaliacoes.csv", self.avaliacoes,
                      ["passo", "partidas", "vitorias", "dano", "decisoes"])
        _escrever_csv(self.pasta / "velocidade.csv", self.velocidade,
                      ["atualizacao", "passo_jogo", "passos_por_s"])

    def gravar_progresso(self):
        dt = time.time() - self.inicio_proc
        pps = (self.passo - self.passo_proc) / dt if dt > 0 else 0.0
        p = {"receita": self.cfg.receita, "semente": self.cfg.semente, "passo": self.passo,
             "orcamento": self.cfg.passos, "passos_por_s": pps, "status": self.status,
             "atualizado": datetime.now().isoformat(timespec="seconds"),
             "avaliacoes": [(a["passo"], a["vitorias"]) for a in self.avaliacoes]}
        with open(self.pasta / "progresso.json", "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
        self.gravar_arquivos()
        try:
            status.escrever()
        except Exception as e:   # o STATUS é só um painel; não pode derrubar o treino
            print(f"(STATUS.md não atualizado: {e})", flush=True)

    # -------------------------------------------------------------- checkpoint
    def salvar_ckpt(self):
        t0 = time.time()
        tmp = self.pasta / "ckpt_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        self.replay.salvar(tmp)
        torch.save({
            "online": self.online.state_dict(), "alvo": self.alvo.state_dict(),
            "opts": [o.state_dict() for o in self.opts],
            "scheds": [s.state_dict() for s in self.scheds],
            "registro": self.registro.estado(),
            "passo": self.passo, "atualizacoes": self.atualizacoes,
            "episodios": self.episodios, "avaliacoes": self.avaliacoes,
            "velocidade": self.velocidade,
            "rng_numpy": self.rng.bit_generator.state,
            "rng_python": random.getstate(), "rng_np_global": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, tmp / "estado.pt")
        if self.ckpt.exists():
            shutil.rmtree(self.ckpt)
        tmp.rename(self.ckpt)
        print(f"[ckpt] passo {self.passo} ({time.time() - t0:.1f} s)", flush=True)

    def carregar_ckpt(self):
        e = torch.load(self.ckpt / "estado.pt", map_location=self.dev, weights_only=False)
        self.online.load_state_dict(e["online"])
        self.alvo.load_state_dict(e["alvo"])
        for o, s in zip(self.opts, e["opts"]):
            o.load_state_dict(s)
        for sc, s in zip(self.scheds, e["scheds"]):
            sc.load_state_dict(s)
        self.registro.restaurar(e["registro"])
        self.passo, self.atualizacoes = e["passo"], e["atualizacoes"]
        self.episodios, self.avaliacoes = e["episodios"], e["avaliacoes"]
        self.velocidade = e["velocidade"]
        self.rng.bit_generator.state = e["rng_numpy"]
        random.setstate(e["rng_python"])
        np.random.set_state(e["rng_np_global"])
        torch.set_rng_state(e["rng_torch"].cpu())
        if e["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in e["rng_cuda"]])
        self.replay.carregar(self.ckpt)
        self.sincronizar_ator()
        self.passo_proc = self.passo
        print(f"[retomada] passo {self.passo}, {self.atualizacoes} atualizações, "
              f"{len(self.episodios)} episódios", flush=True)


def executar(cfg, id_):
    pasta = PASTA_LOGS / id_
    pasta.mkdir(parents=True, exist_ok=True)
    if (pasta / "resultado.json").exists():
        print(f"{id_} já terminou")
        return 0
    arq_cfg = pasta / "config.json"
    if arq_cfg.exists():   # retomada: a config gravada manda
        with open(arq_cfg, encoding="utf-8") as f:
            salvo = json.load(f)["config"]
        if salvo != json.loads(json.dumps(asdict(cfg))):
            raise SystemExit(f"config diferente da gravada em {arq_cfg}")
    else:
        with open(arq_cfg, "w", encoding="utf-8") as f:
            json.dump({"id": id_, "config": asdict(cfg), "receita": RECEITAS[cfg.receita],
                       "atualizacoes_total": cfg.atualizacoes(), "torch": torch.__version__,
                       "data": datetime.now().isoformat(timespec="seconds"),
                       "dispositivo": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                       else "cpu"}, f, indent=2, ensure_ascii=False)
    treino = Treino(cfg, pasta)
    if (pasta / "ckpt" / "estado.pt").exists():
        treino.carregar_ckpt()
    elif any(pasta.glob("*.csv")):
        for p in pasta.glob("*.csv"):   # começou e caiu antes do primeiro checkpoint
            p.unlink()
    return treino.rodar()


def main():
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--receita", choices=list(RECEITAS), required=True)
    p.add_argument("--semente", type=int, required=True)
    p.add_argument("--id")
    p.add_argument("--json", help="campos extras da Config, em JSON (testes)")
    for f in fields(Config):
        if f.name in ("receita", "semente") or f.type not in (int, float, str):
            continue
        p.add_argument("--" + f.name.replace("_", "-"), type=f.type)
    a = vars(p.parse_args())
    id_ = a.pop("id") or f"{a['receita']}-s{a['semente']}"
    extra = json.loads(a.pop("json") or "{}")
    cfg = Config(**{k: v for k, v in a.items() if v is not None}, **extra)
    sys.exit(executar(cfg, id_))


if __name__ == "__main__":
    main()
