Previsão: o PPO passa de 90% de vitórias na
sanidade (só a fase 'calmo', dificuldade 0,5), como o DQN fez (cnn1 94%, cnn1d 100% em 100k
passos).

## Resultado da sanidade

**98,0% de vitórias em 200 partidas** (fase "calmo", dificuldade 0,5; avaliação gulosa nas
sementes 300–499, as da sanidade do DQN). A previsão (> 90%) se confirmou. Parou pelo orçamento
(301.056 passos) em 2,9 min, às 15h26, bem antes do horário-limite (15h55).

Curva (avaliação gulosa, 100 partidas, sementes 300–399), em passos:

| 0 | 20k | 41k | 61k | 82k | 100k | 121k | 141k | 162k | 180k | 201k | 221k | 242k | 260k | 281k | 301k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16% | 12% | 10% | 53% | 68% | 86% | 95% | 81% | **45%** | 80% | 83% | 87% | 87% | 100% | 99% | 98% |

Nas partidas de coleta (política estocástica): 31% (20k) → 52% (82k) → 88% (143k) → 94% (205k)
→ 100% (266k). Entropia 1,10 → 1,41 (61k) → 0,90 (287k); KL aproximada 0,0004 → 0,003–0,006;
fração cortada 0,3% → 4–8%; perda do V 0,12 → 0,02. Tudo em `rl/logs/ppo-sanidade/`.

Comparação com o DQN, com cuidado: a sanidade do DQN usou k = 10 (cnn1 94%, cnn1d 100% em 100k
passos); aqui k = 20, então 100k passos do PPO cobrem o dobro do tempo de jogo de 100k passos do
DQN. O PPO chegou a 86% em 100k passos e passou de 90% de forma estável só depois de ~250k.
A curva gulosa oscila (95% em 121k, 45% em 162k, 80% em 180k): a ação mais provável ainda
mudava enquanto a política estocástica já vencia ~90%.

## O que foi implementado (`rl/ppo.py`)

- **PPO com clip**, crítico V e **GAE** (λ = 0,95). Vantagens normalizadas por minilote;
  perda = −min(r·Â, clip(r, 1 ± ε)·Â) + 0,5 · 0,5·(V − R)² − 0,01 · entropia.
- **Rede:** a cnn1d (`desafio3/modelo.py`, observação v2 de 25 canais). Logits da política =
  o mapa A (5 × 5 × 9, alinhado com `codificar_acao`) + A(esperar); crítico = `cabeca_v`.
  Mesmos pesos e camadas da cnn1d; só a leitura das cabeças muda (sem a combinação dueling).
- **Máscara:** logit −1e9 nas ações inválidas (a máscara sai dos canais 20–24 da observação,
  como na cnn1d); a probabilidade delas é exatamente 0. Cada ação sorteada é conferida na
  máscara real do ambiente (assert) durante a coleta.
- **Coleta:** 16 partidas em paralelo; o simulador roda na CPU, e a inferência é feita em lote
  (1 forward por passo para as 16) na GPU, onde também fica a atualização. Partidas que acabam
  são reiniciadas na hora (fase e semente sorteadas, sementes 1000–9999).
- **k = 20, γ = 0,994.** Só reset/step/observar/mascara do Ambiente.
- **Registro por atualização** (`registro.csv`): passo, episódios encerrados na coleta, retorno
  médio e vitórias deles, entropia, KL aproximada ((r − 1) − log r), fração cortada, perda do V,
  perda da política, passos/s. Avaliação gulosa (argmax dos logits mascarados) com
  `rl.avaliar.avaliar_rede` (que ganhou o parâmetro `formato`), mesma regra de sementes do DQN.

## Hiperparâmetros

| | |
|---|---|
| partidas em paralelo × passos por coleta | 16 × 128 = 2.048 |
| épocas × minilote | 4 × 512 (16 passos de gradiente por coleta) |
| otimizador | Adam, η = 2,5e-4 constante, ε = 1e-5 |
| clip | 0,2 |
| γ, λ (GAE) | 0,994, 0,95 |
| coeficientes V, entropia | 0,5, 0,01 |
| norma máxima do gradiente | 0,5 |
| semente | 0 |

## Velocidade

~1.750 passos/s em média (~2.200 no começo, ~1.700 no fim, quando as partidas ficam mais longas
e com mais plantas). 300k passos em 2,9 min. O DQN, com treino ligado, fazia ~400–500 passos/s
com k = 10.

## Testes (`tests/test_ppo.py`, 4 passam)

- 3.200 passos de coleta (8 partidas × 400) sem nenhuma ação inválida (assert na máscara real
  em cada passo, mais a conferência contra os canais de máscara gravados).
- Formatos: logits (B, 226), V (B,); probabilidade 0 nas inválidas; soma 1.
- GAE conferido à mão (com fim de partida no meio, sem bootstrap).
- Mini-treino de ponta a ponta (600 passos, 3 avaliações, arquivos gravados).

## O que falta

- Rodar nas 21 fases de treino (dificuldade 1) e comparar com a baseline do DQN (9,2%) e com o
  Econ5 (65% com k = 20); várias sementes.
- Retomada (checkpoint e parar/continuar como no rl/dqn.py); o rl/rodar.py ainda não chama o PPO.
- Mais sementes na sanidade (só a 0 foi rodada) e a oscilação da política gulosa (talvez
  avaliar com mais partidas ou reduzir η no fim).
- Nenhum merge feito (branch `ppo`, a partir da `diag`).

## Fases completas

Previsão (fases completas): o PPO nas 21 fases de treino, com k = 20 e uma semente, passa
da baseline do DQN (9,2%) e chega a ≥ 25% nas 420 partidas finais. Regra: ≥ 25% → o PPO
vira o caminho principal; < 15% → investigar (G4, dados) antes de mais sementes. Com uma
semente, o resultado é indicativo, não conclusivo. Referências com k = 20: Econ5 65%,
Guloso 52%.

### Resultado (fases completas)

21 FASES_TREINO, dificuldade 1, sementes de treino 1000–9999 (as do DQN), k = 20, γ = 0,994,
os hiperparâmetros da sanidade, semente 0. Parou pelo **orçamento (2.000.896 passos)** em
17,2 min (~1.940 passos/s), às 16h00. Nenhuma fase de teste usada.

**Final, 420 partidas (FASES_TREINO × 300–319):**

| política | vitórias (420) | dano | decisões/partida | esperar | girassol | ervilha | repetidora | batata | mina |
|---|---|---|---|---|---|---|---|---|---|
| gulosa | **81,7%** (IC 95% ≈ 77,7–85,1) | 0,894 | 200 | 48,8% | 18,6% | 12,8% | 6,0% | 5,8% | 8,1% |
| com sorteio | 78,1% | 0,849 | 173 | 48,3% | 17,1% | 13,0% | 7,4% | 6,3% | 7,9% |

Referências nas mesmas 420 partidas: Econ5 65,2% e Guloso 51,7% com k = 20
(`tools/medir_k.md`); baseline do DQN (lote 1, k = 10) 9,2% ± 7,1.

**Curva** (63 partidas, FASES_TREINO × 300–302):

| passos | 0 | 252k | 502k | 752k | 1,00M | 1,25M | 1,50M | 1,75M | 2,00M |
|---|---|---|---|---|---|---|---|---|---|
| gulosa | 0% | 0% | 0% | 47,6% | 73,0% | 85,7% | 82,5% | 81,0% | 73,0% |
| com sorteio | 0% | 0% | 1,6% | 20,6% | 58,7% | 69,8% | 68,3% | 76,2% | 73,0% |

Até ~500k passos não vence nada (planta batata e mina e quase nenhum atirador, como o DQN);
decola entre 500k e 750k, quando passa a plantar ervilha e repetidora. Do 1,25M em diante fica
num patamar de ~75–85% (as avaliações de 63 partidas têm IC de ±10–12 pontos).

**Previsão: confirmada.** 81,7% ≥ 25% (e acima da baseline do DQN, 9,2%). Pela regra combinada,
**o PPO vira o caminho principal.** Com uma semente, o resultado é indicativo, não conclusivo.

Cuidados na comparação com o DQN:
- o DQN do lote 1 usou k = 10 e 400k passos; o PPO, k = 20 e 2M passos (~10× o tempo de jogo).
  Aos 400k passos o PPO também estava em 0%. Uma parte da diferença pode ser orçamento e k,
  não o algoritmo; o G0 (baseline longa do DQN) e um DQN com k = 20 separam isso;
- uma semente só; falta repetir com mais sementes e avaliar no teste (sementes a partir de 10100).
- O PPO passa o Econ5 (81,7% contra 65,2%) com uma política parecida na proporção de atiradores
  (ervilha + repetidora ≈ 19% das ações) mas que também planta batata e mina (~14%), que o Econ5
  nunca planta.

## Três sementes em CPU (previsão escrita antes de rodar — 10/10/2026)

A semente 0 rodou em CUDA, na máquina do Gustavo. Para a média ± desvio não misturar
backends, as **três** sementes são refeitas no mesmo pipeline (CPU, 8 threads, torch 2.14.1),
com `--passos 2000000 --aval-cada 250000`. Ids: `ppo-fases-s0-cpu`, `ppo-fases-s1`,
`ppo-fases-s2`. O log original em CUDA (`ppo-fases-s0`) é preservado.

**Previsão (antes de rodar):**

1. As três sementes passam do Econ5 (65,2%) na avaliação final de 420 partidas.
2. A semente 0 em CPU reproduz, dentro da variação entre sementes, os 81,7% obtidos em CUDA.
   Se ela ficar muito abaixo, a diferença é de backend ou o 81,7% foi sorte de uma execução —
   e isso precisa aparecer no relatório antes de qualquer conclusão.
3. O desvio-padrão entre as três fica abaixo de 10 pontos percentuais (o DQN variou de
   1,4% a 15,5%, com média 9,2% ± 7,1).

**Refuta:** qualquer semente abaixo de 65,2%, ou desvio acima de 10 pontos.

### Desfecho
(preencher depois das três execuções, sem editar a previsão acima)

---

## Índice das execuções de 10–12/10/2026 (só os fatos; o desfecho fica acima, em branco)

Tudo rodado numa máquina sem CUDA: PPO em **CPU** (8 núcleos, `set_num_threads(4)` na
cópia local; o arquivo versionado segue com 1) e DQN em **MPS**. Backend diferente do
`ppo-fases-s0` original, que rodou em CUDA — por isso a semente 0 foi refeita como
`ppo-fases-s0-cpu`, para as sementes serem comparáveis entre si.

### 1. Oito sementes a 2M passos (`ppo-fases-s0-cpu`, `s1`…`s7`)

| s0-cpu | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|
| 60,5% | 17,1% | 94,0% | 4,0% | 11,9% | 56,2% | 88,6% | 71,7% |

Média **50,5% ± 35,2** (420 partidas cada). Três das oito passam do Econ5 (65,2%).
O `ppo-fases-s0` original em CUDA deu 81,7%, o 2º melhor valor das oito.

### 2. Três sementes a 4M passos (`-4M`)

| semente | 2M | 4M |
|---|---|---|
| s0-cpu | 60,5% | **94,0%** |
| s1 | 17,1% | **86,4%** |
| s3 | 4,0% | **56,7%** |

Nas três, a curva ainda subia no último ponto medido. Como o PPO não tem agenda de η
(lr constante), uma execução de 4M reproduz os primeiros 2M da de 2M com a mesma
semente — então são continuações, não execuções independentes.

### 3. DQN comparável (`dqn-cnn1d-k20-4M-s0`)

Mesma rede (cnn1d), mesmo k (20), mesmo γ (0,994) e mesmo orçamento (4M) do PPO.

```
final (420 partidas)  49,3%
pico na curva         79,4% no passo 1,75M
velocidade            131 passos/s  (o PPO faz ~1.000 na mesma máquina)
```

Referência: o lote 1 (k=10, 400k) deu 9,2% ± 7,1.

### 4. Receitas de otimizador no PPO, 4M, 1 semente (`ppo-rec-*`)

Mecanismo `RECEITAS` portado do `rl/dqn.py`, reaproveitando `OTIMIZADORES` e `AGENDAS`
do `desafio3/treino.py`. Curva de vitórias (63 partidas por ponto):

| receita | 0,5M | 1M | 1,5M | 2M | 2,5M | 3M | 3,5M | 4M | final (420) | cruza 65,2% |
|---|---|---|---|---|---|---|---|---|---|---|
| atual (adam 2,5e-4 const) | 2% | 0% | 8% | 60% | 87% | 89% | 89% | 92% | 94,0% | 2,5M |
| vencedora (sgd 3e-2 cos) | 0% | 27% | 81% | 92% | 97% | 95% | 95% | 97% | 93,8% | **1,5M** |
| rmsprop 3e-4 wc | 0% | 0% | 44% | 84% | 98% | 97% | 97% | 95% | 93,6% | 2,0M |
| muon 1e-3 wc | 11% | 27% | 62% | 87% | 87% | 92% | 97% | 95% | 93,1% | 2,0M |

As quatro terminam entre 93,1% e 94,0% (faixa de 0,9 ponto). Nenhuma divergiu.
**Uma semente por receita** — e a seção 1 mediu ±35 pontos de variação entre sementes
a 2M, então a diferença de velocidade não tem barra de erro.

### 5. Mudanças no `rl/ppo.py`

- `salvar_ckpt` / `carregar_ckpt` e a flag `--continuar` (item 5.2.1 do `PASSAGEM.md`).
- `melhor.pt`, salvo quando a avaliação periódica gulosa melhora.
- `RECEITAS` e `--receita` (`atual`, `vencedora`, `rmsprop`, `muon`), com a agenda
  recebendo o total de passos do otimizador (`Config.atualizacoes()`), não de jogo.
- `tests/test_ppo.py` continua com 4 testes passando.

### 6. O que falta rodar

1. Mais sementes das receitas, para dar barra de erro à diferença de velocidade:
   `python -m rl.ppo --id ppo-rec-vencedora-4M-s1 --receita vencedora --semente 1 --passos 4000000 --aval-cada 250000`
2. DQN × 3 receitas em k=20 e 4M (o lote 1 comparou receitas a k=10 e 400k, quando o
   DQN marcava 9,2% — comparação dominada por ruído). Em CPU/MPS são ~8h por execução;
   em CUDA, ~2,8h.
3. Generalização do PPO nas fases de **teste**, sementes a partir de 10100 (item 5.2.3).
4. s3 a 6M: a 4M ela estava em 56,7% e ainda subindo.

---

## DQN comparável (3 sementes, 2M)

Escrito em 12/10/2026, antes de rodar. Receita baseline do DQN (Adam 1e-4 constante), cnn1d,
k = 20, γ = 0,994, exploração uniforme, sem currículo, 2M passos, sementes 0, 1 e 2, em
sequência na RTX 3050 (CUDA). Ids `dqn-cnn1d-k20-2M-s0`…`s2`. O `rl/dqn.py` passa a guardar
`melhor.pt` (a melhor avaliação periódica gulosa, 63 partidas a cada 50k), como o PPO.

Previsão: com k = 20 e 2M passos, o DQN fica abaixo do PPO nas mesmas sementes (0–2) e
no mesmo orçamento. Critério: |Δ| > 2 s_c. Com a variância observada no PPO (±35 pontos),
o resultado mais provável é empate formal. Por isso também se reportam: em quantas
sementes cada algoritmo passa do Econ5 (65,2%) e o tempo de relógio de cada um.

### Desfecho
