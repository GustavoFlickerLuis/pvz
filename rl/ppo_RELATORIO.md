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
