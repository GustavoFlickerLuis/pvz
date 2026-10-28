# Diagnóstico: por que o DQN não aprende a escolher jogadas?

Pergunta, hipóteses, tabela de decisão e previsões em `rl/diag/PLANO.md` (escrito antes de
rodar, commit e39200d). Tudo aqui é com **k = 20 e γ = 0,994** (mudança de prioridade: k = 10
fica para a noite). Branch `diag`, 09/10/2026.

## C2: baselines por k e o K escolhido

`tools/medir_k.md` (FASES_TREINO × sementes 300–319, 420 partidas por linha; 47 s em 8 processos).

| k | Econ(3) | Econ(4) | Econ(5) | Econ(6) | Econ(8) | Guloso | aleatório |
|---|---|---|---|---|---|---|---|
| 10 | 26,2% | 58,6% | **67,1%** | 54,3% | 38,6% | 41,2% | 0% |
| 20 | 25,2% | 55,2% | **65,2%** | 53,1% | 35,7% | 51,7% | 0% |
| 30 | 25,2% | 52,9% | **63,3%** | 54,0% | 35,7% | 49,3% | 0% |

**K = 20 (γ = 0,994).** Motivo, pela regra combinada: com k = 20 a curva Econ(N) mantém o
pico no meio (N = 5) e o Econ5 cai 1,9 ponto (limite: 6). Com k = 20, as partidas do Econ5
têm metade das decisões (127 contra 256). O Guloso sobe 10 pontos com k = 20: decidindo com
menos frequência, ele acaba juntando sol.

## Dados e provas (C3, C4)

- **Dados de TD** (β = Econ5 + 15% aleatórias, k = 20): treino FASES_TREINO × 0–59 =
  111.262 transições (1.260 partidas, β vence 34%, "esperar" em 68% das ações); validação
  × 100–109 = 18.557. 26 s.
- **Prova de avaliação:** FASES_TREINO × 600–629, 600 estados (200 por faixa de posição:
  < 30, 30–99, ≥ 100), 6,2 candidatas por estado, 8 rollouts. 1.255 pares claros (356 no grupo
  i, com a ação do Econ5; 899 no grupo ii, só entre alternativas). EP mediano de Q̂: 0,006.
  95 s em 16 processos.
- **Prova de treino** (do teto): FASES_TREINO × 700–799, 1.998 estados, 4 rollouts, 3.528
  pares claros. 233 s. Ficou no tamanho cheio (a estimativa era ~4 min, abaixo dos 40).
- Sementes 600–799 conferidas: nenhuma parte do rl/ as usa (treino 1000–9999, avaliação
  300–319, currículo 400–420, sanidade 300–499; desafio3 e dados de TD 0–59 e 100–109).
- Os rollouts de um estado usam o mesmo rng de β em todas as candidatas (números aleatórios
  comuns); o rng muda de um rollout para outro.

## G1 e G2: propagação (100k atualizações)

"Propagou" = |viés das primeiras 30 decisões| < 0,05 e Huber dessa faixa ≤ 1,1 × a do mc na
mesma atualização (medido a cada 5k, na validação).

| execução | propagou? | viés ini (final) | Huber ini (final) | Huber ini do mc (final) | viés meio / fim (final) |
|---|---|---|---|---|---|
| td, alvo a cada 2.000 | **sim, em 25.000** | +0,036 | 0,2232 | 0,2372 | +0,079 / +0,128 |
| td, alvo a cada 500 | **sim, em 25.000** | +0,018 | 0,2220 | 0,2372 | −0,008 / +0,025 |
| mc | — | +0,019 | 0,2372 | 0,2372 | +0,081 / +0,145 |
| G2: cnn1d, td, alvo a cada 2.000 | **sim, em 25.000** | +0,015 | 0,2202 | 0,2372 | — |

G2: alvo a cada 2.000, o que propagou melhor (os dois em 25k; o de 2.000 com o viés estável
por mais tempo). cnn1d (71,2%) × cnn1 (64,2%) com o mesmo alvo: 7 pontos, mais que os 3 que
pediriam mais 2 sementes.

Figura: `rl/diag/figuras/vies_ini.png`. O critério olha a primeira vez; depois dela o viés
**oscila** nos três, inclusive no mc (que vai a +0,135 aos 80k): com Adam 1e-4 constante a
rede não para num ponto. O td-2000 é o mais estável (dentro de ±0,05 de 25k a 65k); o td-500
sai da faixa em 35k, 45–50k, 60k e 70–80k. A Huber das primeiras 30 decisões do td fica
abaixo da do mc a partir de ~25k (o alvo de n passos tem menos variância que G).

## Notas na prova de avaliação (600 estados, 1.255 pares claros)

| rede | nota | grupo i (com Econ5; 356) | grupo ii (só alternativas; 899) | arrependimento | nota sem treino |
|---|---|---|---|---|---|
| ref.: Q constante | 50,0% | 50,0% | 50,0% | 0,144 | — |
| ref.: **Q por tipo, sem estado** | 56,9% | **69,8%** | 51,7% | 0,133 | — |
| ref.: escolha do Econ5 | — | — | — | **0,089** | — |
| G1 td-2000 (cnn1) | 64,2% | 64,6% | 64,1% | 0,112 | 40,4% |
| G1 td-500 (cnn1) | 66,6% | 68,8% | 65,7% | 0,118 | 40,4% |
| G1 mc (cnn1) | 63,4% | 62,9% | 63,6% | 0,118 | 40,4% |
| G2 td-2000 (cnn1d) | **71,2%** | 72,8% | 70,6% | 0,100 | 67,6% |
| G3 teto cnn1d | **73,3%** | 74,4% | 72,9% | 0,107 | 67,6% |
| G3 teto cnn1 | 65,6% | 64,9% | 65,9% | 0,119 | 40,4% |

Curvas da nota (a cada 20k): cnn1d td-2000 67,6 → 73,7 → 74,6 → 74,0 → 73,9 → 71,2 (propagou
em 25k, como a cnn1); cnn1 td-2000 40,4 → 65,4 → 68,6 → 65,3 → 65,4 → 64,2; td-500 40,4 →
65,8 → 69,3 → 66,1 → 66,5 → 66,6; mc 40,4 → 66,1 → 67,7 → 66,5 → 66,6 → 63,4. A nota já
está em ~65% aos 20k e não sobe depois.

Duas referências que **não estavam no PLANO** e que pesam na leitura:
- **Q por tipo** (a média de Q̂ por tipo de ação, ajustada na prova de treino, sem olhar o
  estado) tira 56,9%, e **69,8% no grupo i**: os pares com a ação do Econ5 se ordenam em boa
  parte só pelo tipo (o Econ5 joga ervilha e repetidora, que têm Q̂ médio maior: 0,63 e 0,67,
  contra 0,40 da batata). No grupo ii ela fica em 51,7%.
- **A cnn1d sem treino tira 67,6%** (a cnn1 sem treino, 40,4%): a nota inicial depende de
  vieses fixos entre tipos de ação da rede aleatória. "≥ 60%" sozinho não garante que a rede
  use o estado. As redes td da cnn1 partem de 40,4% e passam o Q por tipo nos dois grupos,
  sobretudo no ii (64–66% contra 51,7%), então usam o estado.
- **Arrependimento:** todas as redes, inclusive os tetos, escolhem pior que o Econ5 (0,100 a
  0,119 contra 0,089); a mais perto é a cnn1d com td (0,100). Elas acertam ~2/3 dos pares, mas a melhor candidata de cada estado
  ainda é mais vezes a do Econ5 do que a da rede.

## Tempo

| fase | tempo |
|---|---|
| C2 (7 agentes × 3 k × 420 partidas) | 47 s |
| C3 (dados de TD) | 26 s |
| C4 (provas de avaliação e de treino) | 95 s + 233 s |
| G1 (3 treinos de 100k, em paralelo) | 52–57 min cada (os primeiros 30 min dividindo a GPU com mais 4 treinos e a CPU com a prova de treino; sozinho, ~20 min) |
| G2 (cnn1d, 100k, sozinha na GPU) | 10 min |
| G3 (tetos, 20k atualizações) | 26–28 min (em paralelo com o G1) |

Duas falhas minhas na execução: lancei 5 treinos de TD e os 2 tetos juntos, o que deixou todos
lentos (a GPU satura); e o td.py só grava as métricas no fim. Para caber no prazo, interrompi as
duas cnn1d do G2 (alvo 500 e 2.000) aos ~30 min e rodei de novo só a do alvo escolhido (2.000).

## Previsões do PLANO

- (a) com k = 20 a curva Econ(N) mantém o pico no meio: **confirmada** (pico em N = 5).
- (b) k = 10 e alvo 2.000 só abaixo de 0,05 depois de 50k; com 500 ou com k = 20, antes:
  **a parte de k = 20 confirmada** (os dois alvos em 25k); a de k = 10 não foi medida (pendente).
- (c) a regressão mc fica perto de 50%: **refutada** (63,4%; 66–68% no meio do treino).
- (d) o teto passa de 60%: **confirmada** (cnn1d 73,3%; cnn1 65,6%).
- (e) as redes de RL dos lotes 1 e 2a ficam perto de 50%: **não medida** (G4 pendente; ela
  pede a prova de k = 10).

## Decisão

Aplicando a tabela do PLANO, linha por linha:

1. "td com alvo a cada 2.000 não propaga em 100k e com 500 propaga": **não se aplica.** Os dois
   propagam em 25.000. Nada a adotar; o alvo a cada 2.000 continua (e foi o mais estável).
2. "O teto ordena e alguma arquitetura treinada com td ordena": **se aplica.** O teto ordena
   (cnn1d 73,3%, cnn1 65,6%; ≥ 60%) e as duas arquiteturas treinadas com td ordenam (cnn1d
   71,2%; cnn1 64,2% com alvo 2.000 e 66,6% com 500; ≥ 60% nos dois grupos). A melhor no G2 é a
   cnn1d, 7 pontos acima da cnn1. Próximo passo pela linha 2: **um treino de RL só com a cnn1d,
   no k escolhido (k = 20, γ = 0,994), 3 sementes** (alvo a cada 2.000, o padrão).
3. "O teto ordena e nenhuma rede td ordena": não se aplica (a cnn1 td ordena).
4. "O teto não ordena": não se aplica.
5. "Os pares com a ação do Econ5 ordenam e os só entre alternativas não": não se aplica (o grupo
   ii também ordena: 64–66%).

**Ressalvas sobre a leitura da linha 2.** A tabela diz "causa B, resolvida pela arquitetura",
mas dois números pedem cuidado:
- **A cnn1, a mesma rede do RL dos lotes 1 e 2a, também ordena com td** (64–67%, partindo de
  40,4% sem treino). Então a arquitetura não é o que separa "ordena" de "não ordena" aqui. O que
  mudou em relação ao RL foi (1) k = 20 em vez de 10 e (2) os dados: aqui eles vêm de β (Econ5
  com 15% de ruído); no RL, da própria política ε-gulosa.
- **A vantagem da cnn1d é em parte de largada:** sem treino ela já tira 67,6% (vieses fixos
  entre tipos de ação da cabeça dueling). O treino td a leva a 71,2% (+3,6); a cnn1 vai de 40,4%
  a 64,2% (+23,8). A cnn1d é a melhor em nota e em arrependimento, mas a margem que se deve ao
  aprendizado é menor do que os 7 pontos sugerem.

O passo da linha 2 (RL com a cnn1d em k = 20) muda a arquitetura e o fator (1) ao mesmo tempo. Se ele não resolver, o candidato é o
fator (2), que é a causa A do PLANO; o G4 (nota das redes de RL) é a medida que separa os
dois. Isso é uma observação sobre a leitura, não uma decisão nova: pela tabela, o próximo
passo é o da linha 2.

## Pendente (para a noite, depois deste relatório)

- G0: a baseline longa.
- G1 com k = 10 (precisa dos dados e das provas de k = 10).
- G2 com cnn2 e cnn3 (e mais sementes se as duas melhores ficarem a menos de 3 pontos).
- G4: nota das redes de RL (baseline-s2 do lote 1 e a última semente do lote 2a), que pede a
  prova de k = 10.
- Retomar o lote 2a: apagar o `rl/PARAR` da pasta original e chamar `python -m rl.rodar --fila
  cnn1:nao:0:intencional cnn1:nao:1:intencional cnn1:nao:2:intencional` (as duas primeiras
  são puladas; a semente 2 retoma do passo 192.431).
