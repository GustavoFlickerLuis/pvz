# Passagem de bastão: onde paramos (09/10/2026)

Para o Pablo continuar sem depender do Claude Code do Gustavo. A seção 1 resume o estado; a 4.3
tem o resultado do PPO; a 5 diz o que fazer, em ordem. Dúvidas: Gustavo pelo WhatsApp.

---

## 1. O estado em seis linhas

1. Simulador, calibração, divisão treino/teste e Desafio 3 prontos (branch `main`).
2. O DQN (branch `rl`) **não aprendeu**: 9,2% ± 7,1 (lote 1, k = 10, 400k passos). A exploração
   intencional (lote 2a) fez pior (1,3%).
3. Um diagnóstico com regras escritas antes (`rl/diag/`, branch `diag`) mostrou que a rede
   consegue aprender a ordenar as jogadas e que o alvo de TD funciona (seção 4).
4. **O professor sugeriu PPO, e o PPO funcionou:** 81,7% de vitórias nas 21 fases de treino
   (420 partidas, IC 95% ≈ 78–85%), acima do Econ5 (65%) e do Guloso (52%) com k = 20. Uma semente
   só, 2M passos, 17 minutos (branch `ppo`, `rl/ppo_RELATORIO.md`).
5. **O PPO é o caminho principal daqui em diante.** O que falta para virar resultado está na seção 5.
6. **Prazo mais próximo:** Desafio 3, quarta 14/10 às 12h (slide + desfechos; seção 5.1).

## 1.1 Branches (nenhuma foi mesclada)
- `main`: simulador e Desafio 3.
- `rl`: DQN, lotes 1 e 2a. A semente 2 do lote 2a ficou **pausada** no passo 192.431 (o
  `rl/PARAR` está na pasta do Gustavo); não precisa ser retomada.
- `diag` (a partir da `rl`): k configurável, `tools/medir_k`, `rl/diag/` (PLANO, RELATORIO,
  provas, td.py).
- `ppo` (a partir da `diag`): `rl/ppo.py`, `tests/test_ppo.py`, `rl/ppo_RELATORIO.md`. **Trabalhe
  a partir desta.**

## 2. O que já sabemos (medido)

| # | fato | fonte |
|---|---|---|
| F1 | A CNN representa bem o valor do tabuleiro (ajusta o retorno do Econ5 quase até o piso de ruído; 94% no "calmo" fácil) | Desafio 3; sanidade do lote 2 |
| F2 | A diferença entre jogadas é ~1% da variação do retorno (trocar uma jogada muda G em 0,006 na mediana; G varia 0,62 entre estados) | clones do ambiente |
| F3 | Uma rede que ajusta bem o G e joga pelo argmax vence 0,5% | `desafio3/rede_joga.md` |
| F4 | Otimizadores (lote 1) e exploração intencional (lote 2a) não ajudaram. Na intencional, 89% das decisões do começo do treino eram "juntando sol para o alvo" | `rl/RELATORIO.md` |
| F5 | Decidir a cada 20 ticks (k = 20) preserva o jogo: Econ(N) = 25/55/65/53/36%, pico em N = 5, e metade das decisões. O Guloso sobe de 41% para 52% | `tools/medir_k.md` |
| F6 | Com dados do Econ5 e k = 20, a rede aprende a ordenar as jogadas: cnn1d 71,2% (td), teto 73,3% | `rl/diag/RELATORIO.md` |
| F7 | O alvo de TD propaga a vitória até o começo da partida em 25 mil atualizações (k = 20), com a rede alvo a cada 2.000 ou a cada 500 | idem |
| F8 | **Ressalva importante:** parte da nota vem só do tipo da planta. Um "Q por tipo" que ignora o tabuleiro tira 56,9% (69,8% nos pares com a ação do Econ5), e a cnn1d sem treino tira 67,6%. A medida limpa do uso do tabuleiro é o grupo ii (pares só entre alternativas): Q por tipo 51,7%, redes treinadas 64–73%. E todas as redes escolhem pior que o Econ5 (arrependimento de 0,100–0,119, contra 0,089) | idem |

## 3. O diagnóstico, em uma página

Hipótese: a rede aprende o **V** ("quão bom é o tabuleiro") e não o **A** ("qual jogada é melhor"),
porque o A é ~1% do V. Três causas possíveis: **(A)** faltam dados das jogadas alternativas,
**(B)** o alvo de treino é ruidoso demais, **(C)** a informação não está na observação.

- **Prova de ordenação:** ~600 estados em que sabemos, por clones do jogo, qual jogada é melhor.
  Nota = % dos "pares claros" (jogadas com diferença real de valor) que a rede põe na ordem certa.
- **Teto:** uma rede treinada direto com a resposta da prova. Se nem ela ordena estados novos,
  a informação não está na observação (C).
- **Propagação:** a máquina de TD do DQN avaliando o Econ5 em dados fixos (a resposta certa é o
  retorno real).

## 4. Resultado e o que ele quer dizer

### 4.1 O que a tabela decidiu
Linha 2 do PLANO: o teto ordena, e uma rede treinada com TD ordena (cnn1d 71,2%). Não é causa C
nem causa A nos dados do Econ5, e a propagação não é gargalo. **Próximo passo pela tabela: um
treino de RL com a cnn1d, k = 20 (γ = 0,994), 3 sementes.**

### 4.2 Como ler com cuidado
- O limiar "ordena = 60%" supunha que chutar dá 50%. Mas ordenar só pelo tipo da planta já dá
  57–68% (F8). **Daqui em diante, compare redes pela nota no grupo ii e pelo ganho sobre o
  "Q por tipo"**, não contra 50%.
- A cnn1, a mesma rede que falhou no RL, também ordena quando treinada com os dados do Econ5
  (de 40% sem treino para 64–67%). Então a arquitetura não é o que separa sucesso de fracasso.
- A nota chega a ~65% em 20 mil atualizações e não sobe depois: o sinal aprendível se esgota cedo.
- O sinal que depende do tabuleiro existe, mas é fraco. Isso combina com F2.
- **O que difere entre o diagnóstico (que funcionou) e o RL (que falhou):**
  1. o k (20 no diagnóstico, 10 no RL);
  2. a origem dos dados (Econ5 × a própria política, que morre cedo e quase não vê fim de fase);
  3. o controle: o DQN escolhe pelo argmax entre 226 valores quase iguais, o que favorece a jogada
     cujo erro foi maior para cima. O diagnóstico só avaliou a política do Econ5, sem argmax.
- O treino da linha 2 testa o item 1. O G4 (nota das redes de RL na prova) ajuda a separar 2 de 3:
  se as redes de RL ordenam mal, o problema são os dados; se ordenam bem e jogam mal, é o argmax.

### 4.3 O PPO (sugestão do professor) e por que ele funcionou aqui
- No DQN, a rede precisa acertar o A das 226 jogadas, e o argmax amplifica erros pequenos (F2, F3).
- No PPO, o crítico só aprende o **V**, que a rede já faz bem (F1). O ator é uma política
  probabilística, ajustada aos poucos na direção das jogadas que saíram melhor que o esperado.
  Não há argmax sobre valores quase iguais. A exploração vem da própria política (sem ε).
- Implementação (`rl/ppo.py`): a cnn1d, com o mapa de A + "esperar" como logits da política
  (inválidas com logit −1e9) e o V como crítico; 16 partidas em paralelo × 128 passos por
  coleta; 4 épocas, minilote 512; Adam 2,5e-4; clip 0,2; GAE λ = 0,95; coeficientes 0,5 (V) e
  0,01 (entropia); norma máxima do gradiente 0,5; k = 20, γ = 0,994. ~1.900 passos/s.
- Sanidade ("calmo", dificuldade 0,5): 98% em 200 partidas.
- Fases completas (21 de treino, semente 0, 2M passos):

  | passos | 0 | 250k | 500k | 750k | 1,0M | 1,25M | 1,5M | 1,75M | 2,0M |
  |---|---|---|---|---|---|---|---|---|---|
  | vitórias, gulosa (63 partidas) | 0% | 0% | 0% | 47,6% | 73,0% | 85,7% | 82,5% | 81,0% | 73,0% |

  Final, 420 partidas: **gulosa 81,7%**, com sorteio 78,1%; dano 0,89; ervilha + repetidora em
  19% das ações (o DQN: 2–6%). Até 500k passos não ganhou nenhuma partida, como o DQN; a virada
  veio quando passou a plantar ervilha e repetidora.
- **Ressalvas:** uma semente só; avaliação nas fases de treino (as de teste ainda não foram
  usadas); a política gulosa oscila durante o treino (45% aos 162k na sanidade; 73% aos 2M,
  contra 85,7% aos 1,25M, com só 63 partidas por ponto); a comparação com o DQN não é justa
  (o DQN rodou k = 10 e 400k passos; aos 400k o PPO também estava em 0%).

### 4.4 Outras perguntas do professor (09/10)
- **Lote:** 256 no DQN (o padrão do Atari é 32); dificilmente é o culpado.
- **O estado é markoviano?** O simulador é. A observação não é totalmente: (1) de propósito, o
  agente não vê os zumbis futuros (o canal 19 resume a história do segmento); (2) faltam relógios
  internos: quanto falta para a mina armar, a posição do zumbi dentro da célula e o tempo até o
  próximo sol do girassol. Esses podem virar 2–3 canais novos sem vazar o futuro, se for preciso.

## 5. O que fazer, em ordem

### 5.1 Desafio 3 (até quarta 14/10, 12h), branch `main`
- `desafio3/slide.pdf` (um slide) e os desfechos do `desafio3/hipoteses.md`, **escritos por nós**.
  As previsões não se editam.
- Resultado central: as 3 finalistas (sgd 3e-2 cosseno, sgd 3e-2 warmup_cosseno, rmsprop 3e-4
  warmup_cosseno; val ≈ 0,1623) empatam entre si e vencem a baseline adam 1e-4 constante
  (0,1676 ± 0,0008); o Muon perde para todos (0,1750). Tabelas em `desafio3/tabelas.md`.

### 5.2 Consolidar o PPO (branch `ppo`), nesta ordem
1. **Parar/retomar com checkpoint** no PPO (como no DQN) e **guardar o melhor checkpoint** pela
   avaliação periódica, já que a política gulosa oscila.
2. **Mais duas sementes** (1 e 2), com a mesma receita e 2M passos (~20 min cada). Antes, escrever
   no `rl/ppo_RELATORIO.md` a previsão: "as três sementes passam do Econ5 (65%)". Média ± dp
   nas 420 partidas.
3. **Generalização (experimento 4):** avaliar nas 6 fases de **teste**, sementes **a partir de
   10100**, 20 a 50 por fase, só depois de fechar a receita. Uma vez só, como no Desafio 3.
4. **Comparação justa com o DQN** (o professor vai perguntar): DQN com cnn1d, k = 20 e o mesmo
   orçamento (2M passos), 3 sementes. Sem isso, não dá para dizer que a diferença é do algoritmo,
   e não do k e do orçamento. Pode substituir o "G0" (baseline longa).
5. Depois: os experimentos da Parte 3 do CONCEITO com o PPO (curvas, baselines, erros,
   generalização, ablações CNN × MLP e com/sem currículo, mapas de comportamento) e a
   "heurística humana" (o baseline mais forte que conseguirmos).
6. **Atualizar o CONCEITO e o CLAUDE.md** (decisão dos dois): algoritmo PPO com máscara (2.5),
   k = 20 e γ = 0,994 (2.3). Hoje os dois ainda dizem DQN, k = 10 e γ = 0,997.

### 5.3 Pendências do diagnóstico (baixa prioridade agora)
- G4 (nota das redes de RL na prova) e G1 com k = 10: só se forem úteis para explicar, no
  relatório, por que o DQN falhou.
- G2 com cnn2/cnn3: dispensável.

## 6. O que NÃO fazer
- Não rodar o lote 2 inteiro (21 execuções) nem varrer otimizadores.
- Não voltar à exploração intencional (lote 2a: 1,3% contra 9,2%).
- Não mexer na receita do PPO antes de ter as 3 sementes: primeiro saber quanto ela varia.
- Não editar `PLANO.md`, `PREVISOES.md` nem previsões já escritas depois de rodar.
- Não usar as fases de teste no treino. O teste final do RL usa sementes **a partir de 10100**
  (10000–10019 já foram usadas no Desafio 3).
- Código de RL só usa `reset`, `step`, `observar` e `mascara` (regra do CLAUDE.md).
- Não tocar na `main` com código experimental. O `rl/PARAR` é global: não rode o teste de
  retomada junto com treinos.
- Não concluir nada de uma semente só: a baseline variou de 1,4% a 15,5% entre sementes.

## 7. Onde está cada coisa
- `rl/diag/`: PLANO.md, PREVISOES.md, RELATORIO.md, dados de TD, provas, nota(), td.py.
- `tools/medir_k.md`: baselines por k.
- `rl/RELATORIO.md`: lote 1 e lote 2a.
- **Pesos do PPO que fez 81,7%:** `rl/logs/ppo-fases-s0/modelo.pt` não vai para o Git (`*.pt` no
  .gitignore); o Gustavo mandou o arquivo à parte. Coloque-o nesse caminho para avaliar sem retreinar.
- Checkpoints e arrays de dados **não estão no Git**: os scripts regeneram os dados; os checkpoints
  dos lotes 1 e 2a ficaram na máquina do Gustavo.

## 8. Glossário rápido
- **k**: ticks entre decisões (10 no lote 1; 20 daqui em diante). **γ**: desconto (0,994 com k = 20).
- **V / A**: valor do tabuleiro / vantagem de uma jogada sobre as outras. Q = V + A.
- **td / mc**: aprender o valor por bootstrap / direto do retorno final.
- **dueling**: cabeça que separa V e A (cnn1d, cnn2, cnn3).
- **teto**: rede treinada com a resposta da prova; mede o máximo que a observação permite.
- **PPO**: método que aprende a política diretamente, com um crítico que estima o V.
- **gulosa / com sorteio**: avaliar o PPO pela ação mais provável / sorteando pela política.
