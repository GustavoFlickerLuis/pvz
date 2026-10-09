# Camada 2 do Desafio 3: as receitas do proxy transferem para o Double DQN?

Fila das 9 execuções: 08/10/2026, 18:56–22:16. Nada commitado.

## 1. O que foi implementado e as decisões

Pacote `rl/` (nada commitado, como pedido):

- `rl/dqn.py`: Double DQN com máscara. Alvo `y = R_n + γⁿ · Q_alvo(s', argmax_{a' válida} Q_online(s', a'))`,
  `y = R_n` se o episódio acabou dentro dos n passos. Huber δ = 1 só na ação tomada (a `perda_q`
  do desafio3, com one-hot). CNN de `desafio3/modelo.py`. Otimizadores e agendas vêm de
  `desafio3.treino.OTIMIZADORES` / `AGENDAS` (importados, não copiados), alimentados por uma
  `desafio3.treino.Config` com a receita; a agenda corre sobre o total de **atualizações**
  (`(passos // 4) − ((10.000 − 1) // 4)`). Clip 10 em todas.
- Replay de 100.000 transições de n passos (obs e s' em float16, máscara de s', R_n, γᵏ, fim).
  Treina a partir de 10.000 passos de jogo, 1 atualização a cada 4 passos, rede alvo copiada a
  cada 2.000 atualizações. Lote 256.
- Instrumentação: a classe `Registro` do desafio3, por herança (`RegistroRL`), a cada 50
  atualizações: perda TD, ‖g‖ antes do clip (global e por camada), ‖θ‖, ρ por camada, η de
  cada otimizador, mais passo de jogo, ε e Q médio previsto (na ação tomada, no lote).
  Por episódio (`episodios.csv`): retorno, vitória, decisões, ε.
- Avaliação gulosa (ε = 0, com máscara), com as partidas andando em lote (`rl/avaliar.py`):
  periódica a cada 50.000 passos em FASES_TREINO × 300–302 (63 partidas), final em
  FASES_TREINO × 300–319 (420). Nunca nas fases de teste (há `assert`).
- Parar/continuar: `rl/PARAR` checado no fim de cada episódio → checkpoint completo (online,
  alvo, otimizadores, agendas, replay, contadores, episódios, avaliações, Registro, rng de
  python/numpy/torch/cuda) e saída com código 3. Checkpoint automático a cada ~20 min.
  `rl/rodar.py`: as 9 execuções em fila, cada uma num processo, na ordem s0 das 3 receitas →
  s1 → s2; pula as terminadas e retoma a interrompida. `rl/STATUS.md` a cada avaliação.
- Regra do CLAUDE.md: o agente só usa `reset`, `step`, `observar` e `mascara`; vitória sai da
  última recompensa.

Decisões:

- **Retorno de n passos, n = 10.** Com 1 passo e ~250 decisões por partida, o +1 da vitória
  precisaria de muitíssimas atualizações para chegar ao começo da partida; com 10, cada
  atualização propaga 10 decisões.
- **Exploração com "esperar" a 50%.** Na ação aleatória, "esperar" sai com probabilidade 0,5;
  senão, planta válida uniforme. Sem isso, o agente explorando gasta todo o sol assim que
  aparece e nunca vê o valor de juntar sol. Medido na fase fácil: essa exploração pura vence 22%.
- **Sem currículo** nesta rodada: as 3 receitas veem a mesma distribuição de fases (as 21 de
  treino, sementes 1000–9999 sorteadas).
- **Orçamento: 400.000 passos de jogo por execução** (igual para as 9). Cálculo pela medida do
  teste 4: o Muon é o mais lento (~190 passos/s com ε = 0,05; ~220 sem a inferência do ator).
  Tempo estimado do Muon com B passos: ~10 s nos 10.000 iniciais (sem treino) + (0,4·B − 10.000)/205
  (ε decaindo) + 0,6·B/190 + ~90 s de avaliações ≤ 2.100 s (35 min) → B ≈ 400.000. Baseline e
  vencedora (~300–320 passos/s) levam ~22 min com o mesmo orçamento. Bem abaixo do teto de 1.000.000.
- **Ator na CPU.** Para agir, uma cópia da rede online na CPU (ressincronizada depois de cada
  atualização): uma inferência de 1 estado leva 0,7 ms na CPU contra 1,4 ms na GPU. Não muda
  a política (mesmos pesos).
- **passos/s fora do `registro.csv`.** Vai para `velocidade.csv`, porque depende do relógio e
  o `registro.csv` tem de sair idêntico na retomada.
- **Divergência no RL** = perda não finita (o critério do proxy, perda > 10× a inicial, não
  serve: a perda TD inicial é quase zero).
- O orçamento termina no meio de um episódio: esse episódio não entra em `episodios.csv`.
- Avaliação também no passo 0 (rede aleatória), para a curva começar do zero.

## 2. Testes antes do treino longo

1. **Unitários** (`tests/test_rl.py`, 12 testes; `pytest tests`: 54 passam): retorno de n
   passos (inclusive cortando no terminal e episódio menor que n), máscara no argmax do alvo
   (ação inválida de Q máximo nunca escolhida), Double DQN (argmax pela online, valor pela
   alvo; terminal sem bootstrap, mesmo com Q_alvo infinito), exploração respeita a máscara
   (e dá ~50% de esperar), receitas, contagem de atualizações, replay circular.
2. **Sanidade** (fase só "calmo", dificuldade 0,5; baseline; 100.000 passos; avaliação em
   sementes 300–399 a cada 10.000 passos): 3, 2, 40, 30, 1, 51, 43, 83, 83, **99, 95%**;
   final **94% em 200 partidas** (sementes 300–499). Passa de 80%. Referências na mesma fase:
   Guloso 100%, exploração pura 22%, só esperar 0%.
3. **Retomada exata** (3.000 passos; treino a partir do passo 500, alvo a cada 100
   atualizações, avaliação a cada 1.000, para exercitar tudo): a execução B foi parada por
   `rl/PARAR` (passos 1096, 1076 e 1017) e retomada. `registro.csv`, `episodios.csv`,
   `avaliacoes.csv` e `avaliacao_final.csv` **idênticos** aos da execução direta, nas **3
   receitas** (`python -m rl.teste_retomada <receita>`). Refeito depois da mudança para o ator
   na CPU. A fila real também passou por uma parada/retomada (baseline-s0, passo 28.083).
4. **Velocidade** com treino ligado e ε = 0,05 (pior caso: inferência em todo passo):
   baseline ~300, vencedora ~320, Muon ~190 passos/s. Uma atualização (lote 256, 3 passagens
   da CNN + backward) leva ~12 ms nesta GPU (RTX 3050 Laptop), limitada por cálculo.

## 3. Por execução

Todas terminaram os 400.000 passos. Nenhuma divergiu e o clip nunca cortou. Duração: baseline
19–23 min, vencedora 18–19 min, Muon 27–29 min (teto de 35). A baseline-s0 passou por uma
parada/retomada real com `rl/PARAR` no passo 28.083 (para trocar o log de UTF-16 por UTF-8).

| execução | passos | episódios | vitórias final (420) | dano final | curva periódica (63 partidas): 0k, 50k, …, 400k |
|---|---|---|---|---|---|
| baseline-s0 | 400.000 | 3.649 | 1,4% | 0,094 | 0, 0, 0, 3, 0, 0, 2, 2, 2% |
| baseline-s1 | 400.000 | 3.360 | **15,5%** | 0,312 | 0, 0, 0, 0, 0, 2, 6, 22, 17% |
| baseline-s2 | 400.000 | 3.474 | **10,7%** | 0,274 | 0, 0, 0, 0, 0, 3, 3, 6, 14% |
| vencedora-s0 | 400.000 | 4.256 | 0,0% | 0,028 | 0% em todas |
| vencedora-s1 | 400.000 | 3.895 | 0,0% | 0,031 | 0, 0, 0, 0, 3, 2, 0, 0, 0% |
| vencedora-s2 | 400.000 | 4.311 | 0,0% | 0,020 | 0% em todas |
| muon-s0 | 400.000 | 3.950 | 0,2% | 0,053 | 0, 0, 0, 0, 0, 2, 3, 2, 2% |
| muon-s1 | 400.000 | 3.892 | 0,0% | 0,025 | 0, 0, 0, 3, 0, 2, 0, 2, 0% |
| muon-s2 | 400.000 | 3.983 | 0,0% | 0,047 | 0, 0, 0, 2, 0, 0, 0, 0, 0% |

Uma vitória em 63 partidas vale 1,6%, então as curvas periódicas são ruidosas.

## 4. Por receita (3 sementes; 420 partidas cada)

| agente | vitórias final | dano final |
|---|---|---|
| baseline: adam 1e-4 constante | **9,2% ± 7,1** (1,4; 15,5; 10,7) | 0,227 |
| vencedora: sgd 3e-2 cosseno | 0,0% ± 0,0 | 0,026 |
| muon 1e-3 (AdamW 3e-4) warmup_cosseno | 0,1% ± 0,1 (0,2; 0; 0) | 0,041 |
| Econ5 (referência, mesmas 420 partidas) | 67,1% | 0,762 |
| Guloso (referência, mesmas 420 partidas) | 41,2% | 0,432 |

Pelo critério do Desafio 3 (|Δ| > 2 s_c), as duas comparações dão **empate formal**, por causa da
variância da baseline entre sementes:

- baseline × vencedora: Δ = 9,2 e s_c = 5,0, ou seja, 1,83 s_c;
- baseline × Muon: 1,81 s_c.

Ainda assim, as 3 sementes da baseline ficam acima de todas as 6 das outras receitas, e no
dano a separação é nítida (0,23 contra 0,03–0,04). Todas as receitas ficam muito longe do Guloso e do Econ5.

Distribuição de ações da política gulosa (63 partidas, `python -m rl.acoes`):

| execução | decisões/partida | esperar | girassol | ervilha | repetidora | batata | mina |
|---|---|---|---|---|---|---|---|
| baseline-s0 | 141 | 81,2% | 8,5% | 3,1% | 0,7% | 2,8% | 3,7% |
| baseline-s1 | 194 | 80,1% | 7,4% | 4,2% | 1,6% | 2,7% | 4,0% |
| baseline-s2 | 185 | 80,2% | 7,3% | 4,6% | 1,5% | 2,4% | 3,9% |
| vencedora-s0 | 98 | 90,5% | 2,7% | 0,7% | 0,0% | 1,8% | 4,3% |
| vencedora-s1 | 118 | 88,2% | 4,8% | 1,4% | 0,0% | 3,2% | 2,4% |
| vencedora-s2 | 99 | 89,9% | 4,0% | 0,9% | 0,2% | 2,2% | 2,9% |
| muon-s0 | 122 | 84,0% | 7,2% | 2,1% | 0,7% | 2,5% | 3,5% |
| muon-s1 | 107 | 87,1% | 5,0% | 0,7% | 0,0% | 3,0% | 4,3% |
| muon-s2 | 114 | 82,8% | 7,6% | 2,0% | 0,3% | 2,9% | 4,4% |

As baselines que vencem são as que mais plantam atiradores (ervilha + repetidora ≈ 6% das ações).
Também são as que sobrevivem mais: 185–194 decisões por partida, contra 256 do Econ5.

## 5. Figuras

- `rl/figuras/curvas_vitoria.png`: curvas de vitória das 3 receitas no mesmo eixo (média em
  linha grossa, cada semente em linha fina), com Econ5 e Guloso.
- `rl/figuras/sinais_baseline.png`, `sinais_vencedora.png`, `sinais_muon.png` (semente 0):
  perda TD, ‖g‖ antes do clip, ‖θ‖, ρ por camada, η, Q médio previsto e ε.

## 6. A ordem é a mesma do proxy?

**Não.**

- **No proxy** (perda menor = melhor): sgd cosseno e adam empatam, e os dois ficam muito à frente do Muon.
- **No DQN:** adam (a baseline) fica à frente, e Muon e SGD empatam atrás. A vencedora do proxy é a
  pior receita, com 0% nas 3 sementes.

O Muon continua atrás da baseline, como no proxy. A inversão é SGD × Adam.

O mecanismo aparece nos sinais (semente 0, mediana em janelas de 50k passos):

| | 50k | 150k | 250k | 350k |
|---|---|---|---|---|
| baseline: ρ global / ‖g‖ | 3,3e-4 / 0,32 | 2,9e-4 / 0,10 | 2,6e-4 / 0,09 | 2,4e-4 / 0,10 |
| vencedora: ρ global / ‖g‖ | 4,6e-4 / 0,13 | 1,2e-4 / 0,036 | 3,9e-5 / 0,021 | 3,1e-6 / 0,012 |
| muon: ρ global / ‖g‖ | 7,0e-4 / 0,15 | 4,3e-4 / 0,09 | 1,9e-4 / 0,09 | 2,4e-5 / 0,09 |

- **O SGD dá passos proporcionais a g.**
  - A perda TD tem gradientes pequenos (‖g‖ de 0,01 a 0,1, contra ~1 no proxy), e eles encolhem com o
    treino. O η = 3e-2 foi escolhido para a escala do proxy.
  - Já no passo 150k, com o η ainda em 0,025, o ρ caiu 4×. Com o cosseno por cima, chega a 3e-6.
  - A vencedora para de melhorar por volta dos 100k passos. Na s0, o dano por partida de treino fica
    em 0,025 entre 100k e 200k e em 0,026 entre 300k e 400k. Isso acontece justamente quando ε cai e a
    política começaria a gerar dados úteis.
- **O Adam normaliza o passo** pela escala do gradiente: o ρ fica em ~2,5e-4 do começo ao fim. Com η
  constante, a baseline continua aprendendo nos últimos 200k passos. Na s1, o dano por partida de
  treino vai de 0,04 (100–200k) a 0,12 (200–300k) e a 0,19 (300–400k).
- **O Muon também não depende da escala de g**, porque o passo é ortogonalizado; o ρ só cai com a
  agenda. Mas o warmup_cosseno leva o η a ~0 justamente na fase em que a baseline decola.
- **Conclusão:** a receita do proxy não transfere por dois motivos.
  - (a) O η do SGD depende da escala do gradiente, e essa escala muda do proxy (regressão para G,
    alvo fixo) para o TD (alvos pequenos e que se movem).
  - (b) Agendas que decaem até 0 supõem um problema estacionário, mas no RL os dados melhoram
    justamente no fim.

  É uma explicação tirada dos sinais, não um experimento. Separar (a) de (b) pediria, por exemplo,
  sgd 3e-2 constante e adam 1e-4 cosseno. Não rodei: seria ajuste novo, fora do pedido.

## 7. Coisas estranhas e limitações

- **O DQN quase não aprende com 400.000 passos** (~3.500 partidas).
  - Na fase fácil (teste 2), ele chega a 94% em 100.000 passos, então o laço funciona. Nas 21 fases
    reais, o orçamento de ~35 min por execução é curto.
  - Na baseline, as vitórias só aparecem depois dos 250k passos e ainda subiam no fim.
  - A comparação entre receitas vale para este orçamento.
- **A variância entre sementes é enorme** na baseline (1,4% a 15,5%). Com 3 sementes, o critério de
  2 s_c não separa as receitas, mesmo com a baseline à frente em todas.
- **As redes plantam minas e batatas** (~7% das ações) **e girassóis, mas poucos atiradores.** O
  Econ5 nunca planta mina nem batata. É o mesmo padrão da rede do proxy jogando
  (`desafio3/rede_joga.md`).
- **O Q médio previsto fica entre −0,65 e −0,7 em todas.** A rede aprende bem que perde (o alvo
  médio é perto de −γ^k), e a perda TD fica baixa (~3e-3) mesmo sem uma política boa. Perda TD baixa
  não indica política boa.
- **O ε fica no piso (0,05) desde os 160k passos**, e a baseline aprendeu quase tudo nessa fase gulosa.
- **A avaliação periódica com 63 partidas** tem resolução de 1,6 pontos e IC largo. Os números que
  valem são os finais, com 420 partidas.
- **Velocidade:** a GPU (RTX 3050 Laptop) é o gargalo, por conta do cálculo: ~12 ms por atualização
  de lote 256, com algoritmos determinísticos. O Muon custa ~50% a mais por atualização, por causa do
  Newton–Schulz.
- **O primeiro `rl/rodar.log` saiu em UTF-16, com acentos quebrados** (o `Tee-Object` do PowerShell
  5.1).
  - Parei a fila com `rl/PARAR`, converti o começo do log para `rl/rodar_inicio.log` e relancei com o
    redirecionamento do `cmd`.
  - A retomada foi exata, pelo mesmo mecanismo do teste 3.
- **Os checkpoints são apagados ao fim de cada execução**, porque o replay ocupa ~300 MB. Ficam
  `modelo.pt`, os CSVs e o `resultado.json` em `rl/logs/<id>/`.

## Arquivos

- **Código e testes:** `rl/dqn.py`, `rl/avaliar.py`, `rl/rodar.py`, `rl/status.py`, `rl/relatorio.py`,
  `rl/acoes.py`, `rl/teste_retomada.py` e `tests/test_rl.py`.
- **Resultados:** `rl/referencias.csv`, `rl/tabelas.md`, `rl/figuras/`, `rl/logs/` (as 9 execuções e
  as `teste-*`), `rl/STATUS.md` e `rl/rodar.log`.
- **`.gitignore`:** ganhou `rl/logs/*/ckpt/`, `rl/logs/*/ckpt_tmp/` e `rl/PARAR`.

## Lote 2a: exploração intencional

Previsão: a cnn1 sem currículo com exploração intencional planta mais atiradores na
política gulosa (ervilha + repetidora > 6% das ações) e vence mais que a baseline do
lote 1 (9,2% ± 7,1, 420 partidas). Critério: |Δ| > 2 s_c. Risco: com alvo, o agente
passa mais tempo sem gastar durante o treino (com Econ5 e ε = 0,05, as vitórias caem
de 62% para 41%), o que pode tirar dados do fim das fases.
