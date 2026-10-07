# Simulador de Plants vs. Zombies (pacote `pvz`)

Trabalho final de GBC073 Inteligência Computacional (FACOM/UFU), de Gustavo Luis
de Siqueira Nascimento e Pablo Rodrigues Cardoso Araújo. Um simulador próprio,
estilo *Plants vs. Zombies*, para treinar por aprendizado por reforço um agente
que decide a cada momento **o que plantar, onde, e quando esperar**.
Converse em português. Repo: https://github.com/GustavoFlickerLuis/pvz (privado).

## Documento de referência: CONCEITO
A fonte única de verdade é `docs/CONCEITO.md` (versão 2). O código cita seções dele
(ex.: 2.2, 2.3, 2.6). Se o código e o CONCEITO divergirem, o CONCEITO decide,
a menos que o usuário diga outra coisa. As seções são:
- **Parte 1:** a ideia. O dilema central é "economia agora ou defesa agora", e a consequência de cada escolha só aparece muito depois.
- **Parte 2:** a especificação. 2.1 jogo, 2.2 simulador, 2.3 estado e ação, 2.4 recompensa, 2.5 rede e algoritmo, 2.6 hordas, 2.7 análise de erros.
- **Parte 3:** os experimentos.
- **Parte 4:** as decisões fechadas e o que ficou de fora.
- **Parte 5:** os riscos.

A versão 1 usava modelo substituto com algoritmo genético. Foi descartada a pedido do professor: o trabalho agora é de RL.

## Regras do jogo (CONCEITO 2.1–2.4)
Os números ficam em `pvz/config.py` e já estão calibrados (ver Calibração).
- **Gramado:** 5 faixas × 9 colunas. Os zumbis entram na coluna 8 e nunca trocam de faixa. A derrota acontece quando um zumbi cruza a coluna 0 (não há cortador de grama). A vitória, quando as ondas acabam e todos os zumbis morrem.
- **Sol:** 100 no início, 25 do céu a cada 240 ticks e 25 por girassol a cada 120 ticks. A coleta é automática. O sol é compartilhado entre as faixas, e é isso que acopla o problema em 2D.
- **Plantas** (custo / recarga / vida):
  - Girassol: 50 / 50 / 300.
  - Ervilha: 100 / 50 / 300; causa 20 de dano a cada 15 ticks.
  - Repetidora: 200 / 50 / 300; dispara duas vezes.
  - Batata: 50 / 200 / 4000.
  - Mina: 25 / 200 / 300; arma em 150 ticks e causa 1800 de dano na célula.
- **Regra-chave:** atiradores só acertam zumbis **à direita ou na própria célula**.
- **Zumbis** (vida / ticks por coluna): comum 200/50, blindado 1100/70, rápido 150/25. Todos mordem 5 por tick.
- **Ordem fixa das operações no tick:** sol → spawn → plantas agem → zumbis bloqueados mordem → zumbis livres andam → remoção → checagem de fim. O dano é instantâneo (não há projétil).
- **Decisão a cada k = 10 ticks.**
- **Estado:** 20 canais × 5 × 9. O agente não vê os zumbis futuros; o canal 19 resume em quais faixas os zumbis vieram no segmento atual.
- **Ações:** 226 (5 plantas × 45 células + esperar), com máscara (sol, recarga, célula ocupada). A máscara é usada no argmax, na exploração e no alvo (a máscara de s' vai para o replay).
- **Recompensa:** dano do passo ÷ vida total da fase, mais +1 na vitória ou −1 na derrota. **Nunca recompensar meios** (sol coletado, plantas plantadas).

## Hordas (CONCEITO 2.6)
Uma fase é uma sequência de segmentos: início, meio e fim. Com 3 classes por
segmento, são 27 fases. Cada classe é uma distribuição com "personalidade", não um
roteiro: a semente sorteia a quantidade (±25%), o começo de cada fila (±100 ticks),
cada intervalo (±30%), a composição (pelas proporções do `padrao`) e a faixa. Com
roteiros fixos, o resultado dependia só da combinação de classes, e o agente poderia
decorar o calendário. O rng é semeado com semente + classes, então fases diferentes não
compartilham sorteio. A intensidade cresce por segmento (`PESO_SEGMENTO` em `hordas.py`,
multiplicado pela `dificuldade`). O currículo (fases fáceis primeiro) ainda falta.

## Calibração
`INTERVALO_GIRASSOL = 120` e `PESO_SEGMENTO = (1.0, 2.0, 2.25)`. Por quê:
- Antes (girassol a cada 60, sem pesos), o guloso vencia as 27 fases com ~3000 de sol
  sobrando, e toda derrota acontecia no início: quem sobrevivia ao começo vencia.
- Aumentar só a `dificuldade` não resolvia: o gargalo virava a recarga (perdia com
  milhares de sol sobrando) e o dilema economia × defesa sumia.
- Com sol mais escasso e meio/fim mais pesados, o investimento em economia tem um ponto
  ótimo no meio. Em 27 fases × 5 sementes: aleatório 0%, guloso 38%, Econ(N) 25/52/67/54/36%
  para N = 3/4/5/6/8, sol final médio < 200. O jogo é sensível ao girassol (com 100, o
  guloso vai a 88%).
- Ainda vale: 81 das 84 derrotas do guloso vêm de zumbis do início (as do Econ5 se
  espalham pelos três segmentos); `cedo_apressado` é a classe de início mais dura.

**Qualquer mudança de regra (config, classes, pesos) exige recalibrar** com `tools/`:
as metas estão no topo de `tools/calibrar.py`.

## Divisão treino / teste / vitrine (`pvz/divisao.py`)
- **Teste:** 6 combinações inéditas, com cada classe exatamente 2 vezes. **Treino:** as
  outras 21, que ainda contêm as 9 classes e os 9 pares início+meio.
- **Vitrine:** 3 fases do teste com sementes fixas (20000–20002), só para demonstração.
- **Sementes disjuntas:** treino de 0 a 9999, teste de 10000 a 19999, vitrine a partir de 20000.
- A divisão é fixa. Mudá-la depois de treinar invalida a generalização.
- **Quantas sementes:** pelo menos 20 por fase no teste e 50 nas avaliações finais. A
  margem de erro (IC 95%, pior caso) é de ±18 pontos com 30 partidas (6 fases × 5
  sementes), ±9 com 120 (× 20) e ±6 com 300 (× 50). Com 5 sementes, o Econ5 pareceu cair
  de 67% para 50% no teste; com 20, deu 64% nos dois.

## Baselines
- `Aleatorio`, `Guloso` e `Econ(N)` (planta N girassóis com prioridade total, depois age
  como o guloso sem girassóis).
- O **Econ(N) é a régua da curva de economia** (pico no meio = o dilema existe). Ele
  **não é** a "heurística humana" do experimento 2. Essa ainda falta e deve ser a mais
  forte que conseguirmos escrever, para a comparação com o agente valer alguma coisa.

## Plano de RL (CONCEITO 2.5, 2.7 e Parte 3)
- **Algoritmo:** Double DQN com máscara, γ = 0,99, ε de 1,0 a 0,05, rede alvo. PPO com máscara fica de reserva.
- **Rede principal:** uma CNN totalmente convolucional. O tronco tem três camadas conv 3×3. Uma conv 1×1 gera o mapa Q de 5×5×9, e um pooling global seguido de camada densa dá o Q(esperar). O baseline para ablação é um MLP.
- **Análise de erros (2.7):** clonar o ambiente numa decisão, testar alternativas, jogar até o fim com o próprio agente e marcar como erro quando alguma alternativa é claramente melhor. O resultado é um limite inferior dos erros.
- **Experimentos:** tudo roda com 3 sementes de treino. São seis: curvas de aprendizado; comparação com os baselines (aleatório, guloso e uma heurística humana); taxa e tipos de erro; generalização (vistas × inéditas); ablações (CNN × MLP e com × sem currículo); e mapas de comportamento.
- **Fora de escopo:** visão computacional, o jogo real (só como bônus no fim), zumbis que trocam de faixa, voam ou cavam, e cliques para coletar sol.
- **Pendente de decisão: γ = 0,997 em vez de 0,99.** Uma vitória leva ~300 decisões
  (260 a 360). Com 0,99, o +1 da vitória chega ao primeiro passo com peso 0,99³⁰⁰ ≈ 0,05
  (0,03 nas mais longas); com 0,997, ≈ 0,41.
  Ainda não está no CONCEITO: se for adotado, atualizar a seção 2.5.

## Desafio 3 (otimizadores)
- **Etapa 1, proxy supervisionado** (barato, sem a instabilidade do RL):
  - Dataset do Econ5 com 10–20% de ações aleatórias, **só nas fases e sementes de treino**.
  - Alvo G = retorno de Monte Carlo. Mesma CNN, mesma saída e mesma Huber loss do DQN.
  - Grade: cada otimizador com o seu melhor LR × schedulers (warmup+cosseno, restarts,
    warmup+constante) × 3 sementes.
  - Medir a loss, as normas de pesos e gradientes por camada, ‖Δw‖/‖w‖ e as ReLUs mortas.
- **Etapa 2:** DQN curto com os 2–3 finalistas, medindo a taxa de vitória.

## Estrutura do código
- `pvz/config.py`: todos os números do jogo.
- `pvz/ambiente.py`: simulador determinístico com `reset`/`step`, máscara, observação em 20 canais, `clonar()` (para a análise de erros) e `info()`.
- `pvz/hordas.py`: classes de segmento (`Classe`/`Fila`), `PESO_SEGMENTO`, `montar_fase(classes, semente, dificuldade, pesos)`, `todas_as_fases()` e `fase_manual()` (para testes).
- `pvz/divisao.py`: `FASES_TREINO`, `FASES_TESTE`, `VITRINE`, `SEMENTES_TREINO` e `SEMENTES_TESTE`.
- `pvz/agentes.py`: `Aleatorio`, `Guloso` e `Econ(N)` (no CLI: `econ3`, `econ4`, `econ5`, `econ6`, `econ8`).
- `pvz/render.py`: desenho do gramado em texto.
- `pvz/__main__.py`: CLI com os comandos `assistir`, `jogar`, `avaliar` e `bench`.
- `main.py`: atalho do PyCharm (roda `assistir` com o guloso).
- `tests/test_ambiente.py`: testes do ambiente, das hordas e da divisão.
- `tools/calibrar.py` e `tools/analisar.py`: calibração dos baselines (metas e uso no topo do arquivo).
- `requirements.txt`: numpy e pytest.

## Comandos
- `python -m pvz assistir --agente guloso --fase calmo,misto,blindados`
- `python -m pvz jogar`
- `python -m pvz avaliar --conjunto {treino,teste,vitrine,todas} --sementes 20 [--agentes guloso econ5] [--por-fase]`
- `python -m pvz bench` (passos/segundo com o agente aleatório; depende das regras, então só compare medições feitas com as mesmas regras)
- Testes: `python -m pytest tests`
- Calibração: `python tools/calibrar.py --girassol 120 --pesos 1 2 2.25 --saida r.json` e `python tools/analisar.py r.json`

## Andamento
<!-- Atualize esta seção ao fim de cada sessão de trabalho: o que foi feito, o que falta. -->
- [x] Simulador, hordas como distribuições, agentes `Aleatorio`, `Guloso` e `Econ(N)`, CLI e testes.
- [x] Calibração (ver acima).
- [x] Divisão treino/teste/vitrine.
- [ ] Decidir γ (0,99 ou 0,997).
- [ ] Revisar o conteúdo das classes de hordas (ainda são um primeiro rascunho); se mudar, recalibrar.
- [ ] Escrever o baseline "heurística humana" (Parte 3, experimento 2), o mais forte possível.
- [ ] Desafio 3: proxy supervisionado e DQN curto com os finalistas.
- [ ] Implementar o Double DQN com CNN, o replay com máscara e o currículo.
- [ ] Implementar a análise de erros (2.7) e os experimentos da Parte 3.
