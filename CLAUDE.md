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
Os números (custos, vidas, dano, ticks, sol) **não ficam aqui**: a fonte é
`pvz/config.py`, e a tabela legível está no CONCEITO 2.1. Os valores já estão calibrados
(ver Calibração). As regras que importam para programar:
- **Gramado:** 5 faixas × 9 colunas. Os zumbis entram pela direita e nunca trocam de faixa. A derrota acontece quando um zumbi cruza a coluna 0 (não há cortador de grama). A vitória, quando as ondas acabam e todos os zumbis morrem.
- **Sol:** a coleta é automática. O sol é compartilhado entre as faixas, e é isso que acopla o problema em 2D.
- **Recarga** por tipo de planta: sem ela, o agente encheria uma faixa no mesmo instante.
- **Regra-chave:** atiradores só acertam zumbis **à direita ou na própria célula**. Zumbi que chega numa planta para e come; a mina só explode armada e na própria célula.
- **Ordem fixa das operações no tick:** sol → spawn → plantas agem → zumbis bloqueados mordem → zumbis livres andam → remoção → checagem de fim. O dano é instantâneo (não há projétil).
- **Decisão a cada k ticks** (`K_TICKS`), com o simulador rodando sozinho entre decisões.
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
- **Algoritmo:** Double DQN com máscara, γ = 0,997, ε de 1,0 a 0,05, rede alvo. PPO com máscara fica de reserva.
- **Rede principal:** uma CNN totalmente convolucional. O tronco tem três camadas conv 3×3. Uma conv 1×1 gera o mapa Q de 5×5×9, e um pooling global seguido de camada densa dá o Q(esperar). O baseline para ablação é um MLP.
- **Análise de erros (2.7):** clonar o ambiente numa decisão, testar alternativas, jogar até o fim com o próprio agente e marcar como erro quando alguma alternativa é claramente melhor. O resultado é um limite inferior dos erros.
- **Experimentos:** tudo roda com 3 sementes de treino. São seis: curvas de aprendizado; comparação com os baselines (aleatório, guloso e uma heurística humana); taxa e tipos de erro; generalização (vistas × inéditas); ablações (CNN × MLP e com × sem currículo); e mapas de comportamento.
- **Fora de escopo:** visão computacional, o jogo real (só como bônus no fim), zumbis que trocam de faixa, voam ou cavam, e cliques para coletar sol.
- **γ = 0,997 (decidido; CONCEITO 2.5).** Uma vitória leva ~300 decisões (260 a 360).
  Com 0,99, o +1 da vitória chega ao primeiro passo com peso 0,99³⁰⁰ ≈ 0,05 (0,03 nas
  mais longas); com 0,997, ≈ 0,41.

## Regra de acesso ao ambiente (RL e dados)
Código de RL e de geração de dados **só usa `reset`, `step`, `observar` e `mascara`**
do `Ambiente`. Nunca `amb.fase.spawns`, `amb.zumbis`, `amb.plantas`, `amb.venceu` etc.:
`amb.fase.spawns` contém os zumbis futuros, e ler o estado interno vaza informação que
o agente não tem. (Vitória/derrota se deduz da última recompensa.) A exceção são os
baselines de `pvz/agentes.py`, que são regras escritas à mão. Usá-los como política de
coleta é permitido, mas o código que os chama continua preso a essa interface.

## Desafio 3 (otimizadores e agendas de η) — feito
Enunciado em `desafio3/enunciado.pdf`, hipóteses e critério em `desafio3/hipoteses.md`
(as previsões não se editam; os desfechos são escritos pelos autores).
- **Problema-proxy (supervisionado):** regressão de Q(s, a) para o retorno de Monte Carlo
  G (γ = 0,997), Huber δ = 1 só na ação tomada, com a CNN do CONCEITO 2.5. Dataset
  (`desafio3/dados.py`): Econ5 com 15% de ações aleatórias. Treino: FASES_TREINO ×
  sementes 0–59 (211 mil amostras); validação: FASES_TREINO × 100–109; teste:
  FASES_TESTE × **10000–10019** (`carregar("teste")` recusa sem `permitir_teste=True`).
  "Esperar" é 82% das ações.
- **Regras fixas:** 10.000 passos, lote 256, clip 10 em todas as execuções. A seleção é
  pela Huber de validação no último passo (val_final); também se registram val_esperar
  e val_plantio. Baseline ("receita atual"): Adam η = 1e-4, constante, wd 0.
- **Otimizadores:** sgd e nesterov (momento 0,9), adam, adamw (wd 0,01), rmsprop e um
  **Muon próprio** (`desafio3/muon.py`). **Agendas:** constante, cosseno, warmup_cosseno
  (warmup em 5%) e wsd (warmup 5%, decaimento linear nos últimos 20%).
- **Protocolo:** varredura de η com warmup_cosseno e 1 semente (5 η por otimizador,
  extensão na borda); agendas para os 2 melhores + Muon, com 1 semente; confirmação
  das 3 melhores combinações (+ Muon, como extra) com **5 sementes**. Critério de
  vitória: |Δ média| > 2 desvios-padrão combinados, s_c = √((s_A² + s_B²)/2).
- **Resultado:** as 3 finalistas (sgd 3e-2 cosseno, sgd 3e-2 warmup_cosseno, rmsprop 3e-4
  warmup_cosseno; val_final ≈ 0,1623) **empatam entre si e vencem a baseline**
  (0,1676 ± 0,0008). O Muon (melhor η 1e-3) **perde para todos**, inclusive para a
  baseline (0,1750). Nada divergiu, e o clip nunca chegou a cortar.
- O DQN curto **não faz parte do desafio**: fica para o projeto.
- **Sementes de teste:** o Desafio 3 já usou as sementes 10000–10019 das fases de teste.
  O teste final do agente de RL do projeto deve usar **outras sementes da faixa de teste,
  a partir de 10100**, para os dois testes serem independentes.

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
- `desafio3/dados.py`: gera o dataset em `desafio3/dados/` (arrays no .gitignore; `meta.json` versionado, com parâmetros, commit, tamanhos e hashes).
- `desafio3/modelo.py`: a CNN (saída 226 = mapa (planta, faixa, coluna) achatado + esperar).
- `desafio3/treino.py`: `Config`, registries `OTIMIZADORES` e `AGENDAS`, `Registro` (instrumentação) e o laço. Logs em `desafio3/logs/<id>/`, resumo em `desafio3/experimentos.csv`.
- `desafio3/muon.py`: Muon próprio (Newton–Schulz com 5 iterações, momento Nesterov 0,95, kernels achatados em (C_out, C_in·k·k), wd desacoplado). Pesos de conv1–3 no Muon; vieses, `cabeca_q` e `cabeca_esperar` no AdamW (`--lr-adamw`). Mesma agenda nos dois.
- `desafio3/analise.py`: tabelas do protocolo (`varredura`, `agendas`, `confirmacao`, `tabelas`) e figuras em `desafio3/figuras/`.
- `desafio3/graficos.py`: painéis de perda, ‖g‖, ‖θ‖, η e ρ de um log.
- `tests/test_desafio3.py`: índices da CNN, registries, agendas e Muon (valores singulares, divisão dos parâmetros, formato, determinismo).
- `requirements.txt`: torch (CUDA), numpy, pytest e matplotlib.

## Comandos
- `python -m pvz assistir --agente guloso --fase calmo,misto,blindados`
- `python -m pvz jogar`
- `python -m pvz avaliar --conjunto {treino,teste,vitrine,todas} --sementes 20 [--agentes guloso econ5] [--por-fase]`
- `python -m pvz bench` (passos/segundo com o agente aleatório; depende das regras, então só compare medições feitas com as mesmas regras)
- Testes: `python -m pytest tests`
- Calibração: `python tools/calibrar.py --girassol 120 --pesos 1 2 2.25 --saida r.json` e `python tools/analisar.py r.json`
- Desafio 3: `python -m desafio3.dados`; `python -m desafio3.treino --otimizador adam --lr 1e-4 --agenda constante --semente 0 --passos 10000 --etapa baseline [--rascunho]` (ou `--config x.json`); `python -m desafio3.graficos desafio3/logs/<id>`
- O treino recusa rodar com alterações não commitadas sem `--rascunho`. Rascunhos vão para `desafio3/logs/rascunho/` e não entram no `experimentos.csv`.

## Andamento
<!-- Atualize esta seção ao fim de cada sessão de trabalho: o que foi feito, o que falta. -->
- [x] Simulador, hordas como distribuições, agentes `Aleatorio`, `Guloso` e `Econ(N)`, CLI e testes.
- [x] Calibração (ver acima).
- [x] Divisão treino/teste/vitrine.
- [x] Decidir γ: 0,997.
- [ ] Revisar o conteúdo das classes de hordas (ainda são um primeiro rascunho); se mudar, recalibrar.
- [ ] Escrever o baseline "heurística humana" (Parte 3, experimento 2), o mais forte possível.
- [x] Desafio 3: proxy supervisionado e protocolo completo (varredura, agendas, confirmação).
- [ ] Desafio 3: extras, teste (uma vez), painel.pdf e slide.pdf.
- [ ] Implementar o Double DQN com CNN, o replay com máscara e o currículo.
- [ ] Implementar a análise de erros (2.7) e os experimentos da Parte 3.
