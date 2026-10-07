# Simulador de Plants vs. Zombies (pacote `pvz`)

Trabalho final de GBC073 Inteligência Computacional (FACOM/UFU), de Gustavo Luis
de Siqueira Nascimento e Pablo Rodrigues Cardoso Araújo. Um simulador próprio,
estilo *Plants vs. Zombies*, para treinar por aprendizado por reforço um agente
que decide a cada momento **o que plantar, onde, e quando esperar**.
Converse em português.

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
Todos estes números são valores iniciais e ficam em `pvz/config.py`.
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
cada intervalo (±30%), a composição (pelas proporções do `padrao`) e a faixa. O rng é
semeado com semente + classes, então fases diferentes não compartilham sorteio. A
intensidade cresce por segmento: `PESO_SEGMENTO = (1.0, 2.0, 2.25)` em `hordas.py`,
multiplicado pela `dificuldade`. O CONCEITO também pede:
- **Treino/teste:** reservar combinações de segmentos inéditas para o teste, **antes de treinar qualquer coisa**.
- Um **teste difícil** opcional, com uma classe nunca vista.
- **Três fases de vitrine**, usadas só para avaliação e demonstração.
- **Currículo:** começar em fases fáceis e aumentar a dificuldade.

## Plano de RL (CONCEITO 2.5, 2.7 e Parte 3)
- **Algoritmo:** Double DQN com máscara, γ = 0,99, ε de 1,0 a 0,05, rede alvo. PPO com máscara fica de reserva.
- **Rede principal:** uma CNN totalmente convolucional. O tronco tem três camadas conv 3×3. Uma conv 1×1 gera o mapa Q de 5×5×9, e um pooling global seguido de camada densa dá o Q(esperar). O baseline para ablação é um MLP.
- **Análise de erros (2.7):** clonar o ambiente numa decisão, testar alternativas, jogar até o fim com o próprio agente e marcar como erro quando alguma alternativa é claramente melhor. O resultado é um limite inferior dos erros.
- **Experimentos:** tudo roda com 3 sementes. São seis: curvas de aprendizado; comparação com os baselines (aleatório, guloso e uma heurística humana); taxa e tipos de erro; generalização (vistas × inéditas); ablações (CNN × MLP e com × sem currículo); e mapas de comportamento.
- **Fora de escopo:** visão computacional, o jogo real (só como bônus no fim), zumbis que trocam de faixa, voam ou cavam, e cliques para coletar sol.

## Estrutura do código
- `pvz/config.py`: todos os números do jogo. A calibração é feita aqui.
- `pvz/ambiente.py`: simulador determinístico com `reset`/`step`, máscara, observação em 20 canais, `clonar()` (para a análise de erros) e `info()`.
- `pvz/hordas.py`: classes de segmento (`Classe`/`Fila`), `PESO_SEGMENTO`, `montar_fase(classes, semente, dificuldade, pesos)`, `todas_as_fases()` e `fase_manual()` (para testes).
- `pvz/agentes.py`: agentes de referência `Aleatorio`, `Guloso` e `Econ(N)` (planta N girassóis com prioridade total, depois age como o guloso sem girassóis; no CLI: `econ3`, `econ4`, `econ5`, `econ6`, `econ8`).
- `pvz/render.py`: desenho do gramado em texto.
- `pvz/__main__.py`: CLI com os comandos `assistir`, `jogar`, `avaliar` e `bench`.
- `main.py`: atalho do PyCharm (roda `assistir` com o guloso).
- `tests/test_ambiente.py`: testes do ambiente.
- `tools/calibrar.py` e `tools/analisar.py`: calibração dos baselines (metas e uso no topo do arquivo).
- `requirements.txt`: numpy e pytest.

## Comandos
- `python -m pvz assistir --agente guloso --fase calmo,misto,blindados`
- `python -m pvz jogar`
- `python -m pvz avaliar --sementes 3 [--por-fase]`
- `python -m pvz bench` (passos/segundo; a CPU é o gargalo esperado)
- Testes: `python -m pytest tests`
- Calibração: `python tools/calibrar.py --girassol 120 --pesos 1 2 2.25 --saida r.json` e `python tools/analisar.py r.json`

## Andamento
<!-- Atualize esta seção ao fim de cada sessão de trabalho: o que foi feito, o que falta. -->
- [x] Simulador, hordas, agentes `Aleatorio` e `Guloso`, CLI e testes.
- [x] Calibração (INTERVALO_GIRASSOL = 120, PESO_SEGMENTO = (1.0, 2.0, 2.25)) e classes de hordas viraram distribuições. Em 27 fases × 5 sementes: aleatório 0%, guloso 38%, Econ(N) 25/52/67/54/36% para N = 3/4/5/6/8 (pico no meio), sol final médio < 200 em todos. O jogo é sensível ao girassol: com 100, o guloso vai a 88%.
- [ ] Ponto em aberto: 81 das 84 derrotas do guloso vêm de zumbis do segmento de início (no Econ5 as derrotas se espalham pelos três segmentos). `cedo_apressado` é a classe de início mais dura.
- [ ] Revisar o conteúdo das classes de hordas (ainda são um primeiro rascunho).
- [ ] Definir a separação treino/teste e as três fases de vitrine, antes de treinar.
- [ ] Escrever o baseline "heurística humana" (Parte 3, experimento 2).
- [ ] Implementar o Double DQN com CNN, o replay com máscara e o currículo.
- [ ] Implementar a análise de erros (2.7) e os experimentos da Parte 3.
