# Agente que aprende a jogar um tower defense em grid

**Trabalho Final — GBC073 Inteligência Computacional (FACOM/UFU)**
Gustavo Luis de Siqueira Nascimento · Pablo Rodrigues Cardoso Araújo

> Este documento é a nossa fonte única de verdade sobre o trabalho. Se a gente
> discordar de alguma coisa, é aqui que resolve. A **Parte 1** explica a ideia de
> forma intuitiva; a **Parte 2** é a especificação técnica; a **Parte 3** são os
> experimentos. Leia a Parte 1 inteira antes da Parte 2.
>
> **Versão 2.** A versão 1 propunha uma rede que *avaliava* tabuleiros estáticos
> (modelo substituto + algoritmo genético). A pedido do professor, o trabalho
> passou a ser de **aprendizado por reforço**: um agente que joga o jogo ao longo
> do tempo. O histórico da mudança está na seção 4.2.

---

## A ideia em uma frase

> Construir um simulador de um jogo estilo *Plants vs. Zombies* e treinar um
> agente, por aprendizado por reforço, que aprende a jogar sozinho — decidindo a
> cada momento **o que plantar, onde, e quando esperar**.

---

# PARTE 1 — Entendimento geral

## 1.1 O problema

Um gramado de 5 faixas por 9 colunas. Zumbis entram pela direita e caminham para
a esquerda, cada um na sua faixa. Se algum deles cruzar a borda esquerda,
perdemos.

Para se defender, o agente planta. Plantar custa **sol**, e o sol chega aos
poucos: um pouco cai do céu, e o resto é produzido por **girassóis** que o
próprio agente precisa plantar. Os zumbis chegam em ondas, primeiro poucos,
depois muitos.

A cada momento o agente olha o gramado e decide uma coisa só:

> **Planto alguma coisa agora? Se sim, o quê e onde? Ou espero?**

O tema é inspirado em *Plants vs. Zombies*, mas não é o jogo. Não lemos a tela,
não usamos visão computacional, não controlamos o jogo real. Construímos nosso
próprio simulador, e é nele que o agente treina e é avaliado.

## 1.2 O dilema que torna o problema interessante

O coração do jogo é um dilema temporal:

> **Investir em economia agora e ficar vulnerável, ou defender agora e ficar
> pobre depois?**

Um girassol plantado no começo não mata nenhum zumbi. Ele só "se paga" dezenas
de segundos depois, quando o sol que produziu vira atiradores. Se o agente
planta girassóis demais, é atropelado pela primeira onda. Se planta de menos,
não tem sol para a onda final.

A consequência de uma decisão aparece **muito depois** dela. É exatamente o tipo
de problema para o qual o aprendizado por reforço existe — e é o que uma regra
fixa ("sempre gaste assim que der") não resolve bem.

## 1.3 Como o agente aprende: a analogia do jogador iniciante

Pense em alguém que nunca jogou e não leu nenhum manual.

Nas primeiras partidas, ele clica mais ou menos ao acaso e perde. Mas percebe
coisas: "quando plantei atirador naquela faixa, o zumbi morreu"; "perdi porque
não tinha sol no final". Aos poucos associa **situações** a **ações que deram
certo**, e passa a jogar melhor.

O agente faz a mesma coisa, só que milhares de vezes mais rápido e com
matemática no lugar da intuição:

| O jogador iniciante | O nosso agente |
|---|---|
| Olha a tela | Recebe o **estado** do simulador |
| Decide o que fazer | Escolhe uma **ação** |
| Vê se deu certo | Recebe uma **recompensa** |
| Com a experiência, joga melhor | A **rede neural** é ajustada para preferir ações que levam a mais recompensa |

Ninguém ensina ao agente quais jogadas são boas. Ele descobre jogando.

## 1.4 As três peças

### O simulador — "o jogo"

Recebe uma ação, avança o tempo, devolve o novo estado e uma recompensa. **Não
aprende nada.** É física pura e determinística: mesma semente e mesmas ações
geram sempre a mesma partida.

### O agente — "o jogador"

Uma rede neural que olha o estado e dá uma nota para cada ação possível: "quanto
eu espero ganhar, daqui até o fim da partida, se fizer isso agora". Ele escolhe a
de nota mais alta. **É aqui que está a inteligência computacional do trabalho.**

### As hordas — "as fases"

O conjunto de ondas de zumbis contra as quais o agente treina e é testado. Como
elas são montadas decide se o agente aprende a jogar ou só decora fases
(seção 2.6).

## 1.5 O fluxo

```
     ┌─────────────────────────┐
     │        SIMULADOR        │
     │   o jogo, determinístico│
     └───────┬─────────▲───────┘
    estado,  │         │  ação
 recompensa  │         │
             ▼         │
     ┌─────────────────┴───────┐
     │         AGENTE          │
     │ rede neural (CNN) que   │
     │ pontua cada ação        │
     └───────┬─────────────────┘
             │ experiências (estado, ação, recompensa, próximo estado)
             ▼
     ┌─────────────────────────┐
     │   MEMÓRIA DE REPLAY     │
     │ a rede é ajustada com   │
     │ lotes sorteados daqui   │
     └─────────────────────────┘
```

**Etapa 0 — Construir o simulador.** Python, sem interface gráfica, com a
interface padrão de ambientes de RL: `reset()` e `step(ação)`.

**Etapa 1 — Baselines sem aprendizado.** Um agente aleatório e duas heurísticas
escritas à mão. Eles dão a régua: se o agente treinado não ganhar delas, algo
está errado.

**Etapa 2 — Treinar o agente.** Double DQN com máscara de ações, começando por
fases fáceis e aumentando a dificuldade (currículo).

**Etapa 3 — Avaliar.** Contra os baselines, em fases que o agente nunca viu, e
procurando as jogadas erradas que ele ainda comete.

## 1.6 O que o agente precisa aprender sozinho

Nenhuma destas regras será programada. Se o agente jogar bem, é porque as
descobriu:

- **Economia primeiro, mas não demais.** Plantar girassóis cedo, enquanto não há
  zumbis, e parar na hora certa.
- **Batata na frente do atirador.** O tanque protege o atirador; atrás dele, não
  serve para nada.
- **Mina com antecedência.** A mina demora para armar. Plantar quando o zumbi já
  está em cima é inútil.
- **Dano concentrado contra blindado.** Uma faixa com blindado pede repetidora ou
  mais de um atirador.
- **Onde está a ameaça.** O sol é um só para as cinco faixas. Gastar numa faixa
  vazia é desperdiçar recurso que falta em outra.
- **Às vezes esperar é a melhor jogada.** Juntar sol para uma repetidora pode
  valer mais que plantar uma ervilha agora.

## 1.7 Três sutilezas que decidem se dá certo

### Sutileza 1 — A recompensa precisa ser graduada, não só "venceu/perdeu"

Se o agente só recebe sinal no fim da partida, nas primeiras milhares de partidas
ele perde sempre e não aprende nada. Por isso ele recebe também recompensa
pequena **durante** a partida, proporcional ao dano causado. Cuidado: tudo o que
é recompensado vira objetivo. Nada de recompensar sol coletado ou plantas
plantadas — o agente aprenderia a encher o gramado de girassóis e perder. Detalhes
na seção 2.4.

### Sutileza 2 — O agente não decide a cada tick

Uma partida tem milhares de ticks. Se o agente decidisse em todos, quase todas as
decisões seriam "esperar" e ficaria muito difícil ligar uma vitória às poucas
decisões que importaram. Ele decide a cada **k ticks fixos** (seção 2.3).

> **Por que não decidir "sempre que chega um sol":** aí a frequência de decisão
> dependeria de quantos girassóis o agente plantou, e o "passo" do aprendizado
> teria duração diferente em cada momento do jogo. Além disso, momentos urgentes
> (a batata acabou de cair) não esperam o sol.

### Sutileza 3 — Treinar em poucas fases é decorar

Se o agente treinar sempre nas mesmas três fases, ele vai decorar onde cada zumbi
aparece. Por isso as fases são montadas combinando **segmentos** de horda, e
algumas combinações ficam guardadas para teste desde o primeiro dia (seção 2.6).

## 1.8 A pergunta que o professor vai fazer

> *"Já existem agentes de RL para Plants vs. Zombies. O que vocês fizeram de
> diferente?"*

Existem, sim, e vamos citá-los (seção 4.3). A resposta tem três partes:

1. **Generalização medida de verdade.** Fases montadas por combinação de
   segmentos, com combinações inéditas reservadas para teste.
2. **Análise de erros.** Como o simulador é determinístico, dá para voltar a
   qualquer momento de uma partida, trocar a jogada do agente por outra e ver o
   que teria acontecido. Isso mede quantas jogadas erradas ele ainda comete e
   de que tipo (seção 2.7). Os trabalhos existentes olham só o placar final.
3. **O que o agente aprendeu.** Verificamos se ele descobre as regras da seção
   1.6, com mapas de onde e quando planta cada coisa.

---

# PARTE 2 — Especificação técnica

> Todos os números desta parte são **valores iniciais**. Eles serão calibrados
> para que o agente aleatório perca quase sempre e uma boa heurística vença as
> fases médias. Um jogo fácil demais ou impossível não ensina nada.

## 2.1 O jogo

| Elemento | Definição |
|---|---|
| **Gramado** | 5 faixas × 9 colunas = 45 células. Coluna 0 é a borda de casa, coluna 8 é por onde os zumbis entram |
| **Tempo** | Ticks discretos. Referência intuitiva: 1 tick ≈ 0,1 s de jogo |
| **Movimento** | Zumbis têm posição contínua ao longo da faixa e andam para a esquerda. **Nunca trocam de faixa** |
| **Bloqueio** | Zumbi que chega a uma célula com planta para e come a planta até ela morrer |
| **Derrota** | Qualquer zumbi cruza a coluna 0 (não há cortador de grama) |
| **Vitória** | Todas as ondas da fase terminam e todos os zumbis morrem |

### Sol

| Fonte | Quantidade | Frequência |
|---|---|---|
| Sol inicial | 100 | uma vez, em t = 0 |
| Céu | 25 | a cada 240 ticks |
| Girassol | 25 | a cada 120 ticks, por girassol |

O céu é 2× mais lento que um girassol, de propósito: é o suficiente para o agente
nunca ficar travado sem sol, mas não para vencer sem investir em girassóis. A
coleta é automática.

### Plantas

| Planta | Custo | Recarga | Vida | Comportamento |
|---|---|---|---|---|
| **Girassol** | 50 | 50 | 300 | Produz sol |
| **Ervilha** | 100 | 50 | 300 | Dispara 20 de dano a cada 15 ticks no zumbi mais próximo **à sua direita ou na própria célula**, na mesma faixa |
| **Repetidora** | 200 | 50 | 300 | Igual à ervilha, dois disparos |
| **Batata (tanque)** | 50 | 200 | 4000 | Não causa dano. Segura o avanço |
| **Mina** | 25 | 200 | 300 | Leva 150 ticks para armar. Armada, explode quando um zumbi entra na célula: 1800 de dano em todos os zumbis da célula. Uso único |

A **recarga** é o tempo mínimo entre dois plantios do mesmo tipo. Sem ela, um
agente com sol sobrando encheria uma faixa inteira no mesmo instante.

### Regra que define o posicionamento

> **Atiradores só atingem zumbis à sua direita (ou na própria célula).**

Se atirassem para os dois lados, a coluna deixaria de importar. Atirando só para
frente, surge o trade-off: a planta mais à direita começa a atirar antes, mas é a
primeira a ser comida.

### O que acopla as faixas

Os zumbis nunca trocam de faixa e nenhuma planta atinge faixas vizinhas. O que
transforma o problema em um problema 2D é o **sol, que é compartilhado**: cada
sol gasto numa faixa falta nas outras. Decidir *qual faixa defender com recurso
escasso* é a dimensão estratégica do gramado.

### Zumbis

| Tipo | Vida | Velocidade | Mordida |
|---|---|---|---|
| **Comum** | 200 | 1 coluna a cada 50 ticks | 5 por tick |
| **Blindado** | 1100 | 1 coluna a cada 70 ticks | 5 por tick |
| **Rápido** | 150 | 1 coluna a cada 25 ticks | 5 por tick |

## 2.2 O simulador

```
ambiente.reset(fase, semente)  -> estado
ambiente.step(ação)            -> estado, recompensa, fim
```

Determinístico: mesma fase, mesma semente e mesmas ações geram sempre a mesma
partida.

### Ordem das operações dentro de um tick

Fixar e não mudar:

1. **Sol** — céu e girassóis produzem, se for o tick deles
2. **Spawn** — zumbis previstos para este tick entram na coluna 8
3. **Plantas agem** — atiradores disparam, minas armadas explodem
4. **Zumbis bloqueados mordem** — e não andam
5. **Zumbis livres andam**
6. **Remoção** — sai o que chegou a zero de vida
7. **Fim?** — algum zumbi cruzou a coluna 0? A fase acabou?

> **Simplificação aceita:** o dano é instantâneo, não há projétil voando. Isso
> preserva toda a estratégia e simplifica bastante.

### Desempenho

RL precisa de centenas de milhares a milhões de passos. **Medir passos por
segundo logo que o simulador rodar** e, se necessário, vetorizar com numpy e
rodar vários ambientes em paralelo. A GPU dá conta da rede; o gargalo será a CPU.

## 2.3 O agente vê e age

### Frequência de decisão

O agente decide a cada **k = 10 ticks**. Entre duas decisões, o simulador roda
k ticks sozinho. k precisa ser bem menor que o intervalo de produção do girassol
(120 ticks), para o agente nunca demorar muito a poder gastar um sol.

### Estado

Um tensor de canais sobre o gramado, como uma imagem com camadas:

| Canais | Conteúdo |
|---|---|
| 0–4 | One-hot da planta em cada célula (girassol, ervilha, repetidora, batata, mina) |
| 5 | Vida da planta na célula (normalizada) |
| 6 | Mina armada (0/1) |
| 7–9 | Quantidade de zumbis de cada tipo na célula |
| 10 | Vida somada dos zumbis na célula (normalizada) |
| 11 | Sol atual (mesmo valor em todas as células) |
| 12–16 | Recarga restante de cada planta (mesmo valor em todas as células) |
| 17 | Progresso da fase (mesmo valor em todas as células) |
| 18 | Índice da coluna (a rede precisa saber a distância até a casa) |
| 19 | Zumbis que já surgiram em cada faixa no segmento atual (mesmo valor ao longo da linha) |

**O que o agente vê e o que não vê.** Ele vê tudo que está no gramado agora:
cada planta, cada zumbi, vidas, sol e recargas. Ele **não vê** os zumbis que
ainda vão entrar — como no jogo, que não avisa em qual faixa virá o próximo.

**Por que não precisa do histórico.** O passado só ajuda a decidir se contém
informação que o presente não tem. Em jogos de Atari, por exemplo, uma única
imagem não mostra para onde a bola está indo, então se empilham os últimos
quadros. Aqui isso não acontece: a velocidade de cada zumbi é fixa pelo tipo,
que está no estado. A única informação do passado que ajuda a prever o futuro é
*em quais faixas os zumbis têm vindo* (num segmento "uma faixa pesada", os
primeiros zumbis revelam a faixa dos próximos). Em vez de uma rede recorrente,
resumimos isso no canal 19. Com ele, o estado atual basta para decidir.

### Ação

5 plantas × 45 células + "esperar" = **226 ações discretas**.

Ações inválidas são **mascaradas**: sol insuficiente, recarga em andamento ou
célula ocupada. "Esperar" é sempre válida.

A cada passo, o simulador devolve, junto com o estado, um vetor de 226
verdadeiro/falso dizendo quais ações são válidas. A máscara entra em três
lugares:

```
# 1. escolher a ação: Q das inválidas vira -infinito antes do argmax
q = rede(estado)                       # 226 valores
q[~mascara] = -inf
acao = argmax(q)

# 2. explorar: sorteio aleatório só entre as válidas
acao = sorteia(indices onde mascara é verdadeira)

# 3. treinar: o "melhor do próximo estado" também só entre as válidas
#    (por isso a máscara de s' é guardada na memória de replay)
q_prox[~mascara_prox] = -inf
```

A alternativa seria deixar o agente tentar o lance ilegal e puni-lo. É pior: ele
gasta boa parte do treino aprendendo regras que já sabemos, e a punição
contamina a recompensa.

## 2.4 A recompensa

| Parcela | Valor | Quando |
|---|---|---|
| **Dano** | dano causado no passo ÷ vida total da fase | a cada passo — soma no máximo 1 na partida |
| **Vitória** | +1 | no fim, se venceu |
| **Derrota** | −1 | no fim, se perdeu |

O retorno de uma partida fica aproximadamente em `[−1, 2]`.

Três cuidados:

> **Não recompensar meios, só fins.** Sol coletado e plantas plantadas não
> entram na recompensa. O agente precisa descobrir que girassóis valem a pena
> *porque* levam a mais dano e a vencer.

> **A parcela de dano não pode dominar.** Se valesse muito, o agente poderia
> aprender a manter zumbis vivos para continuar causando dano. A vitória e a
> derrota precisam pesar mais.

> **Assistir às partidas, não só olhar as curvas.** Agentes de RL exploram
> qualquer brecha da recompensa. Uma curva subindo pode esconder um comportamento
> absurdo.

## 2.5 A rede e o algoritmo

### Algoritmo: Double DQN com máscara de ações

A rede aprende **Q(estado, ação)**: a recompensa total que se espera obter, do
momento atual até o fim da partida, fazendo aquela ação e jogando bem depois.

```
para cada partida:
    s = ambiente.reset(fase, semente)
    enquanto não acabou:
        a = ação válida de maior Q(s)        (ou aleatória, com probabilidade ε)
        s', r, fim = ambiente.step(a)         # roda k ticks
        guarda (s, a, r, s', fim) na memória de replay
        sorteia um lote da memória
        alvo = r + γ · Q_alvo(s', melhor ação válida segundo a rede principal)   (só r se fim)
        ajusta a rede para Q(s, a) se aproximar do alvo
        s = s'
```

A intuição: o valor de uma ação é a recompensa imediata mais o valor do melhor
que dá para fazer depois. É assim que a recompensa de vencer "escorre" para trás,
passo a passo, até a decisão de plantar um girassol lá no começo.

Hiperparâmetros iniciais: `γ = 0,99` (o agente enxerga ~100 decisões à frente,
≈ 1000 ticks), ε decaindo de 1,0 a 0,05, rede alvo atualizada periodicamente.

Por que DQN: a ação é discreta, o simulador é barato de reiniciar e métodos
off-policy se saíram melhor em ambientes parecidos (seção 4.3). PPO com máscara
fica como alternativa se o DQN instabilizar.

### Arquitetura: CNN totalmente convolucional

A ação "plantar X na célula (i, j)" tem a mesma geometria do gramado. Então a
rede devolve um **mapa de Q-valores 5 × 5 × 9** — um valor por planta por
célula, alinhado com a célula onde a ação acontece — mais uma saída escalar para
"esperar", calculada a partir de um pooling global.

```
estado (20 × 5 × 9)
   → conv 3×3 → conv 3×3 → conv 3×3        (tronco)
   ├→ conv 1×1 → mapa Q (5 × 5 × 9)         (uma saída por planta por célula)
   └→ pooling global → densa → Q(esperar)
```

Vantagem: os mesmos filtros percorrem o gramado inteiro, então o que a rede
aprende sobre uma célula vale para as outras. Um **MLP que achata tudo** é o
baseline de comparação (seção 3).

## 2.6 As hordas e as fases

### Estrutura

Uma **fase** é uma sequência de **segmentos**, e cada segmento é uma onda com
duração, composição e intensidade definidas:

| Segmento | Papel | Exemplo de classes |
|---|---|---|
| **Início** | Poucos zumbis, espaço para investir em economia | calmo, cedo-apressado, só-rápidos |
| **Meio** | Pressão crescente | misto, uma-faixa-pesada, rápidos-em-massa |
| **Fim** | Onda grande | blindados, tudo-junto, duas-ondas |

Cada classe é uma **distribuição**, não um roteiro fixo. Ela tem uma
"personalidade" (que tipos de zumbi costumam entrar, em que ritmo, em que parte do
segmento, se concentrados numa faixa), e a semente sorteia a partida concreta:

| O que a semente sorteia | Variação em torno do valor da classe |
|---|---|
| Quantidade de zumbis | ±25% |
| Tick de começo de cada fila de zumbis | ±100 ticks, sem sair do segmento |
| Cada intervalo entre zumbis | ±30%, individualmente |
| Composição | cada zumbi sorteado com as proporções da classe |
| Faixa | uniforme (na "uma faixa pesada", a fila principal vai toda para uma faixa sorteada) |

Assim, a mesma combinação de classes conta histórias diferentes a cada semente, e
o agente não consegue decorar um calendário. A geração é determinística: mesmas
classes, mesma semente e mesma dificuldade geram sempre a mesma fase. O sorteio
depende também das classes, então fases diferentes não compartilham a sequência.

A **intensidade** cresce ao longo da fase: cada segmento tem um peso (início 1,0,
meio 2,0, fim 2,25) que, multiplicado pela dificuldade, aumenta o número de zumbis
e comprime os intervalos, mantendo a duração do segmento.

Com 3 classes por segmento, são 3 × 3 × 3 = **27 fases** diferentes, cada uma
jogável com infinitas sementes.

**Calibração atual** (27 fases × 5 sementes): aleatório 0% de vitórias, guloso
38%; Econ(N), que planta N girassóis antes de defender, vence 25% com N = 3, 52%
com 4, 67% com 5, 54% com 6 e 36% com 8. O melhor investimento fica no meio da
curva: pouca economia não sustenta a defesa e muita economia não sobrevive ao
começo. Esse é o dilema da Parte 1.

### Separação treino / teste — decidida antes de treinar

- **Treino:** parte das combinações de segmentos.
- **Teste de generalização:** combinações que o agente **nunca viu**, montadas
  com segmentos que ele conhece. Ex.: no treino, o "fim-blindados" só aparece
  depois do "meio-misto"; no teste, aparece depois do "meio-rápidos-em-massa".
- **Teste difícil (opcional):** uma classe de segmento inteira nunca vista.
- **Vitrine:** 3 fases pré-montadas, bonitas de mostrar, usadas **só para
  avaliação e demonstração**.

### Currículo

O agente começa treinando em fases fáceis (hordas menores) e a dificuldade sobe
conforme ele passa a vencer. Sem isso, é comum o agente perder todas as partidas
no começo e nunca ver uma vitória para aprender com ela.

## 2.7 Análise de erros

Calcular *a melhor* jogada possível é inviável no gramado completo. Mas julgar
se uma jogada foi **errada** é possível, porque o simulador é determinístico e
pode ser copiado a qualquer momento:

```
para uma amostra de decisões do agente em partidas de teste:
    copia o simulador naquele momento
    para algumas ações alternativas válidas:
        aplica a alternativa
        joga o resto da partida com o próprio agente
    se alguma alternativa gerou retorno claramente maior
    (ou vitória onde o agente perdeu):
        marca a decisão original como erro
```

Isso dá a **taxa de erros** do agente e, olhando os casos marcados, o **tipo** de
erro: mina plantada tarde, girassol demais antes de uma onda, faixa ameaçada
ignorada, sol parado sem uso. A taxa deve cair ao longo do treino.

Dois cuidados: a comparação é só contra as alternativas testadas, então a taxa
é um **limite inferior** dos erros reais; e o custo é uma partida simulada por
alternativa, então usamos uma amostra de decisões, não todas.

---

# PARTE 3 — Os experimentos

Toda configuração roda com **3 sementes**; reportamos média e desvio. Poucos
experimentos bem feitos em vez de muitos com uma semente só.

| # | Experimento | O que prova |
|---|---|---|
| 1 | **Curvas de aprendizado** (retorno e taxa de vitória × passos de treino) | O agente aprende |
| 2 | **Comparação com baselines**: aleatório, guloso ("gasta assim que puder, na faixa mais ameaçada"), heurística humana escrita por nós | O agente joga melhor que regras fixas |
| 3 | **Análise de erros** (seção 2.7): taxa e tipos de jogadas erradas | Onde o agente ainda falha, e se isso melhora com o treino |
| 4 | **Generalização**: combinações de segmentos vistas vs. nunca vistas | Aprendeu a jogar ou decorou fases? **A pergunta científica** |
| 5 | **Ablações**: CNN totalmente convolucional vs. MLP; com vs. sem currículo | Quais escolhas de projeto importam |
| 6 | **Comportamento**: mapas de onde planta cada tipo, curva de girassóis ao longo da partida | Se ele descobriu as regras da seção 1.6 |

**Sobre o experimento 4:** separar as combinações **antes** de treinar qualquer
coisa. Se avaliarmos sempre nas mesmas fases, vamos ajustar tudo a elas sem
perceber — a mesma armadilha das tarefas ocultas do Desafio 1.

**Sobre o experimento 2:** as heurísticas vêm **antes** do agente. Se o DQN não
ganhar do guloso, descobrimos cedo que o problema está fácil demais ou que a
recompensa está mal desenhada.

---

# PARTE 4 — Decisões fechadas

## O que está dentro

| Item | Decisão |
|---|---|
| Simulador | Python, ticks discretos, dano instantâneo, tiro só para a direita, determinístico por semente, interface `reset`/`step` |
| Gramado | 5 × 9 |
| Economia | Sol inicial + céu + girassóis, coleta automática |
| Plantas | Girassol, ervilha, repetidora, batata (tanque), mina |
| Zumbis | Comum, blindado, rápido |
| Decisão | A cada k ticks fixos, com "esperar" como ação |
| Algoritmo | Double DQN com máscara de ações |
| Rede | CNN totalmente convolucional (principal), MLP (baseline) |
| Recompensa | Dano normalizado por passo + vitória/derrota |
| Fases | Composição de segmentos início/meio/fim, combinações reservadas para teste |

## O que está fora, e por quê

| Descartado | Motivo |
|---|---|
| **Visão computacional** | Viraria projeto de visão, não de decisão |
| **Treinar no jogo real** | O jogo roda em tempo real e o agente precisa de milhões de passos. Controlar o jogo exige ler a memória do processo, um projeto de engenharia à parte |
| **Cortador de grama, zumbis que trocam de faixa, voam ou cavam** | Complicam o simulador sem mudar a natureza do problema |
| **Clicar para coletar sol** | Não tem nada de estratégico |
| **Mais plantas e zumbis** | Entram só se o problema ficar fácil demais (ver riscos) |

### Extensão futura: transferir para o jogo real

Se sobrar tempo no fim, rodar o agente treinado no *Plants vs. Zombies* original,
via ferramentas que leem e controlam o processo do jogo. Só funciona se o
simulador usar os mesmos números do jogo, senão o agente aprendeu um jogo
diferente. **Não é escopo; é bônus.**

## 4.1 Por que decidir ao longo do tempo, e não de uma vez

Uma alternativa seria a rede receber a fase e devolver o gramado inteiro pronto,
de uma vez. Descartamos, e o motivo é instrutivo:

- A saída seria 45 escolhas simultâneas. Uma rede recém-inicializada quase
  sempre estouraria o orçamento, e o caminho mais fácil para respeitá-lo seria
  não plantar nada — o treino trava na largada.
- Uma única recompensa no fim para 45 escolhas dá variância altíssima: não há
  como saber qual das 45 foi a boa.

A formulação sequencial resolve os dois: **uma ação por passo**, ações inválidas
mascaradas (nunca estoura o orçamento) e recompensa ao longo da partida.

## 4.2 Histórico: da versão 1 para a versão 2

A versão 1 tratava o posicionamento como **uma decisão única no instante zero**,
sem economia ao longo do tempo. Uma rede aprendia a *avaliar* tabuleiros por
regressão supervisionada, e um algoritmo genético usava essa rede para buscar o
melhor tabuleiro (modelo substituto).

A versão 2 devolve ao jogo a economia e a passagem do tempo, e troca a pergunta
"qual o melhor tabuleiro?" por **"qual a melhor jogada agora?"**. O que foi
aproveitado:

- O **simulador** e suas regras (tiro só para a direita, ordem fixa das
  operações, determinismo por semente).
- A **recompensa graduada**, que na versão 1 era o alvo da regressão e agora
  vira a recompensa de dano por passo.
- A **CNN** sobre o gramado, agora como rede do agente.
- A **separação de hordas para generalização** desde o início.

## 4.3 Trabalhos relacionados

Existem agentes de RL para jogos estilo *Plants vs. Zombies*, que usaremos como
referência e citaremos no relatório:

- **pvz_rl** (curso INF581) — motor do jogo + ambiente Gym, com policy gradient,
  actor-critic, DQN e Double DQN; relata vantagem dos métodos off-policy.
  https://github.com/inf581-pvz-anonymous/pvz_rl
- **Plant-Vs-Zombie-RL-Experiment** — extensão do anterior, comparando DDQN,
  actor-critic, heurística e aleatório.
  https://github.com/Particle9/Plant-Vs-Zombie-RL-Experiment
- **PvZAI** — Q-learning por faixa + algoritmo genético para escolher a faixa
  mais urgente. https://github.com/abzikro/PvZAI

**Regra nossa:** podemos estudar como eles modelaram estado, ação e recompensa,
mas o simulador e o agente são nossos, e o relatório deixa claro o que é
diferente (seção 1.8).

---

# PARTE 5 — Riscos

| Risco | Mitigação |
|---|---|
| **O simulador consumir mais tempo que a IA** | Prazo fixo: duas semanas. É infraestrutura, não é o trabalho. Passou disso, simplifica |
| **Simulador lento demais para RL** | Medir passos/segundo cedo; vetorizar; ambientes em paralelo |
| **Agente não aprende (recompensa esparsa)** | Recompensa de dano por passo, currículo |
| **Agente explora brecha na recompensa** | Recompensar só fins; assistir às partidas |
| **Treino instável** | 3 sementes por configuração; Double DQN; rede alvo |
| **Problema fácil demais** (guloso já vence) | Apertar a economia, hordas mais pesadas, mais tipos de zumbi |
| **Agente decorar as fases** | Fases por composição, combinações reservadas desde o primeiro dia |
| **Tentação de jogar o jogo real** | Fica como extensão, só depois do trabalho principal pronto |

---

# Glossário

| Termo | Significado neste trabalho |
|---|---|
| **Tick** | A menor unidade de tempo do simulador |
| **Passo** | Um ciclo de decisão do agente: ele age, o simulador roda k ticks |
| **Estado** | O que o agente vê: o tensor de canais sobre o gramado |
| **Ação** | Plantar uma planta numa célula, ou esperar |
| **Máscara** | Filtro que remove ações inválidas antes da escolha |
| **Recompensa** | O número que o simulador devolve a cada passo |
| **Retorno** | A soma das recompensas de uma partida |
| **Q-valor** | A estimativa da rede de quanto retorno uma ação vai render |
| **Memória de replay** | Onde ficam guardadas as experiências passadas para treinar a rede |
| **Segmento** | Um trecho de horda (início, meio ou fim) de uma certa classe |
| **Fase** | Uma partida completa: sequência de segmentos |
| **Currículo** | Treinar do fácil para o difícil |
| **Análise contrafactual** | Voltar a um momento da partida, trocar a jogada e ver o que teria acontecido |

---

# Próximos passos

1. **Fechar os números** da Parte 2 (custos, vidas, danos, taxas de sol, k) —
   ainda são valores iniciais.
2. **Definir as classes de segmento** e a separação treino/teste. Antes de
   treinar qualquer coisa.
3. **Codificar o simulador** com `reset`/`step` e **medir passos por segundo**.
4. **Escrever os baselines** (aleatório, guloso, heurística humana) e calibrar a
   dificuldade com eles.
5. **Treinar o Double DQN** com currículo, das fases fáceis às difíceis.

---

# Esqueleto da apresentação

| # | Slide | Conteúdo |
|---|---|---|
| 1 | Capa | Tema e nomes |
| 2 | O jogo | Gramado 5×9, sol, plantas, zumbis. Imagem vale mais que texto |
| 3 | O dilema | Economia agora ou defesa agora? A consequência vem muito depois |
| 4 | Como o agente aprende | A analogia do jogador iniciante; estado → ação → recompensa |
| 5 | **O fluxo** | O diagrama da seção 1.5. Provavelmente o slide mais importante |
| 6 | O que ele aprende sozinho | As regras da seção 1.6 |
| 7 | Como vamos provar | Aprende / supera regras fixas / generaliza / erra pouco |
| 8 | Próximos passos | O que está decidido e o que falta |

Se for um slide só: o **#5**, com o dilema do **#3** por cima.
