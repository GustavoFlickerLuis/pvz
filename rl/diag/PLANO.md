Pergunta: por que o DQN não aprende a escolher jogadas?
Hipótese: a rede aprende o V (valor do tabuleiro) mas não o A (qual jogada é melhor),
porque a diferença entre jogadas é ~1% da variação do retorno. Causas possíveis: (A) faltam
dados das jogadas alternativas; (B) o sinal existe, mas o alvo de treino é ruidoso demais;
(C) a informação não está na observação.
Definições: "ordena" = nota ≥ 60% nos pares claros; "não ordena" = nota < 55%.
Tabela de decisão:
1. td com alvo a cada 2.000 não propaga em 100k e com 500 propaga → adotar 500 daqui em
   diante (pré-requisito).
2. A rede treinada com a resposta (teto) ordena e alguma arquitetura treinada com td
   ordena → causa B, resolvida pela arquitetura: um treino de RL só com ela, no k
   escolhido, 3 sementes.
3. O teto ordena e nenhuma rede td ordena → causa B no alvo: implementar advantage
   learning, medir de novo na prova e só então RL.
4. O teto não ordena → causa C: reformular as ações (tipo + faixa, coluna por regra);
   decisão a discutir com o professor.
5. Os pares que envolvem a ação do Econ5 ordenam e os pares só entre alternativas não →
   causa A: dados e exploração.
Previsões: (a) com k = 20, a curva Econ(N) mantém o pico no meio; (b) com k = 10 e alvo a
cada 2.000, o viés das primeiras 30 decisões só cai abaixo de 0,05 depois de 50k
atualizações; com alvo a cada 500, ou com k = 20, antes; (c) a regressão mc fica perto de
50% na prova; (d) o teto passa de 60%; (e) as redes de RL do lote 1 e do lote 2a ficam
perto de 50%.
