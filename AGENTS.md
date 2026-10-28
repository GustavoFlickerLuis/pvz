# Contexto para continuidade — laboratório PvZ
Converse em português. Inspeção em 19/10/2026. Nesta sessão, modificar somente AGENTS.md; não executar treino, sobrescrever resultados ou promover alterações ao projeto oficial.

## Escopo e fontes
Esta pasta é uma cópia local de ../pvz-ppo feita em 16/10, sem .git próprio. Não usar Git aqui: ele encontra por ascendência um repositório em C:/Users/gului que não representa o laboratório. Não alterar PythonProject1, pvz-ppo ou pvz-diag para incorporar o lab sem decisão do Gustavo.
Leia primeiro [LAB.md](LAB.md), o lab.md identificado nesta transferência. Ele contém protocolo, previsões e resultados até 17/10. [PASSAGEM.md](PASSAGEM.md) é histórico de 09/10 e está anterior ao laboratório. CLAUDE.md e docs/CONCEITO.md preservam especificação e contexto, mas não descrevem todo o estado executado.
Objetivo: reduzir variância do PPO e comparar atenção/transformers com cnn1d. Não duplicar o protocolo: usar as seções nomeadas do LAB.md e configs/logs como evidências.

## Estrutura e implementação
- pvz/: ambiente e divisão fixa; desafio3/modelo.py: registry ARQUITETURAS, cnn1d e braços atencao_antes, atencao_depois, transformer, transformer_d128, cnn1d_dilatada.
- rl/ppo.py: Politica/PoliticaArq, logits A e crítico V; entrada v2 (25,5,9), 226 ações, máscara nos canais 20–24, inválidas com logit -1e9.
- lab/fila.py, monitor.py, processos.py: fila sequencial, recuperação de processos e RSS. lab/fila.json: IDs e configs; lab/receita_f2.txt: escolha.
- lab/avaliacao_extra.py: estresse e final versus melhor em sementes novas; lab/parte0.py e parte1.py: análises; lab/relatorio.py: gera figuras e reescreve seção do LAB.md.
- Evidências brutas: rl/logs/<id>/{config.json,registro.csv,avaliacoes.csv,resultado.json,modelo.pt,melhor.pt,ckpt.pt}; lab/avaliacao_extra.csv, medidas.json, memoria.csv, fila_saida.log, monitor.log.
- tests/test_lab.py e demais tests/: formas/índices, campo de visão, reprodução e retomada; lab/teste_parada.txt, teste_ckpt.txt, teste_monitor.txt: testes históricos de integração.

## Receita, orçamento e diferenças do oficial
PPO: k=20, gamma=0,994, GAE lambda=0,95; 16 ambientes ×128 decisões, 4 épocas, minilote 512, Adam 2,5e-4, clip 0,2, coef V 0,5, norma 0,5.
A receita atual original usava Adam epsilon=1e-5; o mecanismo de receitas do oficial passou a usar 1e-8 pelo proxy. O lab restaura EPS_ADAM_ATUAL=1e-5 nas receitas atual*. Não atribuir diferenças somente à arquitetura, nem juntar resultados CPU e CUDA como réplicas iguais.
Fase 1: atual entropia 0,01; v2a 0,03; v2b 0,03 + receita atual_linear. A agenda do código é max(0,1-t/T) ao longo do orçamento inteiro de atualizações do otimizador, não só uma queda nos últimos passos. As últimas 16 atualizações excedentes ficam com eta=0.
Orçamento nominal 2M decisões; os resultado.json inspecionados têm 2.000.896 porque a coleta termina em blocos de 2.048. Config.atualizacoes usa divisão inteira no cálculo de T. Estender um treino com decaimento exige definir o que acontece com a agenda; não alterar T silenciosamente.
Checkpoint automático a cada 250k, gravação com troca atômica, ambientes em voo e RNGs CPU/CUDA salvos; restauração dos estados de RNG para CPU corrigida. Isso existe no lab, não presumir equivalência com o oficial.

## Executado e conclusões sustentadas
Foram encontrados resultado.json para as 6 sanidades, 15 execuções da Fase 1, 15 da Fase 2 e 6 da Parte 1 (IDs lab-f3), além de testes de infraestrutura. Os finais inspecionados concordam com as tabelas do LAB.md.
- Fase 1, sementes 1/3/4/2/6: atual 86,9 ±4,4%, v2a 96,2 ±1,1%, v2b 94,6 ±1,5%. Ambas aceitas; a regra registrada escolheu v2a. Os rótulos sementes ruins/boas vieram de CPU; em CUDA a referência já teve outro comportamento.
- Fase 2, sementes 1/2/3, referência cnn1d v2a 96,6 ±1,2%: atenção antes 92,9%, depois 92,1%, transformer 89,4%, d128 86,3% perderam; dilatada 94,4% empatou pelo critério. O texto inicial arredonda atenção antes para 93,0; os resultados e a tabela gerada mostram 92,9.
- Parte 1: transformer v2b 94,7 ±4,5% empatou no final com transformer v2a e cnn1d, mas perdeu para cnn1d no estresse. Transformer com coef_ent=0,01 fez 79,8 ±2,1%, pior. A previsão de reduzir a oscilação com v2b não se confirmou; a análise da Parte 0 classificou grande parte como ruído.
- Sanidade ≥98% em todos os braços. Teto: diagnóstico supervisionado, uma semente, não prova de superioridade em RL.
Usar a regra previamente registrada |delta média| > 2*s_c, com desvios amostrais. Empate significa ausência de separação por essa regra; não equivalência comprovada.
Nenhuma conclusão estabelece superioridade geral de PPO sobre DQN ou de uma família arquitetural fora desta receita/orçamento. Há 3 sementes nas arquiteturas; a mesma receita pode favorecer braços diferentes.
A queda no final e os efeitos da agenda são observações; explicações de mecanismo precisam de experimentos próprios.

## Avaliação e proteção dos experimentos
Nunca fases de teste no lab. Treino usa fases de treino e sementes 1000–9999; periódica 300–302 (63 partidas), final 300–319 (420). Sementes novas 320–339 e estresse 1,25/1,5 ainda usam fases de treino: não são generalização para combinações inéditas.
O melhor.pt é escolhido em 300–302, subconjunto da avaliação 300–319; usar 320–339 para comparação independente de seleção. Baselines antigos de estresse econ5/guloso usaram k=10 e ficam apenas como histórico; usar IDs econ5-k20/guloso-k20, conforme lab/avaliacao_extra.py.
Não editar previsões/regras depois de rodar. Não sobrescrever pastas ou IDs. Uma execução por vez na GPU; sem DQN e sem treino oficial simultâneo. RL lê só reset/step/observar/mascara, sem estado interno ou futuros spawns.
Não registrar credenciais.

## Comandos para futura retomada
A partir desta pasta, usando Python com dependências de requirements.txt:
- Consulta: python -m lab.fila --status.
- Treino: python -m lab.monitor (inicia e pode religar a fila; só executar quando solicitado).
- Parar: criar lab/PARAR; retomar: remover esse arquivo e iniciar monitor. Nunca confundir com rl/PARAR oficial.
- python -m lab.avaliacao_extra --tabela consulta tabelas; sem --tabela executa avaliações e grava CSV.
- python -m lab.parte0 e python -m lab.parte1 produzem análises; verificar efeitos antes de executar.
- python -m lab.relatorio reescreve LAB.md e figuras; não é consulta sem efeitos.
- python -m pytest tests para validação futura. Testes de parada/monitor/checkpoint iniciam subprocessos e escrevem artefatos; não executar durante experimentos.

## Validação e pendências da transferência
Nesta sessão somente leitura de código/configs/resultados e documentação; nenhum teste foi reexecutado. LAB.md registra 149 testes passando historicamente; os .txt registram retomada com pesos/CSVs idênticos e exercícios de morte/parada/monitor. Não confundir esses registros com validação atual.
O LAB.md diz que nada rodava ao encerramento de 17/10; isso é estado histórico, não inspeção de processos em 19/10.
Pendente: Gustavo decidir promoção ao oficial, reconciliar especificação e instruções antigas, fechar receita antes da generalização final. Teste oficial RL deve usar sementes a partir de 10100 (10000–10019 já usadas no Desafio 3). O caminho do repo Desafios-InteligenciaComputacional ainda falta.
