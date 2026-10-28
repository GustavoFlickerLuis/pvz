# Continuidade do laboratório PvZ — 27/10/2026

Converse em português. Este arquivo atualiza o contexto para tarefas em `pvz-lab/lab/`. Antes de retomar o laboratório, ler também o AGENTS.md da raiz e os documentos citados, distinguindo suas datas. O AGENTS.md da raiz preserva o estado de 19/10: não editá-lo, pois foi incluído nos hashes congelados da avaliação final. LAB.md preserva os resultados até 17/10. Este arquivo é posterior ao experimento e não faz parte dos inputs congelados.

## Manutenção permanente da continuidade

Ao terminar uma tarefa relevante ou encerrar a sessão, atualizar este arquivo com trabalho realizado, decisões, resultados confirmados, fontes, validações e pendências, sem precisar de novo pedido do usuário. Em tarefas longas, registrar marcos concluídos. Respeitar restrições explícitas de somente leitura. Não modificar arquivos congelados, logs/resultados históricos, previsões ou protocolos para atualizar contexto; não registrar credenciais. A instrução depende da execução do agente: não é um hook do PyCharm nem proteção garantida contra encerramento abrupto. Se a sessão começar na raiz do laboratório, ler explicitamente este arquivo, pois um AGENTS.md aninhado não se aplica automaticamente a toda pasta irmã.

## Avaliação final executada e concluída

O usuário autorizou expressamente preparação e execução da generalização, incluindo as fases reservadas, em 27/10. Foram criados somente scripts/artefatos em `lab/generalizacao_final/`, sem alteração das worktrees oficiais, do simulador, geração de hordas, recompensa, arquitetura, receita ou checkpoints. Nunca executar Git nesta cópia sem .git próprio.

- Diretório: `generalizacao_final/execucao_20261027/` (relativo a esta pasta lab).
- Cinco modelos finais originais em `../rl/logs/lab-f1-v2a-s{1,2,3,4,6}/modelo.pt` (relativo a lab); não usar melhor.pt na comparação principal. CNN1D PPO v2a, observação v2 25×5×9, 226 ações, k=20, dificuldade 1,0, argmax mascarado sem gradientes. Checkpoints seguem no laboratório, sem integração ao oficial.
- Mesmas sementes 10100–10149 em cada fase/modelo/baseline; geração determinística dependente dos segmentos. Conjunto conhecido: 21×50=1.050 por agente. Inédito: 6×50=300 por agente. CNN 6.750 partidas + Econ5/Guloso 2.700 = **9.450**. Não houve novos treinamentos.
- Auditoria anterior examinou 1.636 CSVs e 1.190 JSONs locais, sem conflito registrado na faixa. Limitação: não prova inexistência de execuções manuais não registradas; a auditoria não procura automaticamente rodadas novas dentro de generalizacao_final, que exige conferir separadamente.
- Protocolo SHA-256: `d9c844fd434c0ff3616181c00f3d41ef23c72aa5ba1804f04743f71dde7f680a`. Modelos, configs, fontes e documentos relevantes têm hashes em protocolo.json. Não alterá-los silenciosamente: frozen() bloqueia divergências, inclusive no AGENTS.md da raiz.
- Pré-teste reproduziu exatamente vitórias e dano das 420 partidas históricas conhecidas de cada um dos cinco modelos. Verificou carregamento estrito, formas/máscara, pesos sem gradientes e integridade. Ensaio de morte real após três commits, retomada automática apenas das sete pendentes, igualdade exata à referência, idempotência e lock aprovados.
- Protocolo congelado antes do pré-teste/principal. Abertura do teste após 7.350 partidas conhecidas completas: **2026-10-27 15:17:09 UTC = 12:17:09 America/Sao_Paulo**. Conclusão da avaliação 15:19:07 UTC; análise concluída 15:19:15 UTC. Monitor encerrou normalmente, sem partidas pendentes.

## Resultados finais

Médias com peso igual por fase dentro de cada conjunto e por treinamento. Todos os modelos jogaram o mesmo número de partidas. DP abaixo é amostral entre treinamentos.

| Treinamento | Conhecidas | Inéditas |
|---|---:|---:|
| s1 | 96,48% | 96,33% |
| s2 | 93,81% | 95,67% |
| s3 | 95,62% | 97,00% |
| s4 | 94,19% | 94,33% |
| s6 | 97,62% | 97,67% |
| CNN média ± DP | 95,54% ± 1,58 pp | 96,20% ± 1,28 pp |
| Econ5 | 62,95% | 60,33% |
| Guloso | 48,67% | 49,33% |

Diferença inéditas−conhecidas +0,66 pp (DP entre treinos 0,90 pp); IC t95 condicionado às partidas/fases fixas −0,46 a +1,77 pp. Bootstrap hierárquico exploratório (10.000 réplicas, modelos/fases/sementes) −2,65 a +3,89 pp. Média CNN nas inéditas 1.443/1.500; conhecidas 5.016/5.250. Vantagem inéditas versus Econ5 +35,87 pp e Guloso +46,87 pp.

Por combinação inédita, CNN média 94–100%; pior modelo/fase inédita: s4, cedo_apressado/misto/blindados, 86%. Cinco treinamentos e só seis fases inéditas limitam a interpretação. Resultados sustentam boa generalização combinatória neste protocolo, não equivalência demonstrada, perfeição, causalidade nem generalização para classes/regras/dificuldades novas. Estresse não foi métrica principal.

**O teste reservado PPO já foi aberto.** As fases também tinham usos anteriores por baselines/proxy, e 10000–10019 eram usadas no Desafio 3. Não reutilizar 10100–10149 ou estas fases como se continuassem intocadas. Qualquer ajuste posterior deve ser explicitamente identificado como posterior à abertura. Não selecionar só a melhor semente.

## Artefatos e integridade

Em generalizacao_final/execucao_20261027/: relatorio.md (27 fases e interpretação), resumo_apresentacao.md, protocolo.json/protocolo.sha256 (configurações e hashes), auditoria_sementes.json, validacao.json, preteste.sqlite, retomada.sqlite e arquivos WAL/SHM quando existentes, retomada_referencia.sqlite, abertura_teste.json, resultados.sqlite, partidas.csv, resumo_agente_conjunto.csv, resumo_cnn1d.csv, resumo_por_fase.csv, diferencas_baselines.csv, comparacoes_pareadas.csv, incerteza_bootstrap.csv, figuras/ (PNG/PDF), eventos.jsonl, integridade_final.json e concluido.json.

Conferência final independente: SQLite íntegro; exatamente 9.450 chaves agente/fase/semente únicas no banco e CSV; 50 partidas em cada par agente/fase; conhecidas 7.350 registros/21 fases, inéditas 2.100/6; sete agentes; sementes 10100–10149; zero erros técnicos. Hashes dos cinco pesos preservados, arquivos históricos sem alterações detectadas. As atualizações de continuidade nos AGENTS ocorreram depois dessa conferência, fora dos documentos congelados.

## Operação e retomada

Scripts em generalizacao_final/experimento.py e analise.py; instruções completas em generalizacao_final/README.md. Usar Python `C:/Users/gului/Desktop/PythonProject1/.venv/Scripts/python.exe` com `-B` para não escrever caches nos fontes históricos.

Consulta: `python -B lab/generalizacao_final/experimento.py status` a partir da raiz pvz-lab. O experimento já terminou: não há necessidade de reiniciar. Comando parar cria PARAR somente na pasta da rodada; retomar remove essa parada e inicia o monitor, sem repetir partidas confirmadas. SQLite FULL e chave primária, locks do SO, máximo três falhas consecutivas sem progresso; erro de integridade bloqueia e exige inspeção. Processo independente e oculto foi iniciado por Start-Process -WindowStyle Hidden, sem usar filas/monitores de treinamento antigos. Não confundir com lab/PARAR ou rl/PARAR.

## Próximos passos

Discutir relatório científico e apresentação; decidir posteriormente promoção do laboratório ao oficial. Visualizador ainda não implementado: usar agente/checkpoint/hash/segmentos/semente/k/dificuldade nos dados para reconstruir uma partida no mesmo ambiente; contagens de ações em ordem esperar/girassol/ervilha/repetidora/batata/mina. Não foram gravadas trajetórias por tick. Não repetir avaliações nem iniciar treinamento automaticamente. Reconciliar documentação histórica em tarefa própria sem invalidar os hashes congelados; se necessário, preservar snapshots e registrar qualquer evolução explicitamente.


## Publicação independente — 27/10/2026

Esta cópia agora pertence à worktree `pvz-lab-git`, branch `lab-v2a-generalizacao`,
base PPO `2a1c80d`. A restrição de nunca usar Git aplica-se ao laboratório
original `pvz-lab`, não a esta worktree. Consultar README.md e publicacao/
para instruções portáveis, inventários, exclusões e integridade. Cinco modelos
finais publicados; demais pesos/checkpoints continuam somente no laboratório
original. 100 testes passaram, quatro excluídos na seleção; testes que treinam
não foram executados. Protocolo, resultados e fontes congelados preservados.
Não retomar filas ou repetir a avaliação para continuar a publicação.
