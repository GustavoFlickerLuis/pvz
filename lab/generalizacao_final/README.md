# Generalização final CNN1D PPO v2a

Este diretório contém apenas o novo experimento; não utiliza filas nem monitores de treinamento. Não há treinamento, otimizador, alteração de regras ou escrita nos checkpoints. Todas as execuções usam `-B` para não criar caches nos fontes históricos.

Python: `C:\Users\gului\Desktop\PythonProject1\.venv\Scripts\python.exe`.
Diretório do experimento: `C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final`.
Saídas principais: `execucao_20261027`.

## Consultar o progresso (PowerShell)

```powershell
& 'C:\Users\gului\Desktop\PythonProject1\.venv\Scripts\python.exe' -B 'C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final\experimento.py' status
Get-Content -LiteralPath 'C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final\execucao_20261027\eventos.jsonl' -Tail 12
```

O SQLite é a fonte transacional, um commit FULL por partida, chave primária `(agente,fase,semente)`. CSVs são exportados ao concluir. A contagem de status consulta diretamente o SQLite; progresso.json é atualizado ao terminar cada lote/fase. Não somar os tempos por partida, pois a inferência é compartilhada.

## Parar

```powershell
& 'C:\Users\gului\Desktop\PythonProject1\.venv\Scripts\python.exe' -B 'C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final\experimento.py' parar
```

Cria exclusivamente `execucao_20261027\PARAR`. O worker interrompe no próximo passo observado; partidas incompletas permanecem pendentes. Os registros já confirmados são preservados. Não usar os arquivos PARAR dos treinos antigos.

## Retomar em processo independente e oculto

```powershell
Start-Process -FilePath 'C:\Users\gului\Desktop\PythonProject1\.venv\Scripts\python.exe' -ArgumentList @('-B', 'C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final\experimento.py', 'retomar') -WorkingDirectory 'C:\Users\gului\Desktop\pvz-lab' -WindowStyle Hidden
```

O comando verifica o protocolo, remove somente a parada deste experimento e inicia o monitor. Locks mantidos pelo sistema operacional impedem dois monitores ou workers simultâneos. Quando o worker termina inesperadamente, o monitor retoma apenas os registros pendentes. Se houver três falhas consecutivas sem progresso, cria BLOQUEADO.json e encerra; uma falha de integridade bloqueia imediatamente. O bloqueio não é removido automaticamente: exige inspeção. Se o próprio monitor for encerrado, o worker pode continuar; reiniciar o monitor aguarda o lock do worker antes de tentar uma retomada. Desligamento do computador requer iniciar novamente o comando acima; não foi instalado serviço de inicialização do Windows.

## Reprodução sem sobrescrever a rodada original

Não preparar novamente a mesma pasta. `preparar` recusa diretório existente. Para uma reprodução declarada posterior à abertura, escolher uma nova subpasta, por exemplo `reproducao_posterior_01`, e passar o mesmo `--run` absoluto em todos os comandos:

```powershell
$py = 'C:\Users\gului\Desktop\PythonProject1\.venv\Scripts\python.exe'
$script = 'C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final\experimento.py'
$destino = 'C:\Users\gului\Desktop\pvz-lab\lab\generalizacao_final\reproducao_posterior_01'
& $py -B $script preparar --run $destino
& $py -B $script validar --run $destino
& $py -B $script monitor --run $destino
```

A pasta é explicitamente denominada posterior; não pode ser apresentada como teste ainda intocado. A auditoria dos registros históricos exclui este diretório experimental, portanto não detecta automaticamente o uso na rodada original: conferir também abertura_teste.json e resultados.sqlite desta rodada antes de qualquer nova execução. Uma reprodução posterior deve ser identificada e autorizada como tal. Para verificar/recriar os resumos da rodada concluída, `analise.py --run <pasta>` lê a base e regenera apenas derivados dentro da mesma pasta; preservar/copiar os derivados se for necessária uma comparação byte a byte. Não executar reprodução automaticamente.

## Rastreabilidade

- protocolo.json + protocolo.sha256: escolhas, versões, hashes, modelos, fases e estatística fixados antes da avaliação principal.
- auditoria_sementes.json: inspeção de campos de sementes nos artefatos estruturados locais; não prova inexistência de execução manual não registrada.
- validacao.json e preteste.sqlite: pré-teste somente nas conhecidas; reprodução histórica de cinco modelos.
- retomada.sqlite, retomada_referencia.sqlite, probe_worker.log: teste de morte real após três partidas confirmadas, retomada automática, igualdade com referência e idempotência.
- abertura_teste.json: momento de abertura, depois do pré-teste e das 7.350 partidas conhecidas dos sete agentes.
- resultados.sqlite, partidas.csv: resultados individuais; erros na tabela erros e erros_tecnicos.json.
- eventos.jsonl, worker.log, monitor_estado.json: progresso e interrupções.
- integridade_final.json: conferência de hashes, banco e metadados dos arquivos históricos.
- resumo_*.csv, diferencas_baselines.csv, comparacoes_pareadas.csv, incerteza_bootstrap.csv: análises.
- relatorio.md, resumo_apresentacao.md, figuras/*.png e *.pdf: entregáveis científicos.

As contagens de ações têm ordem `[esperar, girassol, ervilha, repetidora, batata, mina]`. O avaliador PPO só usa reset/step/observar/mascara. Econ5/Guloso mantêm a exceção histórica de regras manuais, sem ler spawns futuros. Não são gravados estados por tick. Para futuro visualizador: agente, checkpoint/hash, segmentos, semente, k e dificuldade permitem reconstrução determinística no mesmo ambiente.
