# Laboratório PPO v2a e generalização

Branch independente `lab-v2a-generalizacao`, baseada em `ppo` no commit
`2a1c80d015dae813ba45a379070cf3fde652f5c7`. O laboratório original foi copiado,
sem modificar simulador, arquiteturas, receita, pesos ou resultados.

## Começar em outro computador

```powershell
git clone --branch lab-v2a-generalizacao --single-branch https://github.com/GustavoFlickerLuis/pvz.git pvz-lab-git
cd pvz-lab-git
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

O ambiente original usa PyTorch 2.11.0+cu128, CUDA e Windows. O avaliador e os
monitores históricos usam CUDA e locks/processos do Windows; Linux e CPU não são
substitutos já validados. NumPy, pytest e matplotlib não têm versões fixadas no
requirements histórico; as versões do experimento constam no protocolo.

## Ler e conferir a avaliação concluída

`lab/generalizacao_final/execucao_20261027/relatorio.md` é o relatório completo;
`resumo_apresentacao.md`, os CSVs, figuras, protocolo, eventos e SQLite ficam na
mesma pasta. São 9.450 partidas; CNN1D 95,54% ± 1,58 pp nas conhecidas e
96,20% ± 1,28 pp nas inéditas, entre cinco treinamentos. O teste foi aberto em
27/10/2026: nenhuma reprodução futura é um teste intocado.

Os cinco pesos finais estão em `rl/logs/lab-f1-v2a-s{1,2,3,4,6}/modelo.pt`,
359.405 bytes cada. São state_dict carregáveis com `weights_only=True` e
carregamento estrito; hashes idênticos ao protocolo. Não precisam de Git LFS.

```powershell
python -B lab/generalizacao_final/experimento.py status
```

O comando acima consulta o estado. Não iniciar a fila antiga para consultar.
Para uma reprodução FUTURA, explicitamente autorizada e posterior à abertura,
usar uma pasta nova; nunca reutilizar a pasta histórica:

```powershell
$rodada = Join-Path (Get-Location) 'lab/generalizacao_final/reproducao_posterior_01'
python -B lab/generalizacao_final/experimento.py preparar --run $rodada
python -B lab/generalizacao_final/experimento.py validar --run $rodada
python -B lab/generalizacao_final/experimento.py monitor --run $rodada
```

Esses três últimos comandos EXECUTAM partidas; não foram executados na publicação.
A auditoria original também consultou pastas irmãs e não será reproduzida
integralmente por um clone isolado. O protocolo e os resultados originais ficam
preservados; uma reprodução gera seu próprio protocolo e ambiente registrado.

## Desenvolvimento e preservação

Nesta máquina, continuar em `C:/Users/gului/Desktop/pvz-lab-git`, na nova branch.
`pvz-lab` continua sendo a cópia experimental original sem Git. As outras
worktrees e branches não recebem estas alterações.

Ler `lab/AGENTS.md` e esta página antes de desenvolver. `AGENTS.md` da raiz,
`LAB.md`, `CLAUDE.md`, fontes e configurações congeladas são evidências históricas:
não atualizar silenciosamente os arquivos referidos em `protocolo.json`.
Os caminhos absolutos em configs, logs, documentos e protocolo são históricos,
não configurações portáveis para novos jobs. Criar nova config/ID e usar caminhos
da máquina atual, especialmente `arquivo_parar`; não copiar o caminho antigo.

`.gitattributes` desativa conversão de finais de linha para manter todos os bytes
e hashes do laboratório nos clones. Isso também explica diferenças de finais
de linha em relação à base PPO. Não houve mudança lógica no simulador.

## O que ficou local

`publicacao/arquivos_excluidos.json` lista cada exclusão, tamanho, SHA-256 e motivo.
Inclui 142 arquivos .pt mantidos somente no laboratório (ckpt.pt, melhor.pt e demais pesos),
datasets diagnósticos .npz, caches e controles locais. Permanecem em seus caminhos
originais sob `C:/Users/gului/Desktop/pvz-lab`; fazer backup dessa pasta para
preservar retomada exata e pesos dos outros experimentos. O clone permite avaliar
os cinco finais, mas não retomar os treinos antigos nem avaliar outros pesos
ausentes. Os fontes e resultados das outras arquiteturas foram incluídos.

Dois pesos históricos já versionados na base PPO (`ppo-fases-s0` e `ppo-sanidade`)
foram herdados sem alteração; o manifesto os distingue das exclusões efetivas.
Os logs textuais pequenos foram mantidos como evidência. O banco de retomada
preserva seu WAL necessário; arquivos SHM e locks são transitórios e não publicados.
Nenhum arquivo novo supera 50 MiB. Nenhum serviço adicional foi configurado.

## Verificação da publicação

`publicacao/arquivos_copiados.json` registra 1.197 cópias com tamanho e hash;
`.gitignore` e `lab/AGENTS.md` recebem somente manutenção da publicação.
`publicacao/integridade_publicacao.json` registra hashes do protocolo, modelos,
SQLite íntegro, CSV/banco com as mesmas 9.450 chaves e zero erros técnicos.
A varredura de padrões de credenciais não encontrou indícios; nenhum arquivo
de credenciais, ambiente virtual ou configuração pessoal de IDE foi selecionado.

100 testes passaram e quatro foram excluídos da seleção, sem executar treinamentos.
Não foi rodada a suíte completa: os testes de reprodução/retomada que treinam,
o mini-treino PPO, testes DQN e otimização supervisionada não foram executados.
O teste de baselines `TestAgentes` foi excluído por jogar nas fases reservadas.
Os testes unitários de determinismo do ambiente incluem uma fase reservada com
semente 3; são verificação de regras, não nova avaliação de desempenho dos modelos.
Os resultados históricos não foram reavaliados nem reescritos.

```powershell
python -B -m pytest tests/test_ambiente.py tests/test_arquiteturas.py tests/test_diag.py tests/test_lab.py tests/test_ppo.py::test_formatos_e_mascara_nos_logits tests/test_ppo.py::test_gae -k "not TestAgentes and not receita_atual_reproduz and not parada_e_retomada" -p no:cacheprovider -q
```
