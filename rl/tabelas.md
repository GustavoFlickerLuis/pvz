# Double DQN × receitas do proxy: tabelas

Vitórias da política gulosa (ε = 0, com máscara). Final: FASES_TREINO × sementes 300–319 = 420 partidas. Periódicas: × 300–302 = 63 partidas, a cada 50.000 passos.

## Por execução

| execução | status | passos | episódios | vitórias final | dano final | curva periódica |
|---|---|---|---|---|---|---|
| baseline-s0 | terminado | 400000 | 3649 | 1.4% | 0.094 | 0k 0%, 50k 0%, 100k 0%, 150k 3%, 200k 0%, 250k 0%, 300k 2%, 350k 2%, 400k 2% |
| vencedora-s0 | terminado | 400000 | 4256 | 0.0% | 0.028 | 0k 0%, 50k 0%, 100k 0%, 150k 0%, 200k 0%, 250k 0%, 300k 0%, 350k 0%, 400k 0% |
| muon-s0 | terminado | 400000 | 3950 | 0.2% | 0.053 | 0k 0%, 50k 0%, 100k 0%, 150k 0%, 200k 0%, 250k 2%, 300k 3%, 350k 2%, 400k 2% |
| baseline-s1 | terminado | 400000 | 3360 | 15.5% | 0.312 | 0k 0%, 50k 0%, 100k 0%, 150k 0%, 200k 0%, 250k 2%, 300k 6%, 350k 22%, 400k 17% |
| vencedora-s1 | terminado | 400000 | 3895 | 0.0% | 0.031 | 0k 0%, 50k 0%, 100k 0%, 150k 0%, 200k 3%, 250k 2%, 300k 0%, 350k 0%, 400k 0% |
| muon-s1 | terminado | 400000 | 3892 | 0.0% | 0.025 | 0k 0%, 50k 0%, 100k 0%, 150k 3%, 200k 0%, 250k 2%, 300k 0%, 350k 2%, 400k 0% |
| baseline-s2 | terminado | 400000 | 3474 | 10.7% | 0.274 | 0k 0%, 50k 0%, 100k 0%, 150k 0%, 200k 0%, 250k 3%, 300k 3%, 350k 6%, 400k 14% |
| vencedora-s2 | terminado | 400000 | 4311 | 0.0% | 0.020 | 0k 0%, 50k 0%, 100k 0%, 150k 0%, 200k 0%, 250k 0%, 300k 0%, 350k 0%, 400k 0% |
| muon-s2 | terminado | 400000 | 3983 | 0.0% | 0.047 | 0k 0%, 50k 0%, 100k 0%, 150k 2%, 200k 0%, 250k 0%, 300k 0%, 350k 0%, 400k 0% |

## Por receita (média ± desvio entre sementes)

| agente | n | vitórias final (420 partidas cada) |
|---|---|---|
| baseline: adam 1e-4 constante | 3 | 9.2% ± 7.1  (1.4, 15.5, 10.7) |
| vencedora: sgd 3e-2 cosseno | 3 | 0.0% ± 0.0  (0.0, 0.0, 0.0) |
| muon 1e-3 (AdamW 3e-4) warmup_cosseno | 3 | 0.1% ± 0.1  (0.2, 0.0, 0.0) |
| Econ5 (referência) | — | 67.1% |
| Guloso (referência) | — | 41.2% |
