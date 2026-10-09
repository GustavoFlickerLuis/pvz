# STATUS do treino (rl/)

Atualizado em 08/10/2026 22:18:59.

Nenhuma execução rodando agora.

## Vitórias até agora

| execução | status | passos | vitórias nas avaliações periódicas (63 partidas) | vitórias final (420) |
|---|---|---|---|---|
| baseline-s0 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 3%, 200k: 0%, 250k: 0%, 300k: 2%, 350k: 2%, 400k: 2% | 1.4% |
| vencedora-s0 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 0%, 200k: 0%, 250k: 0%, 300k: 0%, 350k: 0%, 400k: 0% | 0.0% |
| muon-s0 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 0%, 200k: 0%, 250k: 2%, 300k: 3%, 350k: 2%, 400k: 2% | 0.2% |
| baseline-s1 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 0%, 200k: 0%, 250k: 2%, 300k: 6%, 350k: 22%, 400k: 17% | 15.5% |
| vencedora-s1 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 0%, 200k: 3%, 250k: 2%, 300k: 0%, 350k: 0%, 400k: 0% | 0.0% |
| muon-s1 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 3%, 200k: 0%, 250k: 2%, 300k: 0%, 350k: 2%, 400k: 0% | 0.0% |
| baseline-s2 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 0%, 200k: 0%, 250k: 3%, 300k: 3%, 350k: 6%, 400k: 14% | 10.7% |
| vencedora-s2 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 0%, 200k: 0%, 250k: 0%, 300k: 0%, 350k: 0%, 400k: 0% | 0.0% |
| muon-s2 | terminado | 400.000/400.000 | 0k: 0%, 50k: 0%, 100k: 0%, 150k: 2%, 200k: 0%, 250k: 0%, 300k: 0%, 350k: 0%, 400k: 0% | 0.0% |

## Notas

- Testes 1–4 feitos antes da fila: unitários ok (12 novos, 54 no total), sanidade ok
  (calmo, dificuldade 0,5: 94% em 200 partidas; a exploração aleatória faz 22%), retomada
  exata ok nas 3 receitas, velocidade com treino: baseline ~300, vencedora ~320, Muon ~190 passos/s.
- Orçamento: 400.000 passos de jogo por execução (limitado pelo Muon, o mais lento, a ~35 min).
- Parar: crie o arquivo `rl/PARAR` (para no fim do episódio, com checkpoint).
  Retomar: `python -m rl.rodar` (apaga o PARAR antigo e retoma).
