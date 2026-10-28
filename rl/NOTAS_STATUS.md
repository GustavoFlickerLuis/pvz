## Notas

- Testes 1–4 feitos antes da fila: unitários ok (12 novos, 54 no total), sanidade ok
  (calmo, dificuldade 0,5: 94% em 200 partidas; a exploração aleatória faz 22%), retomada
  exata ok nas 3 receitas, velocidade com treino: baseline ~300, vencedora ~320, Muon ~190 passos/s.
- Orçamento: 400.000 passos de jogo por execução (limitado pelo Muon, o mais lento, a ~35 min).
- Parar: crie o arquivo `rl/PARAR` (para no fim do episódio, com checkpoint).
  Retomar: `python -m rl.rodar` (apaga o PARAR antigo e retoma).
