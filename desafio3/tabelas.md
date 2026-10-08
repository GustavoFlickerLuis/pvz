# Desafio 3 — tabelas finais

Gerado por `python -m desafio3.analise tabelas` a partir do `experimentos.csv` e dos
logs. Média ± desvio amostral (n − 1) nas 5 sementes, exceto onde indicado. Todas as
execuções: 10.000 passos, lote 256, clip 10.

## a) Treino × validação

huber_treino: Huber no lote fixo do treino no fim. Δ: diferença da média contra a baseline
(negativo = melhor).

| config | n | huber_treino | val_final | val_esperar | val_plantio | Δtreino | Δval | val_min | passo do val_min |
|---|---|---|---|---|---|---|---|---|---|
| baseline: adam 1e-4 constante | 5 | 0.14632 ± 0.00357 | 0.16755 ± 0.00083 | 0.16228 ± 0.00101 | 0.19110 ± 0.00134 | +0.00000 | +0.00000 | 0.16706 ± 0.00060 | 9750, 10000, 8250, 9250, 9250 |
| sgd 3e-2 cosseno | 5 | 0.12926 ± 0.00122 | 0.16225 ± 0.00219 | 0.15733 ± 0.00248 | 0.18428 ± 0.00207 | -0.01706 | -0.00529 | 0.16123 ± 0.00221 | 5000, 6500, 8500, 5000, 7750 |
| sgd 3e-2 warmup_cosseno | 5 | 0.13124 ± 0.00273 | 0.16227 ± 0.00189 | 0.15735 ± 0.00189 | 0.18422 ± 0.00273 | -0.01508 | -0.00528 | 0.16169 ± 0.00205 | 6750, 6500, 8500, 5000, 8750 |
| rmsprop 3e-4 warmup_cosseno | 5 | 0.13139 ± 0.00343 | 0.16234 ± 0.00135 | 0.15704 ± 0.00149 | 0.18603 ± 0.00101 | -0.01493 | -0.00521 | 0.16145 ± 0.00125 | 5750, 6500, 6750, 8000, 7250 |
| adam 3e-4 warmup_cosseno (extra) | 5 | 0.13096 ± 0.00204 | 0.16614 ± 0.00164 | 0.16154 ± 0.00206 | 0.18672 ± 0.00122 | -0.01536 | -0.00141 | 0.16547 ± 0.00169 | 8250, 6500, 8500, 5250, 7000 |
| adamw 3e-4 warmup_cosseno (extra) | 5 | 0.13105 ± 0.00201 | 0.16555 ± 0.00147 | 0.16089 ± 0.00186 | 0.18639 ± 0.00124 | -0.01527 | -0.00200 | 0.16475 ± 0.00141 | 8250, 6500, 8500, 5250, 7000 |
| muon 1e-3 warmup_cosseno | 5 | 0.11118 ± 0.00117 | 0.17497 ± 0.00029 | 0.17141 ± 0.00040 | 0.19090 ± 0.00174 | -0.03514 | +0.00743 | 0.17018 ± 0.00067 | 4000, 3500, 3250, 2750, 3250 |
| muon 1e-1 warmup_cosseno (1 semente) | 1 | 0.06276 | 0.19599 | 0.19323 | 0.20836 | -0.08356 | +0.02845 | 0.18274 | 250 |

## b) Velocidade (métrica secundária)

Primeiro passo em que a validação (medida a cada 250 passos) fica ≤ 0.1676
(val_final médio da baseline), por semente, ou "nunca". O tempo é passos × ms/passo.
Nas agendas que decaem, isso é medido dentro de uma execução de 10.000 passos, com η
ainda alto no meio dela, e por isso subestima a velocidade dessas agendas.

| config | passos até o alvo (por semente) | ms/passo | segundos até o alvo (por semente) |
|---|---|---|---|
| baseline: adam 1e-4 constante | 8250, 10000, nunca, 8000, 8250 | 4.85 | 35.3, 42.1, —, 40.5, 51.6 |
| sgd 3e-2 cosseno | 2750, 3500, 3500, 3250, 3250 | 3.79 | 10.3, 13.3, 13.3, 12.4, 12.3 |
| sgd 3e-2 warmup_cosseno | 3500, 3500, 3500, 3250, 4000 | 3.72 | 12.9, 13.0, 13.1, 12.1, 14.9 |
| rmsprop 3e-4 warmup_cosseno | 2750, 3000, 3250, 3000, 3500 | 4.00 | 11.0, 12.1, 13.1, 11.9, 14.0 |
| adam 3e-4 warmup_cosseno (extra) | 4000, 5000, 4750, 5250, 5250 | 4.00 | 15.8, 19.8, 18.8, 21.5, 21.3 |
| adamw 3e-4 warmup_cosseno (extra) | 4000, 3500, 4500, 5000, 4250 | 4.13 | 16.5, 14.3, 18.7, 20.4, 17.7 |
| muon 1e-3 warmup_cosseno | nunca, nunca, nunca, nunca, nunca | 7.44 | —, —, —, —, — |
| muon 1e-1 warmup_cosseno (1 semente) | nunca | 8.52 | — |

## c) Extras × finalistas (critério de vitória)

A vence B se |média_A − média_B| > 2·s_c, com s_c = √((s_A² + s_B²)/2), em val_final.
Os extras não mudam a vencedora: o protocolo terminou no passo 4.

| extra | finalista | Δ média (extra − finalista) | Δ / s_c | veredito |
|---|---|---|---|---|
| adam 3e-4 warmup_cosseno (extra) | sgd 3e-2 cosseno | +0.00389 | +2.01 | finalista vence |
| adam 3e-4 warmup_cosseno (extra) | sgd 3e-2 warmup_cosseno | +0.00388 | +2.19 | finalista vence |
| adam 3e-4 warmup_cosseno (extra) | rmsprop 3e-4 warmup_cosseno | +0.00380 | +2.53 | finalista vence |
| adamw 3e-4 warmup_cosseno (extra) | sgd 3e-2 cosseno | +0.00330 | +1.77 | empate |
| adamw 3e-4 warmup_cosseno (extra) | sgd 3e-2 warmup_cosseno | +0.00329 | +1.94 | empate |
| adamw 3e-4 warmup_cosseno (extra) | rmsprop 3e-4 warmup_cosseno | +0.00321 | +2.28 | finalista vence |
