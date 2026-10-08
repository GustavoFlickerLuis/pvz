# A rede do proxy joga? (extra exploratório, fora do Desafio 3)

21 FASES_TREINO × sementes 200–219 = 420 partidas por agente (as mesmas para todos).
Dano: fração da vida total da fase causada como dano (média por partida).

| agente | vitórias | dano médio | decisões/partida | esperar | girassol | ervilha | repetidora | batata | mina |
|---|---|---|---|---|---|---|---|---|---|
| RedeGulosa (vencedora sgd 3e-2 cosseno) | 0.5% | 0.020 | 85 | 82.9% | 8.9% | 1.4% | 0.1% | 1.9% | 4.9% |
| RedeGulosa (baseline adam 1e-4 constante) | 0.0% | 0.014 | 79 | 82.9% | 9.3% | 1.3% | 0.0% | 1.6% | 4.9% |
| Econ5 (sem aleatoriedade) | 64.5% | 0.742 | 251 | 88.1% | 2.5% | 9.4% | 0.0% | 0.0% | 0.0% |
| Guloso | 40.7% | 0.427 | 173 | 80.8% | 6.9% | 11.5% | 0.8% | 0.0% | 0.0% |
