# Resumo para apresentação

A CNN1D PPO v2a já havia obtido 96,2% ± 1,1 ponto nas combinações conhecidas, em cinco treinamentos, e 95,8% ± 0,5 em sementes novas de três desses modelos. Isso ainda não demonstrava recombinação de fases.

Congelamos os cinco pesos finais e avaliamos 21 combinações conhecidas e seis reservadas, 50 sementes novas por combinação, sem alterar pesos, com k=20 e dificuldade 1,0. Econ5 e Guloso jogaram as mesmas partidas. Foram 9.450 partidas.

Nas inéditas, CNN1D: 96.20% ± 1.28 entre treinamentos; nas conhecidas: 95.54% ± 1.58. Diferença inéditas − conhecidas: +0.66 pontos. Econ5 nas inéditas: 60.33%; Guloso: 49.33%.

O resultado descreve generalização combinatória neste simulador e protocolo. Não demonstra perfeição nem generalização para novas regras ou classes. Há apenas cinco treinamentos e seis fases reservadas; a média deve ser acompanhada dos resultados por combinação, incertezas e baselines. Qualquer ajuste futuro será posterior à abertura do teste.
