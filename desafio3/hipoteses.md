# Desafio 3 — hipóteses registradas antes das varreduras

Escritas depois da baseline de rascunho (Adam, η = 1e-4, constante, 20.000 passos,
semente 0) e **antes** de qualquer varredura de η ou de agenda. O commit deste
arquivo marca o momento. Cada hipótese diz o que a confirmaria e o que a refutaria.
Elas não serão editadas depois dos resultados; o desfecho vai numa seção separada,
no fim.

## O que a baseline de rascunho mostrou (base das previsões)

- Validação: 0,497 → mínimo 0,164 (passo 14.250) → 0,167 no fim. Chega a 1% do
  mínimo no passo ~10.500; depois só cresce a distância treino–validação
  (Huber no lote fixo do treino: 0,150 → 0,130).
- ‖g‖ entre 0,3 e 5,5 e nunca cortado (clip 10), sem picos no início.
- ‖θ‖ cresce sem parar, de 8 para 15 (wd = 0).
- ρ mediano (passos ≥ 2000): conv1 2,9e-4, conv2 4,5e-4, conv3 3,7e-4,
  cabeca_q 2,1e-4, cabeca_esperar 9,6e-5. Tudo abaixo da referência de 1e-3 do enunciado.
- 8–11% dos filtros da conv3 nunca ativam no lote fixo; nenhum na conv1 nem na conv2.

## Hipóteses

### H1. O melhor η do Adam fica acima de 1e-4
Os ρ da baseline (≈ 3–5e-4 nas convs) estão abaixo da referência de ~1e-3: o passo
é pequeno. **Confirma:** na varredura com warmup + cosseno, o menor val_final do Adam
fica em η > 1e-4. **Refuta:** o melhor η é ≤ 1e-4.

### H2. AdamW (weight decay) estabiliza ‖θ‖ e reduz a distância treino–validação
Na baseline, ‖θ‖ cresce linearmente e a distância treino–validação abre depois do
passo ~9000. **Confirma:** no AdamW, com o melhor η dele, ‖θ‖ se estabiliza (ou cresce
bem menos) e a diferença huber_val − huber_treino no fim é menor que a do Adam no
mesmo η. **Refuta:** ‖θ‖ continua crescendo no mesmo ritmo, ou a distância não diminui.

### H3. O warmup ajuda pouco no Adam
Não há picos iniciais em ‖g‖ na baseline, que é o sintoma que o warmup corrige.
**Confirma:** no Adam, com o mesmo η, warmup_cosseno e cosseno diferem menos que a
variação entre sementes. **Refuta:** o warmup melhora o val_final além dessa variação,
ou evita uma divergência que acontece sem ele.

### H4. Boa parte da perda de validação (~0,16) é ruído irredutível; as diferenças entre otimizadores vão ser pequenas
O alvo é o retorno de Monte Carlo de uma única partida. Estados iguais (ou quase) com
desfechos diferentes recebem G diferente, que pode diferir em ~2 entre vitória e
derrota. No subconjunto de 1000 amostras da verificação de sobreajuste, só os estados
idênticos já formam um piso de 1,6% da perda inicial. No dataset inteiro, estados
parecidos com G diferente devem pesar muito mais. **Confirma:** os val_final dos
melhores (otimizador, η) de cada família ficam a poucos % uns dos outros, da mesma
ordem da variação entre sementes. **Refuta:** algum otimizador fica claramente abaixo
dos outros, por bem mais que a variação entre sementes.

### H5. O ρ menor da cabeca_esperar não vem de "esperar" ser rara
**Fato medido:** "esperar" é **81,8%** das ações do treino (79,6% nas partidas
vencidas, 84,8% nas perdidas) e 81,7% na validação. A cabeca_esperar recebe gradiente
em ~4 de cada 5 amostras. Mesmo assim, o ρ dela é 2–5× menor que o das outras camadas
(9,6e-5 contra 2,1e-4 a 4,5e-4).
**Hipótese:** a causa é o tamanho e o sinal, não a frequência. São 65 parâmetros que
recebem a média global de 64 canais, e o alvo de esperar é ruidoso: o mesmo estado
pode virar vitória ou derrota. No Adam, gradientes de sinal instável dão m/√v pequeno,
então o passo efetivo fica bem abaixo de η.
**Confirma:** a razão ρ_cabeca_esperar / ρ_conv se mantém parecida entre Adam, AdamW e
RMSprop, e muda nos SGDs (que não normalizam por √v).
**Refuta:** a razão é a mesma em todos os otimizadores.

## Desfecho
(preencher depois das etapas 2–4, sem editar as hipóteses acima)
