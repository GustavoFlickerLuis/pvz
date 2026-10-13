#!/bin/sh
# DQN comparável (cnn1d, k = 20, γ = 0,994, 2M), sementes 0–2 em sequência.
# Retoma sozinho do checkpoint de cada semente; as que já terminaram são puladas.
cd "$(dirname "$0")/../.."
for s in 0 1 2; do
  id=dqn-cnn1d-k20-2M-s$s
  [ -f rl/logs/$id/resultado.json ] && continue
  echo "=== $id início $(date '+%F %T')" >> rl/logs/fila-dqn-2M.log
  python -u -m rl.dqn --arquitetura cnn1d --receita baseline --curriculo nao --exploracao uniforme \
    --semente $s --k-ticks 20 --gamma 0.994 --passos 2000000 --id $id >> rl/logs/$id.log 2>&1
  c=$?
  echo "=== $id fim $(date '+%F %T') código $c" >> rl/logs/fila-dqn-2M.log
  [ $c -ne 0 ] && exit $c
done
