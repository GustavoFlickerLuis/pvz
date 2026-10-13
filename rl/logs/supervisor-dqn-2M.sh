#!/bin/sh
# Supervisor do DQN comparável (sementes 0–2, 2M): roda rodar-dqn-2M.sh e o retoma do último
# checkpoint quando o processo cai sem erro de código (morto, falta de memória). Para quando:
#   - terminou (as 3 sementes com resultado.json);
#   - erro de código: Traceback que não seja de memória, perda não finita, rl/PARAR;
#   - o mesmo trecho (mesmo último checkpoint) cair 3 vezes seguidas.
# Cada queda e retomada vai para rl/logs/fila-dqn-2M.log.
cd "$(dirname "$0")/../.."
FILA=rl/logs/fila-dqn-2M.log
nota() { echo "=== $1 $(date '+%F %T') $2" >> $FILA; }
ultimo_ckpt() { grep -o "\[ckpt\] passo [0-9]*" "rl/logs/$1.log" 2>/dev/null | tail -1 | grep -o "[0-9]*$"; }
atual() {
  for s in 0 1 2; do
    [ -f rl/logs/dqn-cnn1d-k20-2M-s$s/resultado.json ] || { echo dqn-cnn1d-k20-2M-s$s; return; }
  done
}

trecho_ant=""; seguidas=0
while true; do
  id=$(atual)
  [ -z "$id" ] && { nota supervisor "fim: as 3 sementes terminaram"; exit 0; }
  sh rl/logs/rodar-dqn-2M.sh
  c=$?
  [ $c -eq 0 ] && continue
  id=$(atual)   # o rodar-dqn-2M.sh pode ter passado para a semente seguinte
  novo=$(tail -n 60 "rl/logs/$id.log" 2>/dev/null)
  if [ $c -eq 3 ]; then
    nota "$id" "supervisor: parado por rl/PARAR; esperando o Gustavo"; exit 3
  fi
  if echo "$novo" | grep -q "não finita"; then
    nota "$id" "supervisor: ERRO (perda não finita); esperando o Gustavo"; exit 1
  fi
  if echo "$novo" | grep -q "Traceback" && \
     ! echo "$novo" | grep -qiE "MemoryError|out of memory|Unable to allocate|DefaultCPUAllocator"; then
    nota "$id" "supervisor: ERRO de código (Traceback); esperando o Gustavo"; exit 1
  fi
  ck=$(ultimo_ckpt "$id")
  trecho="$id@${ck:-0}"
  if [ "$trecho" = "$trecho_ant" ]; then seguidas=$((seguidas + 1)); else seguidas=1; fi
  trecho_ant=$trecho
  nota "$id" "supervisor: queda (código $c, sem erro de código), $seguidas ª no trecho a partir do ckpt ${ck:-0}"
  if [ $seguidas -ge 3 ]; then
    nota "$id" "supervisor: 3 quedas seguidas no mesmo trecho; esperando o Gustavo"; exit 2
  fi
  sleep 60
  nota "$id" "supervisor: retomando do ckpt ${ck:-0}"
done
