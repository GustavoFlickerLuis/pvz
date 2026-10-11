#!/bin/sh
# Progresso de todas as sementes do PPO.
cd "$(dirname "$0")"
for d in rl/logs/ppo-fases-s*/; do
  id=$(basename "$d")
  printf '%-22s ' "$id"
  if [ -f "$d/resultado.json" ]; then
    /opt/homebrew/bin/python3 -c "import json;r=json.load(open('$d/resultado.json'));print(f\"TERMINOU {100*r['vitorias_final']:5.1f}%  {r['passos']} passos  {r['minutos']:.0f} min\")"
  elif [ -f "rl/logs/$id.log" ]; then
    tail -n 60 "rl/logs/$id.log" | /usr/bin/grep -oE "passo [0-9]+ .*passos/s" | tail -1 | cut -c1-80
  else
    echo "aguardando"
  fi
done
