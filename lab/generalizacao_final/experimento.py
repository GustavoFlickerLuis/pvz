"""Experimento reservado CNN1D v2a; somente novos artefatos nesta pasta.

Comandos: preparar, validar, monitor, worker, status, parar, retomar, analisar.
O worker só abre fases reservadas com protocolo íntegro e pré-teste aprovado.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (1, 2, 3, 4, 6)
TOTAL = 9450


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with open(temp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class IntegrityError(RuntimeError):
    pass


@contextlib.contextmanager
def lock(path):
    import msvcrt
    with open(path, 'a+b') as f:
        if f.tell() == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def busy(path):
    try:
        with lock(path):
            return False
    except OSError:
        return True


def event(run, kind, **kw):
    with open(run / 'eventos.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(utc=now(), evento=kind, pid=os.getpid(), **kw), ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def dbopen(path):
    c = sqlite3.connect(path, timeout=30)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=FULL')
    c.execute('CREATE TABLE IF NOT EXISTS partidas (agente TEXT, fase TEXT, semente INTEGER, grupo TEXT, treino INTEGER, checkpoint TEXT, hash TEXT, protocolo TEXT, dificuldade REAL, k INTEGER, vitoria INTEGER, dano REAL, decisoes INTEGER, acoes TEXT, segundos REAL, utc TEXT, PRIMARY KEY(agente,fase,semente))')
    c.execute('CREATE TABLE IF NOT EXISTS erros (utc TEXT, agente TEXT, fase TEXT, sementes TEXT, erro TEXT)')
    c.commit()
    return c


def count(run, name='resultados.sqlite'):
    p = run / name
    if not p.exists():
        return 0
    with sqlite3.connect(f'file:{p.as_posix()}?mode=ro', uri=True) as c:
        return c.execute('SELECT COUNT(*) FROM partidas').fetchone()[0]


def environment():
    import torch
    import numpy as np
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise IntegrityError('CUDA indisponível: não mudar backend silenciosamente')
    return dict(python=sys.version, executavel=sys.executable, torch=torch.__version__,
                numpy=np.__version__, device='cuda', gpu=torch.cuda.get_device_name(),
                threads=1, deterministic=True, tf32=False, cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def frozen(run, approved=False):
    p = run / 'protocolo.json'
    if sha(p) != (run / 'protocolo.sha256').read_text().strip():
        raise IntegrityError('Hash do protocolo divergiu')
    cfg = js(p)
    for rel, expected in cfg['hashes'].items():
        path = ROOT / rel
        if not path.exists() or sha(path) != expected:
            raise IntegrityError('Arquivo congelado alterado: ' + rel)
    env = environment()
    if env != cfg['ambiente']:
        raise IntegrityError('Ambiente de inferência divergiu do protocolo')
    if approved:
        v = js(run / 'validacao.json')
        if not v.get('aprovado') or v['protocolo'] != sha(p):
            raise IntegrityError('Pré-teste não aprovado para este protocolo')
    return cfg


def load_pol(agent):
    import torch
    from rl.ppo import criar_politica
    pol = criar_politica('cnn1d').to('cuda')
    pol.load_state_dict(torch.load(ROOT / agent['checkpoint'], map_location='cuda', weights_only=True), strict=True)
    pol.eval()
    pol.requires_grad_(False)
    if any(p.requires_grad for p in pol.parameters()):
        raise IntegrityError('Parâmetro com gradiente')
    return pol


def state_sha(pol):
    h = hashlib.sha256()
    for key, value in pol.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_seeds():
    """Analisa campos estruturados; números de métricas/passos não são sementes."""
    import re
    roots = [ROOT, ROOT.parent / 'PythonProject1', ROOT.parent / 'pvz-ppo', ROOT.parent / 'pvz-diag']
    hits, counts, ranges = [], dict(csv=0, json=0), []
    for root in roots:
        for base in [root / 'rl/logs', root / 'lab', root / 'desafio3']:
            if not base.exists():
                continue
            for path in base.rglob('*'):
                if not path.is_file() or HERE in path.parents or path.suffix.lower() not in ('.csv', '.json'):
                    continue
                if path.suffix == '.csv':
                    counts['csv'] += 1
                    with open(path, encoding='utf-8-sig', newline='') as f:
                        reader = csv.DictReader(f)
                        fields = [k for k in (reader.fieldnames or []) if k and ('sement' in k.lower() or k.lower() in ('seed', 'seeds'))]
                        if not fields:
                            continue
                        for index, row in enumerate(reader, 2):
                            for field in fields:
                                value = row.get(field, '') or ''
                                if re.fullmatch(r'\d+', value):
                                    overlap = 10100 <= int(value) <= 10149
                                elif re.fullmatch(r'\d+[-–]\d+', value):
                                    a, b = map(int, re.split('[-–]', value))
                                    overlap = a <= 10149 and b >= 10100
                                else:
                                    overlap = False
                                if overlap:
                                    hits.append(dict(path=str(path), linha=index, campo=field, valor=value))
                else:
                    counts['json'] += 1
                    try:
                        value = js(path)
                    except (ValueError, UnicodeError):
                        continue
                    def walk(obj, trail=''):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                name = trail + '.' + k
                                if 'sement' in k.lower() or k.lower() in ('seed', 'seeds'):
                                    nums = v if isinstance(v, list) else [v]
                                    ints = [x for x in nums if isinstance(x, int)]
                                    overlap = any(10100 <= x <= 10149 for x in ints)
                                    if len(ints) == 2 and ints[0] <= 10149 and ints[1] > 10100:
                                        overlap = True
                                    if overlap:
                                        hits.append(dict(path=str(path), campo=name, valor=v))
                                    ranges.append(dict(path=str(path), campo=name, valor=v))
                                walk(v, name)
                        elif isinstance(obj, list):
                            for item in obj:
                                walk(item, trail)
                    walk(value)
    return dict(utc=now(), faixa=[10100,10149], contagens=counts, conflitos=hits,
                campos_json=ranges, limitacao='Inspeção de artefatos locais estruturados; execuções manuais não registradas não são demonstráveis.',
                decisao='Usar 10100–10149 em A e B: fora da coleta 1000–9999; mesma realização por fase/modelo/baseline. A numeração não modifica FASES_TREINO/TESTE.')


def prepare(run):
    from pvz.divisao import FASES_TREINO, FASES_TESTE
    if run.exists():
        raise IntegrityError('Pasta de execução já existe; preparar nunca sobrescreve')
    assert len(FASES_TREINO) == 21 and len(FASES_TESTE) == 6
    assert not set(FASES_TREINO) & set(FASES_TESTE)
    audit = audit_seeds()
    if audit['conflitos']:
        raise IntegrityError('Conflito de sementes: ' + repr(audit['conflitos'][:10]))
    run.mkdir()
    atomic(run / 'auditoria_sementes.json', audit)
    historical = {}
    for p in ROOT.rglob('*'):
        if p.is_file() and HERE not in p.parents and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            st = p.stat()
            historical[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
    atomic(run / 'historicos_antes.json', historical)
    agents, hashes = [], {}
    for seed in SEEDS:
        folder = f'rl/logs/lab-f1-v2a-s{seed}'
        c = js(ROOT / folder / 'config.json')['config']
        for key, expected in dict(semente=seed, arquitetura='cnn1d', receita='atual', coef_ent=.03, k_ticks=20, dificuldade=1., fases='treino', gamma=.994).items():
            if c[key] != expected:
                raise IntegrityError('Config inesperada ' + folder + ' ' + key)
        rel = folder + '/modelo.pt'
        agent = dict(id=f'cnn1d-v2a-s{seed}', treino=seed, checkpoint=rel, hash=sha(ROOT / rel), config=c)
        load_pol(agent)
        agents.append(agent)
        for name in ['modelo.pt', 'config.json', 'resultado.json']:
            hashes[folder + '/' + name] = sha(ROOT / folder / name)
    agents += [dict(id=x, treino=None, checkpoint='', hash='') for x in ['econ5','guloso']]
    for folder in ['pvz','rl','desafio3','lab']:
        for p in (ROOT / folder).rglob('*.py'):
            if HERE not in p.parents:
                hashes[str(p.relative_to(ROOT)).replace('\\','/')] = sha(p)
    for p in HERE.glob('*.py'):
        hashes[str(p.relative_to(ROOT)).replace('\\','/')] = sha(p)
    for rel in ['LAB.md','CLAUDE.md','AGENTS.md','docs/CONCEITO.md','lab/avaliacao_extra.csv']:
        hashes[rel] = sha(ROOT / rel)
    cfg = dict(versao=1, criado_utc=now(), objetivo='Generalização combinatória PPO CNN1D v2a sem treinamento',
               agentes=agents, fases={'conhecida':list(FASES_TREINO),'inedita':list(FASES_TESTE)}, sementes=list(range(10100,10150)),
               politica='argmax dos logits válidos; observação v2; 226 ações', k=20, dificuldade=1., batch=50,
               total=TOTAL, ambiente=environment(), hashes=hashes,
               ordem='Todos os agentes nas conhecidas, depois todos nas inéditas; sem parada por desempenho.',
               estatistica=dict(agregacao='Média por fase (50 partidas), depois por conjunto; fases com peso igual dentro de cada conjunto. Modelos com peso igual.',
                  treino='5 réplicas; média, DP amostral, IC t95 com 4 gl e fator 2.776445105; descritivo, não teste de equivalência.',
                  partidas='Wilson 95% por modelo/conjunto, condicional às fases fixas; heterogeneidade e dependência limitam interpretação.',
                  fases='DP entre médias por fase separado do DP entre treinos; apenas 6 fases inéditas, sem extrapolação a novas distribuições.',
                  bootstrap='10000 reamostragens hierárquicas: modelos, fases dentro de cada grupo e sementes dentro de fase. Modelos/baselines pareados nas partidas; IC percentil95 exploratório. RNG 20261027.',
                  sucesso='Sem limiar pós-hoc, sem alegação de equivalência/perfeição. Expor diferenças e incertezas.'),
               limites='Combinações já usadas por baselines/proxy; reserva relativa ao desenvolvimento PPO. Estresse excluído. Não ajustar após abertura.',
               retomada='SQLite FULL, PK agente/fase/semente, lock OS, até 3 falhas consecutivas sem progresso; falha de integridade bloqueia imediatamente.')
    atomic(run / 'protocolo.json', cfg)
    (run / 'protocolo.sha256').write_text(sha(run / 'protocolo.json') + '\n', encoding='ascii')
    event(run, 'protocolo_congelado', hash=sha(run / 'protocolo.json'))
    print(json.dumps(dict(run=str(run), modelos=5, faixa=audit['faixa'], conflitos=0), ensure_ascii=False))


def evaluate(run, cfg, agent, games, db, stop=None, fault_after=0):
    """Instrumenta só reset/step do avaliador existente; não altera arquivos antigos."""
    import numpy as np
    import torch
    import rl.avaliar as av
    from pvz.ambiente import Ambiente
    from pvz.agentes import Econ, Guloso
    from rl.ppo import Gulosa
    pol = load_pol(agent) if agent['checkpoint'] else None
    before = state_sha(pol) if pol else None
    protocol_hash = sha(run / 'protocolo.json')
    terminal_count = 0
    class Tracked(Ambiente):
        def reset(self, fase, semente=0, dificuldade=1.0):
            self.meta = (tuple(fase), int(semente))
            self.counts = [0]*6
            self.reward_sum = 0.
            self.started = time.perf_counter()
            return super().reset(fase,semente,dificuldade)
        def step(self, action):
            nonlocal terminal_count
            if not self.mascara()[action]:
                raise IntegrityError('Ação inválida')
            self.counts[0 if action == 225 else 1 + action//45] += 1
            obs, r, done = super().step(action)
            self.reward_sum += r
            if done:
                fase, seed = self.meta
                group = next(g for g, phases in cfg['fases'].items() if list(fase) in phases)
                win = int(r > 0)
                row = (agent['id'], '|'.join(fase), seed, group, agent['treino'], agent['checkpoint'], agent['hash'], protocol_hash,
                       1.,20,win,self.reward_sum-(1 if win else -1),sum(self.counts),json.dumps(self.counts),time.perf_counter()-self.started,now())
                db.execute('INSERT INTO partidas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', row)
                db.commit()
                terminal_count += 1
                if fault_after and terminal_count >= fault_after:
                    os._exit(77)  # somente ensaio de interrupção, bases isoladas
            if stop and stop.exists():
                raise InterruptedError('Parada manual; partidas incompletas continuam pendentes')
            return obs,r,done
    av.Ambiente = Tracked
    try:
        with torch.inference_mode():
            if pol:
                result = av.avaliar_rede(Gulosa(pol), games, 'cuda', dificuldade=1., k=20, formato='v2')
            else:
                result = av.avaliar_agente(Econ(5) if agent['id']=='econ5' else Guloso(), games, dificuldade=1., k=20)
        if pol and state_sha(pol) != before:
            raise IntegrityError('Pesos em memória mudaram')
        return result
    finally:
        av.Ambiente = Ambiente


def validate(run):
    import numpy as np
    import torch
    from pvz.ambiente import Ambiente
    cfg = frozen(run)
    if (run / 'abertura_teste.json').exists():
        raise IntegrityError('Não repetir validação após abertura do teste')
    checks = []
    vdb = dbopen(run / 'preteste.sqlite')
    if vdb.execute('SELECT COUNT(*) FROM partidas').fetchone()[0]:
        raise IntegrityError('Pré-teste já iniciado; usar nova pasta de preparação')
    for agent in cfg['agentes'][:5]:
        pol = load_pol(agent)
        env = Ambiente(k=20,formato='v2')
        x = env.reset(tuple(cfg['fases']['conhecida'][0]),300,1.)
        assert x.shape == (25,5,9)
        with torch.inference_mode():
            logits,value=pol(torch.from_numpy(x[None]).to('cuda'))
        assert logits.shape == (1,226) and value.shape == (1,)
        assert env.mascara()[int(logits.argmax(1).item())]
        assert np.all(logits.detach().cpu().numpy()[0][~env.mascara()] == -1e9)
        games=[(tuple(f),s) for f in cfg['fases']['conhecida'] for s in range(300,320)]
        t=time.perf_counter()
        result=evaluate(run,cfg,agent,games,vdb)
        expected=js(ROOT / Path(agent['checkpoint']).parent / 'resultado.json')
        check=dict(agente=agent['id'], partidas=result['partidas'], vitorias=result['vitorias'],historico=expected['vitorias_final'],dano=result['dano'],dano_historico=expected['dano_final'],segundos=time.perf_counter()-t)
        checks.append(check)
        event(run,'preteste_historico',**check)
        if abs(result['vitorias']-expected['vitorias_final'])>1e-12 or abs(result['dano']-expected['dano_final'])>1e-9:
            raise IntegrityError('Reprodução histórica falhou: '+repr(check))
    vdb.close()
    # Testa morte abrupta, retomada e lock na mesma rotina, somente conhecidas.
    with lock(run / 'worker.lock'):
        p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'probe-lock','--run',str(run)],capture_output=True,text=True)
        if p.returncode!=0:
            raise IntegrityError('Lock concorrente falhou: '+p.stderr)
    cmd=[sys.executable,'-B',str(Path(__file__)),'probe','--run',str(run)]
    p=subprocess.run([sys.executable,'-B',str(Path(__file__)),'probe-monitor','--run',str(run)],capture_output=True,text=True)
    if p.returncode!=0 or count(run,'retomada.sqlite')!=10:
        raise IntegrityError('Monitor não retomou após morte simulada: '+p.stderr)
    trial=js(run/'probe_monitor_estado.json')
    if trial.get('retomadas')!=1:
        raise IntegrityError('Monitor de ensaio não registrou exatamente uma retomada')
    # Executar novamente deve ser no-op; conferir igualdade com execução direta.
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode!=0 or count(run,'retomada.sqlite')!=10:
        raise IntegrityError('Idempotência falhou')
    ref=dbopen(run/'retomada_referencia.sqlite')
    games=[(tuple(cfg['fases']['conhecida'][0]),s) for s in range(350,360)]
    evaluate(run,cfg,cfg['agentes'][0],games,ref)
    columns='agente,fase,semente,vitoria,dano,decisoes,acoes'
    with sqlite3.connect(run/'retomada.sqlite') as resumed:
        a=resumed.execute(f'SELECT {columns} FROM partidas ORDER BY agente,fase,semente').fetchall()
    b=ref.execute(f'SELECT {columns} FROM partidas ORDER BY agente,fase,semente').fetchall()
    if a!=b:
        raise IntegrityError('Resultados retomados diferem da referência')
    ref.close()
    frozen(run)
    # Mesmo limite de falhas usado pelo monitor: progresso zera, três sem progresso bloqueiam.
    state=dict(falhas=0,concluidas=0)
    for expected in [1,2,3]:
        advance_state(state,0,0)
        assert state['falhas']==expected
    advance_state(state,0,1)
    assert state['falhas']==0
    # Parada manual impede a abertura de novas partidas no ensaio.
    (run/'PROBE_PARAR').touch()
    p=subprocess.run(cmd,capture_output=True,text=True)
    assert p.returncode==0 and count(run,'retomada.sqlite')==10
    (run/'PROBE_PARAR').unlink()
    atomic(run/'validacao.json',dict(aprovado=True,utc=now(),protocolo=sha(run/'protocolo.json'),historicos=checks,interrupcao='monitor retomou automaticamente após morte real em 3 commits; só 7 pendentes, igualdade exata com referência, idempotência, parada e exclusão mútua aprovadas', gradientes=False,otimizador_criado=False))
    event(run,'preteste_aprovado',partidas_historicas=2100)
    print('PRÉ-TESTE APROVADO; nenhuma fase reservada executada.')


def probe(run, fault_after):
    cfg=frozen(run)
    with lock(run/'worker.lock'):
        if (run/'PROBE_PARAR').exists():
            return
        db=dbopen(run/'retomada.sqlite')
        done={s for (s,) in db.execute('SELECT semente FROM partidas')}
        games=[(tuple(cfg['fases']['conhecida'][0]),s) for s in range(350,360) if s not in done]
        if games:
            evaluate(run,cfg,cfg['agentes'][0],games,db,fault_after=fault_after)
        db.close()


def verify_rows(db, cfg, protocol):
    agents={a['id']:a for a in cfg['agentes']}
    for row in db.execute('SELECT agente,fase,semente,grupo,treino,checkpoint,hash,protocolo,dificuldade,k,vitoria,dano,decisoes,acoes FROM partidas'):
        agent,phase,seed,group,train,ck,h,ph,d,k,w,damage,dec,actions=row
        a=agents.get(agent)
        if a is None or group not in cfg['fases'] or phase.split('|') not in cfg['fases'][group] or seed not in cfg['sementes']:
            raise IntegrityError('Partida fora do protocolo')
        if (train,ck,h,ph,d,k)!=(a['treino'],a['checkpoint'],a['hash'],protocol,1.,20):
            raise IntegrityError('Metadados divergentes')
        counts=json.loads(actions)
        if w not in (0,1) or dec<=0 or not math.isfinite(damage) or len(counts)!=6 or sum(counts)!=dec or min(counts)<0:
            raise IntegrityError('Resultado inválido')
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
        raise IntegrityError('SQLite integrity_check falhou')


def worker(run):
    cfg=frozen(run,approved=True)
    with lock(run/'worker.lock'):
        db=dbopen(run/'resultados.sqlite')
        verify_rows(db,cfg,sha(run/'protocolo.json'))
        event(run,'worker_inicio',concluidas=count(run))
        done={(a,f,s) for a,f,s in db.execute('SELECT agente,fase,semente FROM partidas')}
        for group,phases in cfg['fases'].items():
            if group=='inedita':
                if (run/'PARAR').exists():
                    return
                known=db.execute("SELECT COUNT(*) FROM partidas WHERE grupo='conhecida'").fetchone()[0]
                if known!=7350:
                    raise IntegrityError('Abrir teste exige todas as 7350 partidas conhecidas completas')
                frozen(run,approved=True)
                marker=run/'abertura_teste.json'
                if not marker.exists():
                    atomic(marker,dict(utc=now(),protocolo=sha(run/'protocolo.json'),conhecidas=known))
                    event(run,'abertura_teste_reservado',conhecidas=known)
            for agent in cfg['agentes']:
                for phase in phases:
                    if (run/'PARAR').exists():
                        event(run,'parada_manual',concluidas=count(run))
                        return
                    phase_id='|'.join(phase)
                    games=[(tuple(phase),s) for s in cfg['sementes'] if (agent['id'],phase_id,s) not in done]
                    if not games:
                        continue
                    try:
                        evaluate(run,cfg,agent,games,db,stop=run/'PARAR')
                    except InterruptedError:
                        event(run,'parada_manual',concluidas=count(run))
                        return
                    except Exception:
                        db.execute('INSERT INTO erros VALUES (?,?,?,?,?)',(now(),agent['id'],phase_id,json.dumps([s for _,s in games]),traceback.format_exc()))
                        db.commit()
                        raise
                    n=count(run)
                    event(run,'progresso',agente=agent['id'],fase=phase,concluidas=n,total=TOTAL)
                    atomic(run/'progresso.json',dict(utc=now(),concluidas=n,total=TOTAL,agente=agent['id'],grupo=group,fase=phase))
        verify_rows(db,cfg,sha(run/'protocolo.json'))
        if count(run)!=TOTAL:
            raise IntegrityError('Total inesperado')
        frozen(run,approved=True)
        before=js(run/'historicos_antes.json')
        changed=[]
        for rel,expected in before.items():
            p=ROOT/rel
            if not p.exists() or [p.stat().st_size,p.stat().st_mtime_ns]!=expected:
                changed.append(rel)
        atomic(run/'integridade_final.json',dict(utc=now(),partidas=TOTAL,sqlite='ok',historicos_alterados=changed,hashes_congelados='ok'))
        if changed:
            raise IntegrityError('Artefatos históricos alterados: '+repr(changed))
        atomic(run/'concluido.json',dict(utc=now(),partidas=TOTAL,protocolo=sha(run/'protocolo.json')))
        db.close()
        event(run,'avaliacao_concluida',partidas=TOTAL)


def status(run):
    info=dict(run=str(run),concluidas=count(run),total=TOTAL,monitor_ativo=busy(run/'monitor.lock'),worker_ativo=busy(run/'worker.lock'),parar=(run/'PARAR').exists(),bloqueado=(run/'BLOQUEADO.json').exists(),concluido=(run/'concluido.json').exists(),relatorio=(run/'relatorio.md').exists())
    if (run/'progresso.json').exists():
        info['ultimo']=js(run/'progresso.json')
    print(json.dumps(info,ensure_ascii=False,indent=2))


def advance_state(state,before,after):
    state['falhas']=0 if after>max(before,state['concluidas']) else state['falhas']+1
    state['concluidas']=after


def monitor(run,probe_mode=False):
    prefix='probe_' if probe_mode else ''
    dbname='retomada.sqlite' if probe_mode else 'resultados.sqlite'
    current=lambda:count(run,dbname)
    finished=lambda:current()==10 if probe_mode else (run/'concluido.json').exists()
    with lock(run/(prefix+'monitor.lock')):
        event(run,'monitor_inicio')
        while busy(run/'worker.lock'):
            if (run/'PARAR').exists():
                return
            time.sleep(2)
        state_path=run/(prefix+'monitor_estado.json')
        state=js(state_path) if state_path.exists() else dict(falhas=0,concluidas=current(),retomadas=0)
        while not finished():
            if (run/'PARAR').exists() or (run/'BLOQUEADO.json').exists():
                event(run,'monitor_parado')
                return
            frozen(run,approved=not probe_mode)
            before=current()
            command=[sys.executable,'-B',str(Path(__file__)),'probe' if probe_mode else 'worker','--run',str(run)]
            if probe_mode and before==0:
                command+=['--fault-after','3']
            with open(run/(prefix+'worker.log'),'a',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                atomic(run/'worker_pid.json',dict(pid=proc.pid,utc=now()))
                code=proc.wait()
            after=current()
            event(run,'worker_saida',codigo=code,antes=before,depois=after)
            if (run/'PARAR').exists():
                return
            if code==42 or (run/'BLOQUEADO.json').exists():
                event(run,'monitor_bloqueado_integridade')
                return
            if finished():
                break
            advance_state(state,before,after)
            state['retomadas']=state.get('retomadas',0)+1
            atomic(state_path,state)
            if state['falhas']>=3:
                atomic(run/'BLOQUEADO.json',dict(utc=now(),motivo='Três falhas consecutivas sem progresso',estado=state))
                event(run,'monitor_bloqueado_tres_falhas')
                return
            event(run,'retomada_automatica',concluidas=after,falhas=state['falhas'])
            time.sleep(5)
        if not probe_mode and not (run/'relatorio.md').exists():
            with open(run/'analise.log','a',encoding='utf-8') as log:
                code=subprocess.call([sys.executable,'-B',str(HERE/'analise.py'),'--run',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            event(run,'analise_saida',codigo=code)


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('comando',choices=['preparar','validar','monitor','worker','status','parar','retomar','probe','probe-lock','probe-monitor'])
    p.add_argument('--run',type=Path,default=HERE/'execucao_20261027')
    p.add_argument('--fault-after',type=int,default=0)
    a=p.parse_args()
    run=a.run.resolve()
    if run.parent!=HERE:
        raise IntegrityError('Saída deve ser subpasta exclusiva de generalizacao_final')
    try:
        if a.comando=='preparar': prepare(run)
        elif a.comando=='validar': validate(run)
        elif a.comando=='monitor': monitor(run)
        elif a.comando=='worker': worker(run)
        elif a.comando=='status': status(run)
        elif a.comando=='parar': (run/'PARAR').touch(); event(run,'parada_solicitada')
        elif a.comando=='retomar':
            frozen(run,approved=True)
            if (run/'BLOQUEADO.json').exists(): raise IntegrityError('Bloqueio exige inspeção; retomada automática proibida')
            (run/'PARAR').unlink(missing_ok=True)
            monitor(run)
        elif a.comando=='probe': probe(run,a.fault_after)
        elif a.comando=='probe-monitor': monitor(run,probe_mode=True)
        elif a.comando=='probe-lock':
            if not busy(run/'worker.lock'): raise IntegrityError('Não detectou lock do pai')
    except IntegrityError:
        if run.exists():
            atomic(run/'BLOQUEADO.json',dict(utc=now(),motivo=traceback.format_exc()))
            event(run,'erro_integridade',erro=traceback.format_exc())
        traceback.print_exc()
        sys.exit(42)
    except Exception:
        if run.exists(): event(run,'erro_tecnico',erro=traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__=='__main__':
    main()
