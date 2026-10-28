"""Análise descritiva pré-especificada; lê somente o experimento concluído."""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experimento import frozen, verify_rows, sha, ROOT, HERE, TOTAL, atomic, now


def writecsv(path, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(w,n):
    z=1.959963984540054
    p=w/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half


def mean_sd(x):
    return float(np.mean(x)),float(np.std(x,ddof=1))


def main(run):
    cfg=frozen(run,approved=True)
    assert (run/'concluido.json').exists()
    db=sqlite3.connect(f'file:{(run/"resultados.sqlite").as_posix()}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    verify_rows(db,cfg,sha(run/'protocolo.json'))
    rows=[dict(r) for r in db.execute('SELECT * FROM partidas ORDER BY agente,grupo,fase,semente')]
    assert len(rows)==TOTAL
    writecsv(run/'partidas.csv',rows)
    errors=[dict(r) for r in db.execute('SELECT * FROM erros')]
    atomic(run/'erros_tecnicos.json',errors)
    agents=[a['id'] for a in cfg['agentes']]
    groups=['conhecida','inedita']
    phases={g:['|'.join(f) for f in cfg['fases'][g]] for g in groups}
    all_phases=phases['conhecida']+phases['inedita']
    # Tensor pareado: agente × fase × semente; cinco modelos + dois baselines.
    wins=np.empty((7,27,50))
    damage=np.empty_like(wins)
    decisions=np.empty_like(wins)
    for r in rows:
        i,j,k=agents.index(r['agente']),all_phases.index(r['fase']),cfg['sementes'].index(r['semente'])
        wins[i,j,k]=r['vitoria']; damage[i,j,k]=r['dano']; decisions[i,j,k]=r['decisoes']
    summary=[]
    for i,a in enumerate(agents):
        for group,sl in [('conhecida',slice(0,21)),('inedita',slice(21,27))]:
            x=wins[i,sl]
            lo,hi=wilson(int(x.sum()),x.size)
            summary.append(dict(agente=a,grupo=group,fases=x.shape[0],partidas=x.size,vitorias=int(x.sum()),taxa=float(x.mean()),wilson95_inf=lo,wilson95_sup=hi,dano=float(damage[i,sl].mean()),decisoes=float(decisions[i,sl].mean()),dp_entre_fases=float(x.mean(1).std(ddof=1))))
    writecsv(run/'resumo_agente_conjunto.csv',summary)
    phase_rows=[]
    for j,p in enumerate(all_phases):
        for i,a in enumerate(agents):
            x=wins[i,j]
            lo,hi=wilson(int(x.sum()),50)
            phase_rows.append(dict(agente=a,grupo='conhecida' if j<21 else 'inedita',fase_id=f'F{j+1:02}',inicio=p.split('|')[0],meio=p.split('|')[1],fim=p.split('|')[2],partidas=50,vitorias=int(x.sum()),taxa=float(x.mean()),wilson95_inf=lo,wilson95_sup=hi,dano=float(damage[i,j].mean()),decisoes=float(decisions[i,j].mean())))
    writecsv(run/'resumo_por_fase.csv',phase_rows)
    model_means=np.stack([wins[:5,:21].mean((1,2)),wins[:5,21:].mean((1,2))],axis=1)
    gaps=model_means[:,1]-model_means[:,0]
    agg=[]
    for name,x in [('conhecida',model_means[:,0]),('inedita',model_means[:,1]),('inedita_menos_conhecida',gaps)]:
        mean,sd=mean_sd(x)
        half=2.776445105*sd/math.sqrt(5)
        agg.append(dict(metrica=name,media=mean,dp_treinos=sd,ic_t95_inf=mean-half,ic_t95_sup=mean+half,n_treinos=5))
    writecsv(run/'resumo_cnn1d.csv',agg)
    # Bootstrap hierárquico pareado. Sementes reamostradas por fase, comuns a agentes.
    rng=np.random.default_rng(20261027)
    boot=np.empty((10000,7))
    for b in range(10000):
        model_indices=rng.integers(0,5,5)
        means=[]
        for offset,nph in [(0,21),(21,6)]:
            ph=rng.integers(0,nph,nph)+offset
            ss=rng.integers(0,50,(nph,50))
            selected=wins[:,ph[:,None],ss]
            agentmean=selected.mean((1,2))
            means.append(np.r_[agentmean[model_indices].mean(),agentmean[5:]])
        a,bgroup=means
        boot[b]=[a[0],bgroup[0],bgroup[0]-a[0],a[0]-a[1],bgroup[0]-bgroup[1],a[0]-a[2],bgroup[0]-bgroup[2]]
    metrics=['cnn_conhecida','cnn_inedita','inedita_menos_conhecida','cnn_menos_econ5_conhecida','cnn_menos_econ5_inedita','cnn_menos_guloso_conhecida','cnn_menos_guloso_inedita']
    bm=[]
    for i,name in enumerate(metrics):
        lo,hi=np.quantile(boot[:,i],[.025,.975])
        bm.append(dict(metrica=name,ic_boot95_inf=float(lo),ic_boot95_sup=float(hi),reamostragens=10000))
    writecsv(run/'incerteza_bootstrap.csv',bm)
    differences=[]
    for group,sl in [('conhecida',slice(0,21)),('inedita',slice(21,27))]:
        for baseline in [5,6]:
            per_model=wins[:5,sl].mean((1,2))-wins[baseline,sl].mean()
            differences.append(dict(grupo=group,comparacao='cnn1d_menos_'+agents[baseline],media=float(per_model.mean()),dp_treinos=float(per_model.std(ddof=1))))
    writecsv(run/'diferencas_baselines.csv',differences)
    paired=[]
    for r in rows:
        phase=all_phases.index(r['fase']); seed=cfg['sementes'].index(r['semente'])
        if r['agente'] in agents[:5]:
            paired.append(dict(agente=r['agente'],fase=r['fase'],semente=r['semente'],grupo=r['grupo'],vitoria=r['vitoria'],econ5=int(wins[5,phase,seed]),guloso=int(wins[6,phase,seed]),delta_econ5=r['vitoria']-int(wins[5,phase,seed]),delta_guloso=r['vitoria']-int(wins[6,phase,seed])))
    writecsv(run/'comparacoes_pareadas.csv',paired)
    figs=run/'figuras';figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.bbox':'tight'})
    fig,ax=plt.subplots(figsize=(9,5))
    positions=np.arange(3)
    for g,sl,color,label,dx in [('conhecida',slice(0,21),'#2675aa','Conhecidas',-.18),('inedita',slice(21,27),'#d77b24','Inéditas',.18)]:
        values=[wins[:5,sl].mean(),wins[5,sl].mean(),wins[6,sl].mean()]
        ax.bar(positions+dx,np.array(values)*100,width=.34,label=label,color=color)
        ax.scatter(np.full(5,dx)+np.linspace(-.09,.09,5),wins[:5,sl].mean((1,2))*100,c='black',s=18,zorder=4)
    ax.set_xticks(positions,['CNN1D v2a','Econ5','Guloso']);ax.set_ylabel('Vitórias (%)');ax.set_ylim(0,105)
    ax.set_title('Combinações conhecidas e inéditas — pontos: cinco treinamentos')
    ax.legend();ax.grid(axis='y',alpha=.2)
    for ext in ['png','pdf']:fig.savefig(figs/f'comparacao_conjuntos.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,11))
    matrix=np.c_[wins[:5].mean(2).T,wins[:5].mean((0,2)),wins[5:].mean(2).T]*100
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(8),['s1','s2','s3','s4','s6','CNN média','Econ5','Guloso'],rotation=25)
    ax.set_yticks(range(27),[f'F{j+1:02} '+p.replace('|',' / ')+(' *' if j>=21 else '') for j,p in enumerate(all_phases)],fontsize=7)
    ax.axhline(20.5,color='red',lw=2)
    for i in range(27):
        for j in range(8):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=6,color='white' if matrix[i,j]>65 else 'black')
    fig.colorbar(im,ax=ax,label='Vitórias (%)');ax.set_title('Desempenho por combinação — * reservada')
    for ext in ['png','pdf']:fig.savefig(figs/f'por_fase.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.axhline(0,color='gray',lw=1)
    ax.bar(range(5),gaps*100,color='#526c83')
    ax.set_xticks(range(5),['s1','s2','s3','s4','s6']);ax.set_ylabel('Inéditas − conhecidas (pontos percentuais)')
    ax.set_title('Diferença por treinamento independente')
    for ext in ['png','pdf']:fig.savefig(figs/f'diferenca_generalizacao.{ext}')
    plt.close(fig)
    fmt=lambda x:f'{100*x:.2f}'
    md=['# Avaliação final de generalização — CNN1D PPO v2a','',f'Gerado em {now()}. Protocolo SHA-256: `{sha(run/"protocolo.json")}`.','',
    '## Objetivo e protocolo','',
    'Medir novas realizações de 21 combinações conhecidas e generalização para seis combinações reservadas, sem treinamento ou seleção adicional. Cinco pesos finais das sementes 1, 2, 3, 4 e 6; Econ5 e Guloso. Mesmas sementes 10100–10149 para cada combinação e agente, dificuldade 1,0, k=20, política argmax mascarada, observação v2 e inferência CUDA determinística. Total: 9.450 partidas (6.750 CNN, 2.700 baselines). As sementes numéricas são compartilhadas; o gerador incorpora a combinação. A separação de fases permanece intacta.','',
    'Não há critério binário de sucesso ou equivalência. Robustez por aumento de dificuldade não integra este experimento. A reserva é relativa ao desenvolvimento do PPO: baselines e proxy já usaram essas combinações historicamente.','',
    '## Preparação e integridade','',
    'Os cinco modelos foram carregados estritamente. O pré-teste reproduziu os resultados históricos em 420 partidas conhecidas por modelo, verificou formas e máscara, ausência de gradientes, morte abrupta/retomada exata, idempotência e exclusão mútua. O protocolo foi congelado antes das partidas principais; as 7.350 partidas conhecidas de todos os agentes terminaram antes da abertura das reservadas. Datas e hashes estão em protocolo.json, validacao.json, abertura_teste.json e eventos.jsonl.','',
    f'Integridade final: {json.dumps(json.loads((run/"integridade_final.json").read_text()),ensure_ascii=False)}. Erros técnicos registrados na avaliação principal: {len(errors)}. Interrupções e saídas de processo também constam em eventos.jsonl.','',
    '## Modelos','', '| Agente | SHA-256 dos pesos finais |','|---|---|']
    for a in cfg['agentes'][:5]:md.append(f'| {a["id"]} | `{a["hash"]}` |')
    md+=['','## Resultados por modelo e conjunto','','| Agente | Conhecidas | Inéditas | Inéditas − conhecidas (pp) |','|---|---:|---:|---:|']
    for i,a in enumerate(agents):
        x=wins[i,:21].mean();y=wins[i,21:].mean()
        md.append(f'| {a} | {fmt(x)}% | {fmt(y)}% | {100*(y-x):+.2f} |')
    md+=['','## Configuração CNN1D: cinco réplicas','','| Medida | Média | DP entre treinos | IC t 95% |','|---|---:|---:|---:|']
    for a in agg:md.append(f'| {a["metrica"]} | {fmt(a["media"])} | {fmt(a["dp_treinos"])} | [{fmt(a["ic_t95_inf"])}, {fmt(a["ic_t95_sup"])}] |')
    md+=['','Valores em % ou pontos percentuais para diferenças. IC t com cinco treinamentos é aproximado; condiciona às fases e partidas fixadas e não prova equivalência.','',
    '## Comparação com heurísticas','','| Conjunto | Diferença CNN − baseline (pp) | Média | DP entre modelos |','|---|---|---:|---:|']
    for d in differences:md.append(f'| {d["grupo"]} | {d["comparacao"]} | {100*d["media"]:+.2f} | {100*d["dp_treinos"]:.2f} |')
    md+=['','## Incerteza adicional','','| Métrica | IC bootstrap 95% (%, ou pp) |','|---|---:|']
    for b in bm:md.append(f'| {b["metrica"]} | [{fmt(b["ic_boot95_inf"])}, {fmt(b["ic_boot95_sup"])}] |')
    md+=['','Bootstrap hierárquico exploratório com 10.000 reamostragens de modelos, fases e sementes dentro da fase, pareado entre agentes. Trata as fases como população empírica; as seis inéditas são poucas e foram fixadas por desenho, não sorteadas de toda distribuição possível. O IC não autoriza generalização a novos tipos de zumbis, regras ou dificuldades.','',
    'Wilson por modelo/conjunto e por fase está nos CSVs. Esses intervalos condicionam às fases escolhidas e pressupõem partidas Bernoulli independentes; heterogeneidade e sorteios compartilhados limitam a leitura. As 6.750 partidas da CNN não são 6.750 réplicas de treinamento.','',
    '## Resultados por combinação','','| ID | Grupo | Início / meio / fim | CNN média ± DP treinos | Econ5 | Guloso |','|---|---|---|---:|---:|---:|']
    for j,p in enumerate(all_phases):
        x=wins[:5,j].mean(1)
        md.append(f'| F{j+1:02} | {"conhecida" if j<21 else "inédita"} | {p.replace("|"," / ")} | {fmt(x.mean())} ± {fmt(x.std(ddof=1))} | {fmt(wins[5,j].mean())} | {fmt(wins[6,j].mean())} |')
    md+=['','## Dano, decisões e variação entre fases','','| Grupo | Agente | Dano médio | Decisões médias | DP das taxas entre fases (pp) |','|---|---|---:|---:|---:|']
    for group,sl in [('conhecida',slice(0,21)),('inedita',slice(21,27))]:
        for name,indices in [('CNN1D média',slice(0,5)),('Econ5',slice(5,6)),('Guloso',slice(6,7))]:
            phase_mean=wins[indices,sl].mean((0,2))
            md.append(f'| {group} | {name} | {damage[indices,sl].mean():.4f} | {decisions[indices,sl].mean():.2f} | {100*phase_mean.std(ddof=1):.2f} |')
    hard=np.argsort(wins[:5,21:].mean((0,2)))[:3]+21
    md+=['','Combinações inéditas com menor vitória média: '+ '; '.join(f'F{j+1:02} ({fmt(wins[:5,j].mean())}%)' for j in hard)+'.',
    '', '## Interpretação e limitações','',
    f'A configuração obtém {fmt(model_means[:,1].mean())}% nas inéditas contra {fmt(model_means[:,0].mean())}% nas conhecidas: diferença inéditas − conhecidas de {100*gaps.mean():+.2f} pp. Os cinco resultados devem ser lidos em conjunto; não foi escolhido o melhor modelo. As diferenças por fase e por treinamento acima delimitam o desempenho observado. Uma média elevada não significa generalização perfeita.',
    '', 'As fases dos dois conjuntos são diferentes; a diferença de taxas mistura a composição dos conjuntos e possível dificuldade de generalização. Os baselines fornecem contexto, não eliminam esse fator e não estabelecem causalidade. A avaliação mede recombinação de segmentos conhecidos, não classes inteiramente novas. A receita foi selecionada em fases conhecidas. Cinco treinamentos e seis fases reservadas limitam precisão e alcance. A inferência determinística foi verificada neste hardware/software; outro backend pode produzir pequenas diferenças numéricas.','',
    '## Próximos experimentos possíveis','',
    'Qualquer ajuste após esta abertura deve ser declarado posterior ao teste. Possíveis extensões: análise de erros nas partidas registradas, estudo previamente especificado de mudanças de distribuição e novos conjuntos reservados independentes. Nenhuma extensão foi executada aqui.','',
    '## Figuras','', '![Conjuntos](figuras/comparacao_conjuntos.png)','', '![Por fase](figuras/por_fase.png)','', '![Diferenças](figuras/diferenca_generalizacao.png)','',
    '## Dados, reprodução e continuidade','',
    'partidas.csv contém um registro único por agente/fase/semente: grupo, treinamento, caminho/hash do checkpoint, hash do protocolo, k, dificuldade, vitória, dano, decisões, contagens de ações, tempo e UTC. Ações: [esperar, girassol, ervilha, repetidora, batata, mina]. O dano é a soma das recompensas sem o bônus terminal ±1. O tempo por partida inclui espera de inferência compartilhada; não somar esses tempos como tempo de parede do experimento.',
    '', 'A fonte transacional é resultados.sqlite (WAL e synchronous=FULL), com chave primária e erros. Os CSVs derivam dessa base; nunca são usados para retomar. resumo_agente_conjunto.csv, resumo_cnn1d.csv, resumo_por_fase.csv, diferencas_baselines.csv, comparacoes_pareadas.csv e incerteza_bootstrap.csv documentam agregações. O protocolo registra o ambiente e hashes de todos os fontes usados.',
    '', 'Para um futuro visualizador, usar partidas.csv, protocolo.json, os checkpoints originais, segmentos/semente/k/dificuldade e os hashes. O simulador determinístico permite reconstruir uma partida com a mesma política; não foram guardados estados por tick nem implementado front-end. Contagens de ações auxiliam seleção de exemplos sem substituir a sequência de ações.']
    (run/'relatorio.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    brief=f'''# Resumo para apresentação

A CNN1D PPO v2a já havia obtido 96,2% ± 1,1 ponto nas combinações conhecidas, em cinco treinamentos, e 95,8% ± 0,5 em sementes novas de três desses modelos. Isso ainda não demonstrava recombinação de fases.

Congelamos os cinco pesos finais e avaliamos 21 combinações conhecidas e seis reservadas, 50 sementes novas por combinação, sem alterar pesos, com k=20 e dificuldade 1,0. Econ5 e Guloso jogaram as mesmas partidas. Foram 9.450 partidas.

Nas inéditas, CNN1D: {fmt(model_means[:,1].mean())}% ± {fmt(model_means[:,1].std(ddof=1))} entre treinamentos; nas conhecidas: {fmt(model_means[:,0].mean())}% ± {fmt(model_means[:,0].std(ddof=1))}. Diferença inéditas − conhecidas: {100*gaps.mean():+.2f} pontos. Econ5 nas inéditas: {fmt(wins[5,21:].mean())}%; Guloso: {fmt(wins[6,21:].mean())}%.

O resultado descreve generalização combinatória neste simulador e protocolo. Não demonstra perfeição nem generalização para novas regras ou classes. Há apenas cinco treinamentos e seis fases reservadas; a média deve ser acompanhada dos resultados por combinação, incertezas e baselines. Qualquer ajuste futuro será posterior à abertura do teste.
'''
    (run/'resumo_apresentacao.md').write_text(brief,encoding='utf-8')
    output_hashes={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.suffix in ('.csv','.png','.pdf','.md')}
    atomic(run/'artefatos_analise.json',dict(utc=now(),hashes=output_hashes))
    print(json.dumps(dict(cnn=agg,baselines=[r for r in summary if r['agente'] in ('econ5','guloso')],diferencas=differences,partidas=len(rows)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run.resolve())
