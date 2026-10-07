"""Separação treino / teste / vitrine (seção 2.6), fixada ANTES de treinar.

- Teste de generalização: 6 combinações que o agente nunca vê no treino, montadas
  só com classes conhecidas. Cada classe aparece exatamente 2 vezes no teste, e
  cada par início+meio perde no máximo uma combinação, então o treino ainda vê
  todas as 9 classes e todos os 9 pares início+meio.
- Treino: as outras 21 combinações.
- Vitrine: 3 fases do teste com sementes fixas, só para avaliação e demonstração.
- Sementes: faixas disjuntas, para nenhuma partida de teste ou vitrine aparecer
  no treino.
"""
from .hordas import todas_as_fases

FASES_TESTE = (
    ("calmo", "misto", "blindados"),
    ("cedo_apressado", "uma_faixa_pesada", "tudo_junto"),
    ("so_rapidos", "rapidos_em_massa", "duas_ondas"),
    ("calmo", "rapidos_em_massa", "duas_ondas"),
    ("cedo_apressado", "misto", "blindados"),
    ("so_rapidos", "uma_faixa_pesada", "tudo_junto"),
)

FASES_TREINO = tuple(f for f in todas_as_fases() if f not in FASES_TESTE)

SEMENTES_TREINO = range(0, 10_000)
SEMENTES_TESTE = range(10_000, 20_000)

# (classes, semente). As três primeiras fases do teste cobrem as 9 classes uma vez cada.
VITRINE = (
    (FASES_TESTE[0], 20_000),
    (FASES_TESTE[1], 20_001),
    (FASES_TESTE[2], 20_002),
)
