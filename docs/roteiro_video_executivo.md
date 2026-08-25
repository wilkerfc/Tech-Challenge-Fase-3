# Roteiro do vídeo executivo — até 5 minutos

## 0:00–0:40 — Objetivo e honestidade do escopo

Apresente o desafio e a fonte. Explique que a tabela indicada é agregada por UF, não contém alunos, e que o experimento prevê se a rede pública estadual alcança a meta oficial de 2024 usando resultados de 2023.

## 0:40–1:25 — Dados

Mostre os três snapshots: resultados por UF, metas oficiais e dicionário. Destaque o recorte do 2º ano e rede pública. A base final tem 24 UFs emparelhadas, sendo 11 metas atingidas e 13 não atingidas.

## 1:25–2:20 — Pipeline confiável

Mostre o fluxo BigQuery → validação e junção temporal → pipeline → validação cruzada aninhada → relatórios. Explique que os resultados de 2024 constroem apenas o rótulo; as features vêm de 2023, além da meta conhecida para 2024. Imputação e escala são aprendidas somente no treino.

## 2:20–3:15 — Resultado do modelo

Apresente a ROC-AUC de 0,497 da regressão logística, contra 0,388 da Random Forest e 0,437 do baseline. Diga claramente: o desempenho é equivalente ao acaso e a recomendação é **não implantar**. Não transforme um resultado fraco em promessa de negócio.

## 3:15–4:15 — Leitura descritiva

Mostre os gráficos. Nas 24 UFs emparelhadas, a média simples subiu de 54,25% para 56,83%. Os maiores déficits observados frente à meta de 2024 foram RS, AM, BA, PA e RN. Apresente os três clusters como semelhanças exploratórias, sem atribuir causas.

## 4:15–5:00 — Valor, limites e próximos passos

Conclua que o pipeline oferece monitoramento reproduzível e evita decisões sustentadas por um modelo sem evidência. Para evoluir: incorporar séries históricas maiores e dados municipais/escolares e socioeconômicos; validar no tempo e no território; medir calibração e viés. Feche reforçando que decisão educacional deve permanecer humana e contextualizada.
