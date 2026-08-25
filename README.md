# Tech Challenge — Fase 3 | Alfabetização por UF

Solução reproduzível para analisar a alfabetização no 2º ano do Ensino Fundamental e testar, com dados anteriores, se é possível antecipar quais redes públicas estaduais alcançarão a meta oficial de 2024.

## Conclusão executiva

O protótipo **não deve ser colocado em produção**. A regressão logística, melhor candidata avaliada, obteve ROC-AUC de **0,497** em validação cruzada aninhada — desempenho equivalente ao acaso. Esse resultado negativo é relevante: com apenas os agregados estaduais disponíveis nesta tabela, não há evidência de sinal preditivo suficiente para orientar decisões automatizadas.

A entrega permanece útil como monitor descritivo, pipeline auditável e ponto de partida para uma versão futura com dados municipais, escolares e socioeconômicos.

## Escopo correto dos dados

A fonte indicada é `basedosdados.br_inep_avaliacao_alfabetizacao.uf`, uma tabela **agregada por UF, ano, série e rede**. Ela não contém registros de alunos. Por isso, este projeto não afirma prever alfabetização individual: o alvo experimental é `1` quando a taxa observada da rede pública em 2024 alcança a meta oficial da UF, e `0` caso contrário.

Foram versionados os snapshots consultados no BigQuery:

- `data/raw/uf.csv`: 145 linhas de resultados de 2023 e 2024;
- `data/raw/meta_alfabetizacao_uf.csv`: 81 metas oficiais de 2023 a 2025;
- `data/raw/dicionario_uf.csv`: códigos usados no recorte (`rede=5`, pública estadual e municipal; `serie=2`, 2º ano).

A base de modelagem contém 24 UFs com observações públicas emparelhadas em 2023 e 2024. Não há imputação artificial de UFs ausentes.

## Metodologia

As variáveis de entrada são a taxa de alfabetização de 2023, a média de Língua Portuguesa de 2023 e a meta oficial de 2024. O resultado observado de 2024 é usado somente para construir o rótulo e avaliar o modelo; nunca entra como atributo preditor.

O pipeline aplica imputação e padronização dentro de cada dobra. A comparação usa validação cruzada estratificada aninhada (4 dobras externas e 3 internas), evitando selecionar hiperparâmetros nos próprios dados de avaliação. Foram comparados regressão logística, Random Forest e um classificador ingênuo pela prevalência.

| Modelo | ROC-AUC | Average Precision | Acurácia balanceada | F1 |
| --- | ---: | ---: | ---: | ---: |
| Regressão logística | 0,497 | 0,531 | 0,542 | 0,522 |
| Random Forest | 0,388 | 0,451 | 0,458 | 0,435 |
| Baseline ingênuo | 0,437 | 0,436 | 0,437 | 0,235 |

As métricas são previsões fora da dobra agregadas. Com apenas 24 casos, têm alta incerteza e não sustentam uso operacional.

## Achados descritivos

- Entre as 24 UFs emparelhadas, a média simples das taxas estaduais passou de 54,25% em 2023 para 56,83% em 2024. Não é uma taxa nacional ponderada.
- Os maiores déficits observados em relação à meta de 2024 foram RS (-21,53 p.p.), AM (-7,63), BA (-7,44), PA (-5,40) e RN (-4,51).
- Uma segmentação exploratória com taxa e média de Português de 2024 escolheu 3 grupos pelo silhouette (0,537). O Ceará formou um grupo isolado; isso descreve similaridade estatística, não causalidade ou recomendação de política.
- Importâncias e correlações são associações exploratórias. A pequena amostra impede afirmar causas da alfabetização.

## Estrutura

```text
data/raw/           snapshots e dicionário do BigQuery
data/processed/     tabela analítica gerada
src/preprocessing/  validação e junção temporal
src/modeling/       classificação e clustering
src/visualization/  gráficos reproduzíveis
scripts/            SQL usado na extração
reports/            métricas, predições e relatório
images/             visualizações finais
tests/              teste de integridade temporal
```

## Reprodução

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
pytest -q
```

O SQL completo está em `scripts/extract_bigquery.sql`. Os CSVs já incluídos permitem executar o pipeline sem credenciais do Google Cloud. A semente aleatória é 42.

## Limitações e uso responsável

- Dois anos e 24 UFs são insuficientes para generalização robusta.
- A tabela não permite responder quais alunos, escolas ou municípios estão em risco.
- Não há atributos socioeconômicos, de infraestrutura, docentes ou investimento.
- A meta oficial é conhecida previamente, mas é também relacionada ao contexto estadual; seu uso deve ser reavaliado em validações temporais futuras.
- Os déficits observados servem para monitoramento e investigação humana, não para ranquear qualidade de gestores ou punir redes.

Para avançar, recomenda-se ampliar a série histórica e usar dados no nível municipal/escolar, mantendo separação temporal e territorial, auditoria de viés e avaliação de calibração.
