# Documentação técnica

## Problema e unidade de análise

O enunciado propõe classificação supervisionada de alfabetização. Entretanto, a fonte indicada é a tabela `uf`, cujo grão é UF/ano/série/rede. A unidade deste experimento é a **rede pública agregada de uma UF**, não o aluno.

O alvo indica se a taxa pública observada em 2024 foi maior ou igual à meta oficial de 2024. A tabela analítica usa somente medidas de desempenho de 2023 e a meta previamente publicada para 2024. Das UFs presentes, 24 possuem o par temporal completo; 13 não atingiram e 11 atingiram a meta.

## Preparação e controle de vazamento

1. Filtragem documentada para 2º ano (`serie=2`) e rede pública (`rede=5`).
2. Validação de chaves únicas por UF/ano antes das junções.
3. Junção dos resultados de 2023, resultados de 2024 e metas oficiais.
4. Construção do rótulo somente após a junção.
5. Exclusão explícita da taxa, média e níveis de proficiência de 2024 das features.
6. Imputação e padronização dentro de `Pipeline`, ajustadas apenas nas dobras de treino.

## Avaliação

Foi usada validação cruzada estratificada aninhada: quatro dobras externas para estimar generalização e três internas para selecionar hiperparâmetros. O método produz uma predição fora da amostra para cada uma das 24 UFs, embora a amostra continue muito pequena.

| Candidato | ROC-AUC | AP | Matriz de confusão `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | --- |
| Regressão logística | 0,4965 | 0,5307 | `[[7, 6], [5, 6]]` |
| Random Forest | 0,3881 | 0,4507 | `[[6, 7], [6, 5]]` |
| Dummy por prevalência | 0,4371 | 0,4356 | `[[9, 4], [9, 2]]` |

A regressão logística foi mantida como melhor candidata relativa, mas recebeu a decisão `DO_NOT_DEPLOY`: ROC-AUC próxima de 0,5 não demonstra discriminação útil. O modelo serializado é apenas um artefato experimental local e está ignorado pelo Git.

## Interpretação e segmentação

Os coeficientes absolutos da regressão são exportados em `reports/importancia_variaveis.csv`. Eles não provam causalidade. O clustering exploratório usa somente os resultados de 2024 padronizados (taxa e média de Português), testa k de 2 a 4 e seleciona k=3 por silhouette de 0,537. Os grupos estão em `reports/clusters_ufs.csv`.

## Rastreabilidade

- Consulta: `scripts/extract_bigquery.sql`.
- Dados brutos: `data/raw/`.
- Tabela modelada: `data/processed/base_modelagem.csv`.
- Métricas completas: `reports/metricas.json` e `reports/clustering_metricas.json`.
- Código de preparação, treino e gráficos: `src/`.
- Semente: 42.

## Limitações

Não é possível inferir risco individual, municipal ou escolar; avaliar equidade entre grupos de alunos; nem atribuir causalidade. A média entre UFs é simples e não representa indicador nacional ponderado. Uma solução operacional exigiria mais anos, maior granularidade e variáveis de contexto, seguida de validação temporal e territorial externa.
