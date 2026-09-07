# Linhagem dos dados e construção da camada Gold

## Limite de proveniência

O snapshot atualmente versionado em `data/indicador_municipio.csv` é a entrada
Silver deste repositório. Ele contém identificadores e nomes anonimizados, como
`MUNICIPIO_RO_1`, mas não veio acompanhado de consulta, URL, data de extração ou
credencial de uma fonte oficial. Portanto, este projeto **não o apresenta como
uma extração oficial auditável**.

A pipeline adicionada torna reproduzível o trecho que pode ser comprovado:

```text
fonte externa (não registrada no snapshot atual)
        ↓
data/indicador_municipio.csv — camada Silver normalizada
        ↓  src/data_pipeline/build_gold.py
data/generated_gold/ml_features.* — Gold para Machine Learning
data/generated_gold/evolucao_uf.* — Gold agregado por UF
data/generated_gold/painel_nacional.* — Gold agregado nacional
data/generated_gold/manifest.json — hash e contagens da execução
```

Para completar a proveniência externa, forneça uma Silver obtida de uma fonte
oficial e informe sua URI com `--source-uri`. O manifesto registra o caminho, o
SHA-256, a quantidade de linhas e a regra usada para construir o alvo.

## Contrato da camada Silver

Cada linha deve representar um único par município/ano e conter:

- identificação: `id_municipio`, `nome`, `sigla_uf`, `id_uf`, `nome_uf`, `regiao`, `ano`;
- alfabetização e metas: `indicador_alfabetizacao`, `meta_municipio`, `meta_nacional`;
- contexto: `quantidade_matriculas`, `PIB_per_capita`, `IDHM`.

A pipeline rejeita chaves duplicadas, nulos obrigatórios, percentuais fora de
0–100, IDHM fora de 0–1 e valores econômicos negativos.

## Regra do alvo e proteção temporal

`meta_atingida` é recalculada como:

```text
indicador_alfabetizacao do ano t >= meta_municipio do ano t
```

Qualquer coluna de alvo recebida na Silver é ignorada. As features históricas
usam apenas `shift(1)` e `shift(2)` dentro de cada município. Linhas sem pelo
menos um ano anterior são removidas da Gold de Machine Learning.

## Execução

```bash
python -m src.data_pipeline.build_gold \
  --input data/indicador_municipio.csv \
  --output-dir data/generated_gold \
  --source-uri "URI ou identificação da extração Silver"
```

CSV e Parquet são gerados conjuntamente. O diretório de saída é ignorado pelo
Git para evitar sobrescrita acidental dos snapshots e artefatos existentes.

## Testes

```bash
pytest -q
```

Os testes usam dados artificiais mínimos e verificam contrato, duplicidade,
faixas válidas, lags, alvo, agregações, separação temporal e ausência das
colunas de leakage na matriz de atributos.
