# Tech Challenge - Fase 3 | Predição de Alfabetização

## Contexto e objetivo

Esta solução apoia decisões educacionais ao estimar a probabilidade de um aluno ser considerado alfabetizado. O problema é uma classificação supervisionada binária, com rótulo `alfabetizado` da tabela pública `basedosdados.br_inep_avaliacao_alfabetizacao.alunos`. O objetivo não é substituir a avaliação oficial: é gerar um sinal de risco para orientar investigação, alocação de apoio pedagógico e monitoramento de desigualdades.

## Dados e escopo

A extração usa o BigQuery informado no desafio e salva um snapshot local em `data/raw/alunos.parquet` (ignorado pelo Git). As variáveis candidatas são ano, município, escola, série, rede, presença e preenchimento do caderno. `proficiencia` foi deliberadamente excluída: ela é um resultado da mesma avaliação que compõe o rótulo e produziria **data leakage**. O mesmo vale para qualquer campo calculado após a prova.

O repositório contém apenas código e artefatos vazios; dados de alunos não são versionados por privacidade, tamanho e reprodutibilidade. A conexão exige uma conta GCP autorizada a executar consultas no projeto de cobrança escolhido.

## Pipeline de modelagem

1. Extração SQL parametrizada da tabela pública.
2. Normalização explícita do rótulo para 0/1 e descarte de categorias ambíguas.
3. Separação treino/validação/teste por **município**, impedindo que a mesma localidade apareça em treino e teste.
4. Imputação dentro do `Pipeline` do scikit-learn, one-hot encoding com `handle_unknown="ignore"` e Random Forest com balanceamento de classe.
5. Busca de hiperparâmetros somente no conjunto de treino, validação por ROC-AUC e avaliação final no teste separado.
6. Exportação de métricas, predições, importância de variáveis e gráficos.

A transformação é treinada junto ao modelo. Portanto, ajustes de imputação e categorias são aprendidos apenas no treino. O split territorial reforça a generalização e reduz a chance de o modelo memorizar escolas ou municípios.

## Como executar

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
# configure as credenciais padrão do Google e edite o projeto no ambiente
set GCP_PROJECT_ID=seu-projeto-gcp
python run_pipeline.py --extract --max-rows 500000
pytest -q
```

Para executar sobre um snapshot já extraído:

```bash
python run_pipeline.py --input data/raw/alunos.parquet
```

## Avaliação e interpretação

As métricas geradas em `reports/metricas.json` incluem ROC-AUC, Average Precision e relatório de classificação do conjunto de teste. Os resultados numéricos não são preenchidos no repositório porque dependem do recorte e da data de extração. `reports/importancia_variaveis.csv` e `images/importancia_variaveis.png` permitem inspecionar a relevância preditiva; essa relevância descreve associação no modelo e não causalidade.

## Perguntas estratégicas que a solução apoia

- **Fatores associados:** importância de variáveis e análise por recortes de série, rede e presença.
- **Risco educacional:** agregação da probabilidade média por município/escola após validação e revisão humana.
- **Padrões regionais:** o arquivo de predições pode ser unido ao diretório de municípios da Gold da Fase 2 para UF/região.
- **Metas futuras:** combinar as probabilidades com metas municipais/UF da Gold e observar a evolução temporal, sem usar informações futuras no treino.

## Limitações e cuidados éticos

- A base mede desempenho de uma avaliação, não todo o potencial da criança.
- Predições individuais não devem ser usadas para punição, exclusão ou rotulação de alunos; decisões devem ter supervisão pedagógica e análise de viés.
- Variáveis territoriais podem refletir desigualdades históricas. É obrigatório monitorar métricas por região, rede e outros grupos disponíveis antes de qualquer uso operacional.
- O dataset pode ter cobertura temporal e representatividade limitadas; validação temporal é a evolução prioritária quando houver anos suficientes.

## Aplicação em políticas públicas e próximos passos

Gestores podem priorizar apoio técnico, formação e busca ativa em locais com maior concentração de risco, sempre combinando o sinal do modelo com evidências qualitativas. Evoluções recomendadas: integrar dados socioeconômicos e Censo Escolar na Gold, validação temporal, calibração de probabilidades, auditoria de equidade e explicabilidade local (SHAP) aprovada sob requisitos de privacidade.

## Estrutura

```text
data/                 # snapshots locais ignorados pelo Git
scripts/              # SQL de extração
src/preprocessing/    # extração, rótulo e limpeza
src/modeling/         # split, pipeline e treino
src/evaluation/       # explicabilidade
src/visualization/    # gráficos
reports/ e images/    # saídas reproduzíveis
tests/                # testes unitários
docs/                 # documentação e roteiro executivo
```

## Versionamento

Fluxo recomendado: branches `feature/*`, pull request para `develop` e merge revisado em `main`. Cada alteração de dados, hipótese, variável e métrica deve ser registrada no PR.
