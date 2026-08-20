# Documentação técnica e decisões analíticas

## Decisão de unidade de análise

O enunciado pede predição do status de alfabetização do aluno. A tabela `alunos` é, por isso, a fonte principal. As tabelas `uf`, `municipio` e metas da camada Gold devem ser usadas para análise agregada e enriquecimento futuro, mas não como substitutas do rótulo individual.

## Controles contra vazamento

| Risco | Controle implementado |
| --- | --- |
| Proficiência determina/é medida da avaliação do rótulo | Excluída da consulta e das features |
| Estatísticas de imputação calculadas em toda a base | `SimpleImputer` dentro do `Pipeline` |
| Categoria nova na validação/teste | `OneHotEncoder(handle_unknown="ignore")` |
| Mesma localidade em treino e teste | Split por `id_municipio` |
| Escolha de parâmetros vendo o teste | Teste permanece separado até a avaliação final |

## Reprodutibilidade

O `RANDOM_STATE` padrão é 42. A consulta é versionada em `scripts/query_alunos.sql`, é parametrizada por limite de linhas e ordenada deterministicamente. Para auditoria, registre no PR: projeto de cobrança, horário de extração, hash do arquivo local, linhas extraídas, distribuição do alvo e versão de dependências.

## Critério de operacionalização

Não há limiar de ação fixo. O limiar deve ser escolhido com gestores considerando capacidade de atendimento, custo de falso negativo e métricas de equidade. A saída é recomendação de priorização, não decisão automatizada.
