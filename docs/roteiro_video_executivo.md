# Roteiro para vídeo executivo (até 5 minutos)

## 0:00-0:35 - Problema

“A alfabetização no início da trajetória escolar condiciona o restante da aprendizagem. Hoje, gestores precisam identificar onde o apoio pode chegar antes que a defasagem se aprofunde.” Mostre o objetivo: estimar risco de não alfabetização para apoiar a priorização de políticas.

## 0:35-1:20 - Dados e governança

Apresente a tabela pública do INEP/Base dos Dados, a camada Gold da Fase 2 e as variáveis utilizadas. Explique que proficiência não foi usada porque vazaria a resposta da prova. Destaque que nenhum dado de aluno é armazenado no repositório.

## 1:20-2:20 - Método confiável

Mostre o diagrama: BigQuery → preparação → split por município → pipeline scikit-learn → avaliação → relatório. Diga que imputação e encoding estão no pipeline, e que municípios de teste não aparecem no treino.

## 2:20-3:20 - Resultado e leitura

Após a execução, apresente ROC-AUC, Average Precision e o gráfico de importância gerados na pasta `reports/` e `images/`. Não diga que importância é causalidade. Mostre um ranking agregado por município somente com validação e revisão humana.

## 3:20-4:20 - Valor para políticas públicas

Explique que o sinal permite orientar apoio pedagógico, formação e acompanhamento territorial. A decisão continua sendo humana, combinada com contexto local e capacidade de atendimento.

## 4:20-5:00 - Limites e próximos passos

Fale sobre auditoria de viés, validação temporal, integração de Censo/IBGE na Gold e calibração. Feche: “A solução transforma dado público em um instrumento reproduzível e responsável para priorizar a alfabetização.”
