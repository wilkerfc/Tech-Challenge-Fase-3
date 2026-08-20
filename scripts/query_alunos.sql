-- Fonte: basedosdados.br_inep_avaliacao_alfabetizacao.alunos
-- `proficiencia` e `peso_aluno` nao entram no modelo: sao consequencias/insumos
-- da avaliacao que define o rotulo e causariam vazamento de dados.
SELECT
  ano, id_municipio, id_escola, serie, rede, presenca,
  preenchimento_caderno, alfabetizado
FROM `basedosdados.br_inep_avaliacao_alfabetizacao.alunos`
WHERE alfabetizado IS NOT NULL
  AND id_municipio IS NOT NULL
  AND presenca IS NOT NULL
ORDER BY ano, id_municipio, id_escola
LIMIT @max_rows;
