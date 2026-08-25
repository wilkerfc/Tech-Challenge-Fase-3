-- Fonte pública: basedosdados.br_inep_avaliacao_alfabetizacao
SELECT * FROM `basedosdados.br_inep_avaliacao_alfabetizacao.uf`
ORDER BY ano, sigla_uf, serie, rede;

SELECT * FROM `basedosdados.br_inep_avaliacao_alfabetizacao.meta_alfabetizacao_uf`
ORDER BY ano, sigla_uf, rede;

SELECT * FROM `basedosdados.br_inep_avaliacao_alfabetizacao.dicionario`
WHERE id_tabela = 'uf' AND nome_coluna IN ('rede', 'serie')
ORDER BY nome_coluna, chave;
