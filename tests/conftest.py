import pandas as pd
import pytest


@pytest.fixture
def silver_frame():
    rows = []
    cities = [
        (110001, "Cidade A", "RO", 11, "Rondônia", "Norte", 1000, 20000.0, 0.70),
        (120001, "Cidade B", "AC", 12, "Acre", "Norte", 500, 15000.0, 0.65),
    ]
    values = {110001: [40.0, 50.0, 70.0], 120001: [60.0, 55.0, 50.0]}
    metas = {2022: 45.0, 2023: 55.0, 2024: 60.0}
    for city_id, name, uf, uf_id, uf_name, region, students, pib, idhm in cities:
        for year, indicator in zip([2022, 2023, 2024], values[city_id]):
            rows.append({
                "id_municipio": city_id, "nome": name, "sigla_uf": uf,
                "id_uf": uf_id, "nome_uf": uf_name, "regiao": region,
                "ano": year, "indicador_alfabetizacao": indicator,
                "meta_municipio": metas[year], "meta_nacional": metas[year],
                "quantidade_matriculas": students, "PIB_per_capita": pib,
                "IDHM": idhm,
            })
    return pd.DataFrame(rows)
