# Configurações do Projeto - Análise de Localização CD Magalu
# ===========================================================

# APIs e Serviços Externos
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
GRAPHHOPPER_API_URL = "https://graphhopper.com/api/1"
IBGE_API_URL = "https://servicodados.ibge.gov.br/api/v1"

# Coordenadas das cidades candidatas
RECIFE_COORDS = (-8.0476, -34.8770)
SALVADOR_COORDS = (-12.9714, -38.5014)

# Capitais do Nordeste para análise de cobertura
CAPITAIS_NORDESTE = {
    "Fortaleza": (-3.7327, -38.5267),
    "Natal": (-5.7945, -35.2110),
    "João Pessoa": (-7.1195, -34.8450),
    "Maceió": (-9.6658, -35.7353),
    "Aracaju": (-10.9091, -37.0677),
    "São Luís": (-2.5387, -44.2825),
    "Teresina": (-5.0892, -42.8019),
    "Petrolina": (-9.3891, -40.5006)
}

# Pesos para análise MCDA - CORRIGIDOS para análise mais equilibrada
CRITERIOS_PESOS = {
    "custo_imobiliario": 0.20,    # Reduzido de 0.25
    "tempo_entrega": 0.20,        # Reduzido de 0.30 (estava super-valorizado)
    "potencial_mercado": 0.30,    # Aumentado de 0.20 (mais importante para CD)
    "infraestrutura": 0.20,       # Aumentado de 0.15 (infraestrutura é crucial)
    "tco_operacional": 0.10       # Mantido (já refletido nos outros custos)
}

# Parâmetros de simulação Monte Carlo
MONTE_CARLO_ITERACOES = 10000
CONFIANCA_INTERVALO = 0.95

# Configurações de TCO (Total Cost of Ownership)
TCO_HORIZONTE_ANOS = 10
TAXA_DESCONTO = 0.12

# Configurações de capacidade do CD
CAPACIDADE_CD_M2 = 50000
PRODUTOS_POR_M2 = 150
FUNCIONARIOS_POR_1000M2 = 25