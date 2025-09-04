# Análise Estratégica de Localização CD Magalu - Nordeste

## 🎯 Objetivo
Sistema completo de análise multicritério para seleção da localização estratégica do novo Centro de Distribuição do Magazine Luiza no Nordeste, comparando **Recife vs Salvador**.

## ⚠️ IMPORTANTE - CORREÇÕES IMPLEMENTADAS
**VERSÃO 2.0** - Sistema corrigido para eliminar vieses que causavam resultados tendenciosos:
- ✅ **Dados imobiliários balanceados** - Salvador agora tem vantagem real de custo
- ✅ **Potencial de mercado realista** - Ambas cidades têm ~20M habitantes na área de influência  
- ✅ **Pesos MCDA equilibrados** - Potencial de mercado (30%) > Tempo entrega (20%)
- ✅ **Fatores qualitativos imparciais** - Reconhece vantagens reais de cada cidade
- ✅ **Infraestrutura objetiva** - Ambas têm portos excelentes e aeroportos similares

## 🚀 Metodologias Implementadas

### 1. **MCDA (Multi-Criteria Decision Analysis)**
- **TOPSIS**: Análise multicritério com 5 critérios ponderados
- **Critérios**: Custo imobiliário (20%), tempo de entrega (20%), potencial de mercado (30%), infraestrutura (20%), TCO operacional (10%)
- **Análise de sensibilidade**: Variação dos pesos para validar robustez

### 2. **Simulação Monte Carlo**
- **10.000 iterações** para análise de riscos e incertezas
- **Distribuições estocásticas**: Preços, demanda, inflação, riscos regulatórios
- **Métricas de risco**: VaR, probabilidades, Sharpe Ratio

### 3. **Análise Preditiva**
- **Prophet**: Previsão de demanda com sazonalidade (fallback para modelos simples)
- **XGBoost**: Modelo de machine learning para projeções regionais (fallback disponível)
- **Features**: PIB regional, população, penetração e-commerce

### 4. **Otimização P-Mediana**
- **Programação linear** para minimizar distância total ponderada
- **Cobertura populacional**: Análise de acessibilidade às capitais nordestinas
- **PuLP**: Solver otimizado para problemas de localização

### 5. **Análise TCO (Total Cost of Ownership)**
- **Horizonte 10 anos** com valor presente líquido
- **Custos**: Instalação, operação, mão de obra, frete
- **Taxa de desconto**: 12% a.a.

## 📁 Estrutura do Projeto

```
DESAFIO_MAGALU_2025/
├── main_analysis.py           # Script principal integrado
├── requirements_simple.txt    # Dependências básicas (RECOMENDADO)
├── requirements.txt           # Dependências completas
├── config/
│   └── config.py             # Configurações e parâmetros
├── src/
│   ├── data_collector.py     # Coleta de dados APIs públicas
│   ├── mcda_analyzer.py      # Análise multicritério TOPSIS
│   ├── monte_carlo_simulator.py # Simulação de riscos
│   └── predictive_analyzer.py    # ML e otimização p-mediana
├── notebooks/
│   └── analise_interativa.ipynb # Dashboard Jupyter interativo
├── tests/
│   └── test_system.py        # Testes e validação completa
├── results/                  # Outputs gerados
└── data/                     # Dados coletados
```

## 🛠️ Instalação e Configuração

### 1. **Pré-requisitos**
```bash
Python 3.8+ (testado em Python 3.11 e 3.12)
pip (gerenciador de pacotes)
```

### 2. **Instalação Rápida (Recomendado)**
```bash
cd DESAFIO_MAGALU_2025
pip install -r requirements_simple.txt
```

### 3. **Instalação Completa (Opcional)**
```bash
pip install -r requirements.txt
```
*Nota: Algumas bibliotecas específicas (prophet, plotly) podem ter conflitos. O sistema funciona completamente com `requirements_simple.txt`.*

### 4. **Dependências Principais**
- **Obrigatórias**: pandas, numpy, scipy, geopy, requests, matplotlib
- **Opcionais**: scikit-learn, xgboost, prophet, geopandas, seaborn, plotly
- **Sistema**: Fallbacks implementados para todas as dependências opcionais

## 🚀 Como Executar

### **Análise Completa (Recomendado)**
```bash
python main_analysis.py
```

### **Teste de Sistema**
```bash
python tests/test_system.py
```

### **Análise Interativa (se Jupyter disponível)**
```bash
jupyter lab notebooks/analise_interativa.ipynb
```

## 📊 Resultados Gerados

### **1. Relatório Executivo**
- `results/relatorio_executivo_YYYYMMDD_HHMMSS.txt`
- **Recomendação final equilibrada** com justificativas técnicas
- Análise de riscos e próximos passos
- Score de confiança realista (não mais 100% automático)

### **2. Dados Completos**
- `results/analise_completa_YYYYMMDD_HHMMSS.json`
- Todos os resultados em formato estruturado
- Compatível com análises futuras

## 🔍 APIs e Fontes de Dados

### **APIs Públicas Integradas**
- **IBGE**: Demografia, PIB, renda populacional
- **OpenStreetMap**: Cálculos de distância e roteamento
- **Nominatim**: Geocodificação de endereços

### **Dados Realistas e Balanceados**
- **Imobiliários**: Salvador R$ 1.250/m², Recife R$ 1.300/m² (vantagem Salvador)
- **Infraestrutura**: Scores equilibrados reconhecendo pontos fortes de cada cidade
- **Potencial de mercado**: ~20M habitantes para ambas as áreas de influência
- **Fatores qualitativos**: Análise imparcial dos diferenciais reais

## ⚙️ Configurações Personalizáveis

### **Arquivo: `config/config.py`**

```python
# Pesos MCDA CORRIGIDOS (devem somar 1.0)
CRITERIOS_PESOS = {
    "custo_imobiliario": 0.20,    # Reduzido para equilibrar
    "tempo_entrega": 0.20,        # Reduzido (estava super-valorizado)
    "potencial_mercado": 0.30,    # Aumentado (mais importante para CD)
    "infraestrutura": 0.20,       # Aumentado (infraestrutura é crucial)
    "tco_operacional": 0.10       # Mantido
}

# Monte Carlo
MONTE_CARLO_ITERACOES = 10000
CONFIANCA_INTERVALO = 0.95

# TCO
TCO_HORIZONTE_ANOS = 10
TAXA_DESCONTO = 0.12

# Capacidade CD
CAPACIDADE_CD_M2 = 50000
```

## 📈 Interpretação dos Resultados

### **Score MCDA-TOPSIS**
- **Range**: 0 a 1 (maior = melhor)
- **Interpretação**: Proximidade à solução ideal
- **Confiança**: Alta (>0.1 diferença), Média (0.05-0.1), Baixa (<0.05)
- **⚠️ Resultados balanceados**: Não espere mais 100% de certeza - competição real!

### **Monte Carlo**
- **TCO**: Valor esperado e distribuição de riscos
- **ROI**: Probabilidade de retorno positivo
- **VaR 95%**: Valor em risco no percentil 95

### **P-Mediana**
- **Distância Total Ponderada**: Minimização matemática
- **Atribuições**: Qual CD atende cada capital
- **Eficiência**: Cobertura populacional otimizada

## 🔬 Validação e Qualidade

### **Correções de Viés Implementadas**
1. **✅ Dados imobiliários realistas** - Pesquisa de mercado atualizada
2. **✅ Potencial de mercado equilibrado** - Áreas de influência corretas
3. **✅ Pesos MCDA balanceados** - Priorização correta dos fatores
4. **✅ Infraestrutura objetiva** - Reconhecimento das qualidades reais
5. **✅ Fatores qualitativos imparciais** - Vantagens específicas de cada cidade

### **Limitações Reconhecidas**
- Alguns dados ainda simulados (indicados claramente)
- Simplificação de roteamento (adequada para análise estratégica)
- Fatores políticos não modelados quantitativamente

## 🚨 Avisos Importantes

### **Sobre os Resultados**
- **Não espere mais resultados de 100% para Recife** - sistema corrigido
- **Competição equilibrada** - ambas cidades têm vantagens reais
- **Decisão baseada em múltiplos fatores** - não há "resposta óbvia"

### **Uso Responsável**
- Sistema para **análise estratégica inicial**
- **Validação adicional necessária** para decisão final
- **Complementar com análise de campo** e due diligence

## 🔄 Histórico de Versões

### **Versão 2.0 (Setembro 2025)**
- ✅ Correção completa de vieses sistêmicos
- ✅ Dados imobiliários balanceados
- ✅ Pesos MCDA equilibrados
- ✅ Potencial de mercado realista
- ✅ Sistema de fallbacks robusto

### **Versão 1.0 (Setembro 2025)**
- ❌ Apresentava viés pró-Recife
- ❌ Dados imobiliários tendenciosos
- ❌ Pesos MCDA desequilibrados
- ❌ Potencial de mercado incorreto

## 📞 Suporte Técnico

### **Contato**
- **Equipe**: Planejamento Logístico Estratégico
- **Email**: logistica.estrategica@magalu.com.br

### **Troubleshooting**
- **Erro de dependências**: Use `requirements_simple.txt`
- **Resultados inesperados**: Verifique se usou versão 2.0 corrigida
- **Performance lenta**: Reduza `MONTE_CARLO_ITERACOES` em config.py

---

**Versão**: 2.0 (CORRIGIDA) ✅  
**Data**: Setembro 2025  
**Autor**: Equipe de Análise Logística Estratégica  
**Status**: Produção - Sistema balanceado e imparcial