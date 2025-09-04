"""
Módulo de Análise Preditiva e Otimização de Localização
Implementa modelos de previsão e algoritmos p-mediana/p-hub
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost não disponível, usando modelo simplificado")
    XGBOOST_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    print("⚠️ PuLP não disponível, usando otimização simplificada")
    PULP_AVAILABLE = False

from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalyzer:
    """Classe para análise preditiva e otimização de localização"""
    
    def __init__(self, config):
        self.config = config
        self.xgboost_model = None
        
    def generate_synthetic_demand_data(self) -> pd.DataFrame:
        """Gera dados sintéticos de demanda histórica para treinamento"""
        # Simula dados históricos de 3 anos
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='M')
        
        # Tendência base com sazonalidade
        trend = np.linspace(100000, 150000, len(dates))
        seasonality = 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 5000, len(dates))
        
        # Efeito COVID (redução em 2022)
        covid_effect = np.where(dates < '2023-01-01', -15000, 0)
        
        demand = trend + seasonality + noise + covid_effect
        demand = np.maximum(demand, 50000)  # Demanda mínima
        
        df = pd.DataFrame({
            'ds': dates,
            'y': demand,
            'regiao': np.random.choice(['nordeste'], len(dates)),
            'pib_regional': np.random.normal(250000, 25000, len(dates)),
            'populacao': np.random.normal(15000000, 500000, len(dates)),
            'ecommerce_penetration': np.linspace(0.15, 0.35, len(dates))
        })
        
        return df
    
    def simple_forecast_model(self, df: pd.DataFrame) -> Dict:
        """Modelo de previsão simplificado sem Prophet"""
        print("Gerando previsões com modelo simplificado...")
        
        # Calcular tendência linear
        x = np.arange(len(df))
        y = df['y'].values
        
        # Regressão linear simples
        z = np.polyfit(x, y, 1)
        trend_func = np.poly1d(z)
        
        # Calcular sazonalidade média
        df['month'] = df['ds'].dt.month
        seasonal_pattern = df.groupby('month')['y'].mean()
        overall_mean = df['y'].mean()
        seasonal_factors = seasonal_pattern / overall_mean
        
        # Gerar previsões para próximos 24 meses
        future_dates = pd.date_range(start='2025-01-01', end='2026-12-31', freq='M')
        future_x = np.arange(len(df), len(df) + len(future_dates))
        
        future_predictions = []
        for i, date in enumerate(future_dates):
            trend_value = trend_func(future_x[i])
            seasonal_factor = seasonal_factors[date.month]
            prediction = trend_value * seasonal_factor
            future_predictions.append(prediction)
        
        future_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': future_predictions,
            'yhat_lower': [p * 0.9 for p in future_predictions],
            'yhat_upper': [p * 1.1 for p in future_predictions]
        })
        
        return {
            'forecast': future_df,
            'model': 'LinearTrend+Seasonality',
            'future_demand': future_df
        }
    
    def train_xgboost_model(self, df: pd.DataFrame) -> Dict:
        """Treina modelo XGBoost para previsão de demanda por região"""
        if not XGBOOST_AVAILABLE:
            return self.simple_ml_model(df)
        
        print("Treinando modelo XGBoost...")
        
        # Preparar features
        df_ml = df.copy()
        df_ml['month'] = df_ml['ds'].dt.month
        df_ml['year'] = df_ml['ds'].dt.year
        df_ml['quarter'] = df_ml['ds'].dt.quarter
        
        # Features para o modelo
        features = ['month', 'year', 'quarter', 'pib_regional', 'populacao', 'ecommerce_penetration']
        X = df_ml[features]
        y = df_ml['y']
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar XGBoost
        self.xgboost_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.xgboost_model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.xgboost_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Importância das features
        feature_importance = dict(zip(features, self.xgboost_model.feature_importances_))
        
        # Previsões futuras
        future_features = self.generate_future_features()
        future_predictions = self.xgboost_model.predict(future_features)
        
        return {
            'model': self.xgboost_model,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': feature_importance,
            'future_predictions': future_predictions,
            'future_features': future_features
        }
    
    def simple_ml_model(self, df: pd.DataFrame) -> Dict:
        """Modelo de ML simplificado usando sklearn"""
        from sklearn.linear_model import LinearRegression
        
        print("Treinando modelo de regressão linear...")
        
        # Preparar features
        df_ml = df.copy()
        df_ml['month'] = df_ml['ds'].dt.month
        df_ml['year'] = df_ml['ds'].dt.year
        df_ml['quarter'] = df_ml['ds'].dt.quarter
        
        features = ['month', 'year', 'quarter', 'pib_regional', 'populacao', 'ecommerce_penetration']
        X = df_ml[features]
        y = df_ml['y']
        
        # Treinar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Previsões futuras
        future_features = self.generate_future_features()
        future_predictions = model.predict(future_features)
        
        return {
            'model': model,
            'mae': 0,  # Placeholder
            'rmse': 0,  # Placeholder
            'feature_importance': dict(zip(features, abs(model.coef_))),
            'future_predictions': future_predictions,
            'future_features': future_features
        }
    
    def generate_future_features(self) -> pd.DataFrame:
        """Gera features para previsões futuras"""
        future_months = pd.date_range(start='2025-01-01', end='2026-12-31', freq='M')
        
        future_df = pd.DataFrame({
            'month': future_months.month,
            'year': future_months.year,
            'quarter': future_months.quarter,
            'pib_regional': np.linspace(275000, 320000, len(future_months)),
            'populacao': np.linspace(15500000, 16000000, len(future_months)),
            'ecommerce_penetration': np.linspace(0.35, 0.50, len(future_months))
        })
        
        return future_df
    
    def solve_p_median_problem(self, dados: Dict, num_facilities: int = 1) -> Dict:
        """Resolve problema p-mediana para otimização de localização"""
        if not PULP_AVAILABLE:
            return self.simple_optimization(dados)
        
        print("Resolvendo problema p-mediana...")
        
        # Candidatos (Recife e Salvador)
        candidates = {
            'Recife': self.config.RECIFE_COORDS,
            'Salvador': self.config.SALVADOR_COORDS
        }
        
        # Pontos de demanda (capitais do Nordeste)
        demand_points = self.config.CAPITAIS_NORDESTE
        
        # Calcular matriz de distâncias
        distance_matrix = {}
        for cand_name, cand_coords in candidates.items():
            distance_matrix[cand_name] = {}
            for demand_name, demand_coords in demand_points.items():
                dist = geodesic(cand_coords, demand_coords).kilometers
                distance_matrix[cand_name][demand_name] = dist
        
        # Criar problema de otimização
        prob = pulp.LpProblem("P_Median_CD_Location", pulp.LpMinimize)
        
        # Variáveis de decisão
        x = pulp.LpVariable.dicts("facility", candidates.keys(), cat='Binary')
        y = pulp.LpVariable.dicts("assignment", 
                                [(i, j) for i in candidates.keys() for j in demand_points.keys()], 
                                cat='Binary')
        
        # Pesos de demanda
        demand_weights = {
            "Fortaleza": 2686612, "Natal": 890480, "João Pessoa": 817511,
            "Maceió": 1025360, "Aracaju": 664908, "São Luís": 1108975,
            "Teresina": 868075, "Petrolina": 354317
        }
        
        # Função objetivo
        prob += pulp.lpSum([demand_weights.get(j, 100000) * distance_matrix[i][j] * y[i,j] 
                           for i in candidates.keys() for j in demand_points.keys()])
        
        # Restrições
        prob += pulp.lpSum([x[i] for i in candidates.keys()]) == num_facilities
        
        for j in demand_points.keys():
            prob += pulp.lpSum([y[i,j] for i in candidates.keys()]) == 1
        
        for i in candidates.keys():
            for j in demand_points.keys():
                prob += y[i,j] <= x[i]
        
        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extrair resultados
        selected_facilities = [i for i in candidates.keys() if x[i].varValue == 1]
        assignments = {j: i for i in candidates.keys() for j in demand_points.keys() 
                      if y[i,j].varValue == 1}
        
        total_cost = pulp.value(prob.objective)
        
        return {
            'selected_facilities': selected_facilities,
            'assignments': assignments,
            'total_weighted_distance': total_cost,
            'optimization_status': pulp.LpStatus[prob.status]
        }
    
    def simple_optimization(self, dados: Dict) -> Dict:
        """Otimização simplificada sem PuLP"""
        print("Executando otimização simplificada...")
        
        # Calcular score simples baseado em distância média ponderada
        candidates = ['Recife', 'Salvador']
        demand_points = self.config.CAPITAIS_NORDESTE
        
        scores = {}
        for cidade in ['recife', 'salvador']:
            total_weighted_distance = 0
            for dest_city, dest_coords in demand_points.items():
                if cidade == 'recife':
                    origin_coords = self.config.RECIFE_COORDS
                else:
                    origin_coords = self.config.SALVADOR_COORDS
                
                distance = geodesic(origin_coords, dest_coords).kilometers
                population = self.get_population_estimate(dest_city)
                total_weighted_distance += distance * population
            
            scores[cidade.capitalize()] = total_weighted_distance
        
        # Selecionar a cidade com menor distância total ponderada
        selected_city = min(scores, key=scores.get)
        
        return {
            'selected_facilities': [selected_city],
            'assignments': {city: selected_city for city in demand_points.keys()},
            'total_weighted_distance': scores[selected_city],
            'optimization_status': 'Optimal'
        }
    
    def calculate_network_efficiency(self, dados: Dict) -> Dict:
        """Calcula métricas de eficiência da rede logística"""
        efficiency_metrics = {}
        
        for cidade in ['recife', 'salvador']:
            cidade_name = cidade.capitalize()
            distancias = dados[cidade]['distancias']
            
            # Métricas de cobertura
            tempos_entrega = [d['tempo_horas'] for d in distancias.values()]
            custos_frete = [d['custo_frete_estimado'] for d in distancias.values()]
            
            # População total coberta
            pop_12h = sum([self.get_population_estimate(city) for city, data in distancias.items() 
                          if data['tempo_horas'] <= 12])
            pop_24h = sum([self.get_population_estimate(city) for city, data in distancias.items() 
                          if data['tempo_horas'] <= 24])
            
            efficiency_metrics[cidade_name] = {
                'tempo_medio_entrega': np.mean(tempos_entrega),
                'tempo_max_entrega': np.max(tempos_entrega),
                'custo_medio_frete': np.mean(custos_frete),
                'populacao_12h': pop_12h,
                'populacao_24h': pop_24h,
                'cobertura_12h_pct': (pop_12h / 15000000) * 100,
                'cobertura_24h_pct': (pop_24h / 15000000) * 100,
                'indice_acessibilidade': pop_24h / np.mean(tempos_entrega)
            }
        
        return efficiency_metrics
    
    def get_population_estimate(self, city: str) -> int:
        """Retorna estimativa populacional por cidade"""
        populations = {
            "Fortaleza": 2686612, "Natal": 890480, "João Pessoa": 817511,
            "Maceió": 1025360, "Aracaju": 664908, "São Luís": 1108975,
            "Teresina": 868075, "Petrolina": 354317
        }
        return populations.get(city, 500000)
    
    def run_predictive_analysis(self, dados: Dict) -> Dict:
        """Executa análise preditiva completa"""
        print("Executando análise preditiva e otimização...")
        
        # Gerar dados sintéticos
        demand_data = self.generate_synthetic_demand_data()
        
        # Treinar modelos
        simple_forecast = self.simple_forecast_model(demand_data)
        xgb_results = self.train_xgboost_model(demand_data)
        
        # Otimização de localização
        p_median_results = self.solve_p_median_problem(dados)
        
        # Análise de eficiência
        network_efficiency = self.calculate_network_efficiency(dados)
        
        relatorio_preditivo = {
            'previsao_demanda': {
                'simple_forecast': simple_forecast,
                'xgboost': xgb_results,
                'crescimento_esperado_24m': simple_forecast['future_demand']['yhat'].mean()
            },
            'otimizacao_localizacao': p_median_results,
            'eficiencia_rede': network_efficiency,
            'recomendacao_otimizacao': p_median_results['selected_facilities'][0],
            'justificativa_tecnica': self.generate_technical_justification(
                p_median_results, network_efficiency, simple_forecast
            )
        }
        
        print("Análise preditiva concluída!")
        return relatorio_preditivo
    
    def generate_technical_justification(self, p_median: Dict, efficiency: Dict, forecast: Dict) -> str:
        """Gera justificativa técnica da recomendação"""
        selected_city = p_median['selected_facilities'][0]
        efficiency_city = efficiency[selected_city]
        
        justification = f"""
        JUSTIFICATIVA TÉCNICA - OTIMIZAÇÃO P-MEDIANA:
        
        1. COBERTURA POPULACIONAL:
           - {selected_city} oferece cobertura de {efficiency_city['cobertura_24h_pct']:.1f}% da população nordestina em 24h
           - Tempo médio de entrega: {efficiency_city['tempo_medio_entrega']:.1f} horas
        
        2. EFICIÊNCIA LOGÍSTICA:
           - Índice de acessibilidade: {efficiency_city['indice_acessibilidade']:.0f}
           - Custo médio de frete: R$ {efficiency_city['custo_medio_frete']:.2f}
        
        3. OTIMIZAÇÃO MATEMÁTICA:
           - Distância total ponderada minimizada: {p_median['total_weighted_distance']:.0f} km
           - Status da otimização: {p_median['optimization_status']}
        
        4. RECOMENDAÇÃO:
           - {selected_city} apresenta a melhor solução segundo critério p-mediana
           - Otimização matemática confirma vantagem competitiva
        """
        
        return justification.strip()