# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:50:57 2021

@author: Anderson Lomba
"""

import pandas as pd 
import numpy as np

base = pd.read_csv('credit_data.csv')


# Corrigindo os valores negativos de idade

base.loc[base.age< 0, 'age'] = 40.92 

# Criação da matriz de recursos e da matriz dependente

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:,4].values


# Corrigindo valores faltantes (NaN)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean' )
imputer = imputer.fit(previsores[:, 1:4])
previsores[:,1:4] = imputer.transform(previsores[:,1:4])

# Colocando a base de dados na mesma escala de valores

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores =scaler.fit_transform(previsores)

# Divisão das bases de dados em teste e treinamento(Aprendizagem supervisionada)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,\
    classe, test_size = 0.25, random_state=0)
    
# Construção da tabela de probabilidade do Naïve Bayes
    
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB() 
classificador.fit(previsores_treinamento, classe_treinamento) 

# Resultado da predição- O que o algoritmo Naïve Bayes entende como resultado da predição
previsoes = classificador.predict(previsores_teste)
    
# Verificando percentual de acerto do Naïve Bayes

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)

# precisao * 100 : Acerto de 93,8%



