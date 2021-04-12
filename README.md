# Base_Credito_Classificacao

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
matriz = confusion_matrix(classe_teste, previsoes )

# precisao * 100 : Acerto de 93,8%

''' Análise de matriz (confusion_matrix)

Linha 0 , coluna 0 - Quantos registros o algoritmo classificou como pessoas que não pagaram o empréstimo (428)

Linha 0, coluna 1 - Registros dos clientes que não pagaram,mas que o algoritmo classificou como pagadores(8)

Linha 1, coluna 0 - Registros dos clientes que pagaram ,porém o algoritmo os classificou como não pagadores (23)

Linha 1, coluna 1 - 41 registros foram classificados como pagadores e o algoritmo os classificou como pagadores

Para saber exatamente quantos registros foram classificados corretamente soma -se 428 + 41 = 469

Para saber exatamente quantos registros foram classificados com erro soma-se 23 + 8 = 31

E para conferir se todos os valores analisados batem com o total (500) basta somar todos os valores da tabela matriz
428 + 8 + 23 + 41 = 500

'''



