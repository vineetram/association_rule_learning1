#APRIORI

#IMPORTING THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#IMPORTING THE DATASET
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#TRAINIG APRIORI ON THE DATASET
from apyori import apriori
rules = apriori(transactions, min_support = 0.003,  min_confidence = 0.2, min_lift = 3 , min_length = 2)

#VISUALIZING THE RESULTS
results = list(rules)

 