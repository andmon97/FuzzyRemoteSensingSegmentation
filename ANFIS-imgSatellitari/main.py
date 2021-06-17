import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from training import train_model
from test import test_model
import anfis
from membership import *
from utility import load_model, plot_import, split_dataset

dataset = 'datasets/reducedV3.csv'
model = None
n_terms = 2 #IMPOSTA NUMERO DI FUZZY SET
batch_size = 32
num_categories = 4 #IMPOSTA NUMERO DI CLASSI
epoch = 100
model_l = False #SEMPRE A FALSE
hybrid = False #SEMPRE A FALSE
i = 0
lista_acc = []
k_fold = False

# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'GaussMembFunc',
                           'TriangularMembFunc',
                           'TrapezoidalMembFunc',
                           ]}

# DEFINIRE LA MEMBERSHIP FUNCTION ATTUALMENTE E' ABILITATA SOLO GAUSSIANA E TRIANGOLARE
membership_function = get_class_for['GaussMembFunc']

d_data, d_target = split_dataset('datasets/reducedV3.csv')


# Split train into trainval-test
X_trainval, X_test, y_trainval, y_test = train_test_split(d_data, d_target, test_size=0.3, shuffle=True,
                                                          stratify=d_target, random_state=42)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, shuffle=True,
                                               stratify=y_trainval, random_state=42)


model = train_model(model, X_train, y_train, X_val, y_val, n_terms, num_categories,
                                   batch_size, epoch, model_l, hybrid, membership_function, i)

torch.save(model, 'models/G_model_geo_' + str(i) + '.h5')

model = torch.load('models/G_model_geo_0.h5')

test_model(model, X_test, y_test, num_categories, k_fold, i, lista_acc)

model = torch.load('models/G_model_geo_0.h5')

X_test = torch.Tensor(X_test)

y_pred = model(X_test)

cat_act = torch.argmax(y_pred, dim=1)

#PREDIZIONE SUI DATI DI TEST
print(cat_act)

cm = confusion_matrix(y_test, cat_act)
print(cm)

cl = classification_report(y_test, cat_act)
print(cl)



