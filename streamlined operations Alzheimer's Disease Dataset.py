
#%%
import pandas as pd
import numpy as np
from pycaret.classification import *
sns.set(style = 'whitegrid')
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv(r"C:\Users\phali\Downloads\archive\alzheimers_disease_data.csv")
df.head()

#%%
df.drop(columns = ['DoctorInCharge' , 'PatientID' ] , inplace = True )

#%%
df.shape

#%%
df.info()

#%%
df.describe
# %%
import pandas as pd
import numpy as np
from pycaret.classification import *

my_exp = setup(data = df,
               target = 'Diagnosis',
               train_size= 0.7,
               preprocess= True,
               bin_numeric_features= ['Age' ,
                                      'AlcoholConsumption',
                                      'PhysicalActivity' ,
                                      'DietQuality'] , 
               outliers_method= 'iforest',
               outliers_threshold= 0.1 ,
               fix_imbalance= True,
               fix_imbalance_method= 'SMOTE',
               transformation= True,
               transformation_method= 'yeo-johnson',
               normalize= True,
               normalize_method= 'zscore',
               remove_multicollinearity=True,
               multicollinearity_threshold= 0.9,
               feature_selection= True,
               feature_selection_method= 'classic',
               feature_selection_estimator= 'catboost',
               n_features_to_select= 18,
               fold_strategy= 'kfold',
               fold= 5,
               n_jobs= -1,
               experiment_name= 'my_exp_001')

best = compare_models()

#%%
best_3 = compare_models(n_select= 3)
# %%
plot_model(best , plot = 'pr')
# %%
plot_model(best , plot = 'feature')

#%%
plot_model(best , plot = 'auc')

#%%
plot_model(best , plot = 'pipeline')
# %%
plot_model(best , plot = 'confusion_matrix')
# %%
plot_model(best , plot = 'threshold')
#%%
plot_model(best , plot = 'vc')
# %%
