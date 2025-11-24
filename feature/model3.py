### setup

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

rev = pd.read_csv('./data/revenue_distribution_by_sector.csv')
train_raw = pd.read_csv('./data/train.csv')

### model 1 - nace level 1

train1 = train_raw
train1 = pd.merge(train1,
                 rev[['entity_id', 'nace_level_1_code', 'revenue_pct']],
                 how='inner', on='entity_id')
train1['revenue_pct_new'] = train1.groupby(['entity_id', 'nace_level_1_code'])['revenue_pct'].transform('sum')
train1 = train1.drop('revenue_pct', axis=1).drop_duplicates(keep='first')
train1['revenue_in_nace'] = train1['revenue'] * train1['revenue_pct_new']

X1 = train1.pivot_table(index='entity_id', columns='nace_level_1_code', values='revenue_in_nace', aggfunc='first').sort_values(by='entity_id').fillna(0).to_numpy()
y1_scope1 = train_raw.sort_values(by='entity_id')[['target_scope_1']]
y1_scope2 = train_raw.sort_values(by='entity_id')[['target_scope_2']]

model1_scope1 = LinearRegression(fit_intercept=False)
model1_scope1.fit(X1, y1_scope1)

model1_scope2 = LinearRegression(fit_intercept=False)
model1_scope2.fit(X1, y1_scope2)

coeff1_scope1 = model1_scope1.coef_
coeff1_scope2 = model1_scope2.coef_

coeff1_scope1_df = pd.DataFrame(data={'nace_level_1_code':list(train1.pivot_table(index='entity_id', columns='nace_level_1_code', values='revenue_in_nace', aggfunc='first').columns), 'coefficient':coeff1_scope1[0]})
coeff1_scope2_df = pd.DataFrame(data={'nace_level_1_code':list(train1.pivot_table(index='entity_id', columns='nace_level_1_code', values='revenue_in_nace', aggfunc='first').columns), 'coefficient':coeff1_scope2[0]})

coeff1 = pd.DataFrame(data={'nace_level_1_code':coeff1_scope1_df['nace_level_1_code'],
                            'coefficient_scope_1':coeff1_scope1_df['coefficient'],
                           'coefficient_scope_2':coeff1_scope2_df['coefficient']})

coeff1.to_csv('./feature/nace1_coeff.csv', index=False)

### model 2 - nace level 2

train2 = train_raw
train2 = pd.merge(train2,
                 rev[['entity_id', 'nace_level_2_code', 'revenue_pct']],
                 how='inner', on='entity_id')
train2['revenue_pct_new'] = train2.groupby(['entity_id', 'nace_level_2_code'])['revenue_pct'].transform('sum')
train2 = train2.drop('revenue_pct', axis=1).drop_duplicates(keep='first')
train2['revenue_in_nace'] = train2['revenue'] * train2['revenue_pct_new']

X2 = train2.pivot_table(index='entity_id', columns='nace_level_2_code', values='revenue_in_nace', aggfunc='first').sort_values(by='entity_id').fillna(0).to_numpy()
y2_scope1 = train_raw.sort_values(by='entity_id')[['target_scope_1']]
y2_scope2 = train_raw.sort_values(by='entity_id')[['target_scope_2']]

model2_scope1 = LinearRegression(fit_intercept=False)
model2_scope1.fit(X2, y2_scope1)

model2_scope2 = LinearRegression(fit_intercept=False)
model2_scope2.fit(X2, y2_scope2)

coeff2_scope1 = model2_scope1.coef_
coeff2_scope2 = model2_scope2.coef_

coeff2_scope1_df = pd.DataFrame(data={'nace_level_2_code':list(train2.pivot_table(index='entity_id', columns='nace_level_2_code', values='revenue_in_nace', aggfunc='first').columns), 'coefficient':coeff2_scope1[0]})
coeff2_scope2_df = pd.DataFrame(data={'nace_level_2_code':list(train2.pivot_table(index='entity_id', columns='nace_level_2_code', values='revenue_in_nace', aggfunc='first').columns), 'coefficient':coeff2_scope2[0]})

coeff2 = pd.DataFrame(data={'nace_level_2_code':coeff2_scope1_df['nace_level_2_code'],
                            'coefficient_scope_1':coeff2_scope1_df['coefficient'],
                           'coefficient_scope_2':coeff2_scope2_df['coefficient']})

coeff2.to_csv('./feature/nace2_coeff.csv', index=False)

## testing - nace level 1

train_estimate1 = train_raw.drop(['target_scope_1', 'target_scope_2'], axis=1)
train_estimate1 = pd.merge(train_estimate1,
                 rev[['entity_id', 'nace_level_1_code', 'revenue_pct']],
                 how='inner', on='entity_id')
train_estimate1['revenue_pct_new'] = train_estimate1.groupby(['entity_id', 'nace_level_1_code'])['revenue_pct'].transform('sum')
train_estimate1 = train_estimate1.drop('revenue_pct', axis=1).drop_duplicates(keep='first')

train_estimate1 = train_estimate1.merge(coeff1[['nace_level_1_code', 'coefficient_scope_1']],
                                      on='nace_level_1_code',
                                      how='left')

train_estimate1 = train_estimate1.merge(coeff1[['nace_level_1_code', 'coefficient_scope_2']],
                                      on='nace_level_1_code',
                                      how='left')

train_estimate1['target_scope_1'] = train_estimate1['revenue_pct_new'] * train_estimate1['revenue'] * train_estimate1['coefficient_scope_1']
train_estimate1['target_scope_2'] = train_estimate1['revenue_pct_new'] * train_estimate1['revenue'] * train_estimate1['coefficient_scope_2']

train_estimate1['target_scope_1'] = train_estimate1['target_scope_1'].where(train_estimate1['target_scope_1'] >= 0, 0)
train_estimate1['target_scope_2'] = train_estimate1['target_scope_2'].where(train_estimate1['target_scope_2'] >= 0, 0)

train_estimate1 = train_estimate1.drop(['nace_level_1_code', 'revenue_pct_new', 'coefficient_scope_1', 'coefficient_scope_2'], axis=1)
train_estimate1['target_scope_1'] = train_estimate1.groupby('entity_id')['target_scope_1'].transform('sum')
train_estimate1['target_scope_2'] = train_estimate1.groupby('entity_id')['target_scope_2'].transform('sum')
train_estimate1 = train_estimate1.drop_duplicates(keep='first')

mae1_scope1 = mean_absolute_error(train_raw.sort_values(by='entity_id')[['target_scope_1']], train_estimate1.sort_values(by='entity_id')[['target_scope_1']])
mae1_scope2 = mean_absolute_error(train_raw.sort_values(by='entity_id')[['target_scope_2']], train_estimate1.sort_values(by='entity_id')[['target_scope_2']])

### testing - nace level 2

train_estimate2 = train_raw.drop(['target_scope_1', 'target_scope_2'], axis=1)
train_estimate2 = pd.merge(train_estimate2,
                 rev[['entity_id', 'nace_level_2_code', 'revenue_pct']],
                 how='inner', on='entity_id')
train_estimate2['revenue_pct_new'] = train_estimate2.groupby(['entity_id', 'nace_level_2_code'])['revenue_pct'].transform('sum')
train_estimate2 = train_estimate2.drop('revenue_pct', axis=1).drop_duplicates(keep='first')

train_estimate2 = train_estimate2.merge(coeff2[['nace_level_2_code', 'coefficient_scope_1']],
                                      on='nace_level_2_code',
                                      how='left')

train_estimate2 = train_estimate2.merge(coeff2[['nace_level_2_code', 'coefficient_scope_2']],
                                      on='nace_level_2_code',
                                      how='left')

train_estimate2['target_scope_1'] = train_estimate2['revenue_pct_new'] * train_estimate2['revenue'] * train_estimate2['coefficient_scope_1']
train_estimate2['target_scope_2'] = train_estimate2['revenue_pct_new'] * train_estimate2['revenue'] * train_estimate2['coefficient_scope_2']

train_estimate2['target_scope_1'] = train_estimate2['target_scope_1'].where(train_estimate2['target_scope_1'] >= 0, 0)
train_estimate2['target_scope_2'] = train_estimate2['target_scope_2'].where(train_estimate2['target_scope_2'] >= 0, 0)

train_estimate2 = train_estimate2.drop(['nace_level_2_code', 'revenue_pct_new', 'coefficient_scope_1', 'coefficient_scope_2'], axis=1)
train_estimate2['target_scope_1'] = train_estimate2.groupby('entity_id')['target_scope_1'].transform('sum')
train_estimate2['target_scope_2'] = train_estimate2.groupby('entity_id')['target_scope_2'].transform('sum')
train_estimate2 = train_estimate2.drop_duplicates(keep='first')

mae2_scope1 = mean_absolute_error(train_raw.sort_values(by='entity_id')[['target_scope_1']], train_estimate2.sort_values(by='entity_id')[['target_scope_1']])
mae2_scope2 = mean_absolute_error(train_raw.sort_values(by='entity_id')[['target_scope_2']], train_estimate2.sort_values(by='entity_id')[['target_scope_2']])