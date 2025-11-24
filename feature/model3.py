### setup

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sdg = pd.read_csv('./data/sustainable_development_goals.csv')
rev = pd.read_csv('./data/revenue_distribution_by_sector.csv')
train_raw = pd.read_csv('./data/train.csv')
test_raw = pd.read_csv('./data/test.csv')

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

## adding sdgs (7, 12, 13)

# sdg_n = sdg[sdg['sdg_id'].isin([7, 12, 13])]
# sdg_n = pd.DataFrame(sdg_n.groupby('entity_id').size())
# sdg_n.columns = ['n']
# sdg_n = sdg_n.reset_index()

# sdg_2 = train_raw[train_raw['entity_id'].isin(sdg_n[sdg_n['n'] == 2]['entity_id'])]
# sdg_1 = train_raw[train_raw['entity_id'].isin(sdg_n[sdg_n['n'] == 1]['entity_id'])]
# sdg_0 = train_raw[~train_raw['entity_id'].isin(sdg_n['entity_id'])]

# sdg_2['scope_1_per_usd'] = sdg_2['target_scope_1'] / sdg_2['revenue']
# sdg_1['scope_1_per_usd'] = sdg_1['target_scope_1'] / sdg_1['revenue']   
# sdg_0['scope_1_per_usd'] = sdg_0['target_scope_1'] / sdg_0['revenue']
# sdg_2['scope_2_per_usd'] = sdg_2['target_scope_2'] / sdg_2['revenue']
# sdg_1['scope_2_per_usd'] = sdg_1['target_scope_2'] / sdg_1['revenue']
# sdg_0['scope_2_per_usd'] = sdg_0['target_scope_2'] / sdg_0['revenue']

# sdg_2 = sdg_2.drop(['target_scope_1', 'target_scope_2'], axis=1)
# sdg_1 = sdg_1.drop(['target_scope_1', 'target_scope_2'], axis=1)
# sdg_0 = sdg_0.drop(['target_scope_1', 'target_scope_2'], axis=1)

# sdg_0_coeff_1 = np.mean(sdg_0['scope_1_per_usd'])
# sdg_1_coeff_1 = np.mean(sdg_1['scope_1_per_usd']) / sdg_0_coeff_1
# sdg_2_coeff_1 = np.mean(sdg_2['scope_1_per_usd']) / sdg_0_coeff_1

# sdg_0_coeff_2 = np.mean(sdg_0['scope_2_per_usd'])
# sdg_1_coeff_2 = np.mean(sdg_1['scope_2_per_usd']) / sdg_0_coeff_2
# sdg_2_coeff_2 = np.mean(sdg_2['scope_2_per_usd']) / sdg_0_coeff_2

# sdg_coeff_dict = {1: (sdg_1_coeff_1, sdg_1_coeff_2),
#                   2: (sdg_2_coeff_1, sdg_2_coeff_2),
#                   0: (1, 1)}

# def apply_sdg_coeff(df):
#     df['n'] = df['entity_id'].map(sdg_n.set_index('entity_id')['n']).fillna(0)
#     df['n'] = pd.to_numeric(df['n'], downcast='integer')
#     df['coeff'] = df['n'].map(sdg_coeff_dict)
#     df['scope_1_coeff'] = [t[0] for t in df['coeff']]
#     df['scope_2_coeff'] = [t[1] for t in df['coeff']]
#     df = df.drop(['n', 'coeff'], axis=1)
#     df['target_scope_1'] = df['target_scope_1'] * df['scope_1_coeff']
#     df['target_scope_2'] = df['target_scope_2'] * df['scope_2_coeff']
#     df = df.drop(['scope_1_coeff', 'scope_2_coeff'], axis=1)
#     return df

## nation?

# country_n = pd.DataFrame(train_raw.groupby('country_code').size())
# country_n.columns = ['n']
# country_n = country_n[country_n['n'] > 1].reset_index()

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

train_estimate1 = apply_sdg_coeff(train_estimate1)

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

train_estimate2 = apply_sdg_coeff(train_estimate2)

mae2_scope1 = mean_absolute_error(train_raw.sort_values(by='entity_id')[['target_scope_1']], train_estimate2.sort_values(by='entity_id')[['target_scope_1']])
mae2_scope2 = mean_absolute_error(train_raw.sort_values(by='entity_id')[['target_scope_2']], train_estimate2.sort_values(by='entity_id')[['target_scope_2']])

### testing with test dataset

test_estimate = test_raw
test_estimate = pd.merge(test_estimate,
                 rev[['entity_id', 'nace_level_2_code', 'revenue_pct']],
                 how='inner', on='entity_id')
test_estimate['revenue_pct_new'] = test_estimate.groupby(['entity_id', 'nace_level_2_code'])['revenue_pct'].transform('sum')
test_estimate = test_estimate.drop('revenue_pct', axis=1).drop_duplicates(keep='first')

test_estimate = test_estimate.merge(coeff2[['nace_level_2_code', 'coefficient_scope_1']],
                                    on='nace_level_2_code',
                                    how='left')

test_estimate = test_estimate.merge(coeff2[['nace_level_2_code', 'coefficient_scope_2']],
                                    on='nace_level_2_code',
                                    how='left')

test_estimate['target_scope_1'] = test_estimate['revenue_pct_new'] * test_estimate['revenue'] * test_estimate['coefficient_scope_1']
test_estimate['target_scope_2'] = test_estimate['revenue_pct_new'] * test_estimate['revenue'] * test_estimate['coefficient_scope_2']

test_estimate['target_scope_1'] = test_estimate['target_scope_1'].where(test_estimate['target_scope_1'] >= 0, 0)
test_estimate['target_scope_2'] = test_estimate['target_scope_2'].where(test_estimate['target_scope_2'] >= 0, 0)

test_estimate = test_estimate.drop(['nace_level_2_code', 'revenue_pct_new', 'coefficient_scope_1', 'coefficient_scope_2'], axis=1)
test_estimate['target_scope_1'] = test_estimate.groupby('entity_id')['target_scope_1'].transform('sum')
test_estimate['target_scope_2'] = test_estimate.groupby('entity_id')['target_scope_2'].transform('sum')
test_estimate = test_estimate.drop_duplicates(keep='first')

test_estimate[['entity_id', 'target_scope_1', 'target_scope_2']].to_csv('./feature/test_estimate.csv', index=False)