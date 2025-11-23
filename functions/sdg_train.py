# find avg emissions of train.csv entities given SDG score

import pandas as pd

train_file = pd.read_csv("data/train.csv")
test_file = pd.read_csv("data/test.csv")
sdg_file = pd.read_csv("data/sustainable_development_goals.csv")
rev_file = pd.read_csv("data/revenue_distribution_by_sector.csv")

test_id = set(test_file['entity_id'])
train_id = set(train_file['entity_id'])
sdg_id = set(sdg_file['entity_id'])

common_entities_1 = train_id.intersection(test_id)
common_entities_2 = train_id.intersection(sdg_id)

# confirm train and test do not have common entities
print(len(common_entities_1)) # expect 0 (true)

# confirm train and sdg have common entities
print(len(common_entities_2)) # 118 common entities

# find sdg score of train entities
sdgScore = pd.DataFrame()

# make a new df with train entity ids
sdgScore = train_file[['entity_id']].copy()

# initialize a new column to all zeroes
sdgScore["SDG_Score"] = 0

#...............debug....................
# pd.set_option('display.max_rows', None)

sdgScore = sdg_file.groupby('entity_id')['entity_id'].size().reset_index(name='SDG_Score')
train_sdg = train_file.merge(sdgScore, on='entity_id', how='left')

# print(sdgScore)
# print(result)

train_sdg['SDG_Score'] = train_sdg['SDG_Score'].fillna(0).astype(int)
train_sdg = train_sdg.sort_values(by='SDG_Score',ascending=1)

print(train_sdg[['entity_id','SDG_Score']]) # 0 to 3

# confirm train and rev have common entities
rev_id = set(rev_file['entity_id'])
common_entities_3 = train_id.intersection(rev_id)


# make two new columns:
# emissions (1 and 2) per dollar

train_sdg['s1_per_dollar'] = 0
train_sdg['s2_per_dollar'] = 0

train_sdg['s1_per_dollar'] = train_sdg['target_scope_1'] / train_sdg['revenue']
train_sdg['s2_per_dollar'] = train_sdg['target_scope_2'] / train_sdg['revenue']

avg_emissions_sdg = train_sdg.groupby('SDG_Score')[['s1_per_dollar','s2_per_dollar']].mean()
print(avg_emissions_sdg)

avg_emissions_sdg.to_csv('avg_emissions_sdg.csv',index=False)

#debug
# print(train_sdg[['entity_id','SDG_Score',
#                  'revenue','target_scope_1',
#                  'target_scope_2','s1_per_dollar',
#                  's2_per_dollar']])

# individual
# train_sdg['avg1'] = train_sdg.groupby('SDG_Score')['s1_per_dollar'].transform('mean')
# train_sdg['avg2'] = train_sdg.groupby('SDG_Score')['s2_per_dollar'].transform('mean')
# print(train_sdg[['entity_id','SDG_Score','avg1','avg2']])

