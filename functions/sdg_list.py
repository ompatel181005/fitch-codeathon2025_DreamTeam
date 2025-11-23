import pandas as pd

test_file = pd.read_csv("data/test.csv")

sdg_file = pd.read_csv("data/sustainable_development_goals.csv")

print("***********\n List of Entities and the goals they have publicly committed to:\n")


entity_dict = sdg_file.groupby('entity_id')['sdg_id'].apply(set).to_dict()
for entity_id, sdg_set in list(entity_dict.items()):
    print(f"{entity_id} : {sdg_set}")
print(f"\nNumber of Entities: {len(entity_dict)}")

test_id = set(test_file['entity_id'])
sdg_id = set(sdg_file['entity_id'])

common_entities = test_id.intersection(sdg_id)

print(f"Common Entities in test and sdg file: {common_entities}")
print(f"Number of Common Entities {len(common_entities)}")


print("\n\nSDG Scores:")

sdgScore = pd.DataFrame()

sdgScore = test_file[['entity_id']].copy()

sdgScore["SDG_Score"] = 0

pd.set_option('display.max_rows', None)

sdgScore = sdg_file.groupby('entity_id')['entity_id'].size().reset_index(name='SDG_Score')

result = test_file.merge(sdgScore, on='entity_id', how='left')

result['SDG_Score'] = result['SDG_Score'].fillna(0).astype(int)

print(result[['entity_id','SDG_Score']]) 

result.to_csv('sdg_scores.csv',index=False)
