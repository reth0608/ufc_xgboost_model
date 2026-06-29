import pandas as pd
df = pd.read_csv('data/fights_features.csv')

# Does age_diff correlate with experience diff?
print(df[['age_diff', 'a_fights_in_ufc', 'b_fights_in_ufc']].corr())

# Distribution — is it extreme values driving importance?
print(df['age_diff'].describe())
print(f"Fights where |age_diff| > 10 years: {(df['age_diff'].abs() > 10).sum()}")