import pandas as pd
df = pd.read_csv('_outputs/pbs_only_evidence.csv', dtype=str, engine='python', encoding='latin-1')
print('Evidence rows:', len(df))
print('Fields:', df['FIELD'].value_counts().to_dict() if 'FIELD' in df.columns else 'no FIELD col')
print('Rule decisions:', df['RULE_DECISION'].value_counts().to_dict() if 'RULE_DECISION' in df.columns else 'no RULE_DECISION col')
