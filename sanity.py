import pandas as pd
import numpy as np
from scipy.stats import chi2

df = pd.read_csv('_outputs/validation_merged_patient_level.csv', dtype=str)

def to_bin(s):
    s = s.fillna('').astype(str).str.strip().str.lower()
    return s.map(lambda x: 1 if x in ['1','true','yes'] else (0 if x in ['0','false','no'] else np.nan))

outcomes = {
    'Stage1_MinorComp': ('Stage1_MinorComp_pred', 'Stage1_MinorComp_gold'),
    'Stage1_MajorComp': ('Stage1_MajorComp_pred', 'Stage1_MajorComp_gold'),
    'Stage1_Reoperation': ('Stage1_Reoperation_pred', 'Stage1_Reoperation_gold'),
    'Stage1_Rehospitalization': ('Stage1_Rehospitalization_pred', 'Stage1_Rehospitalization_gold'),
    'Stage1_Failure': ('Stage1_Failure_pred', 'Stage1_Failure_gold'),
}

# Derive AnyComp
df['AnyComp_pred'] = ((to_bin(df['Stage1_MinorComp_pred'])==1) | (to_bin(df['Stage1_MajorComp_pred'])==1)).astype(int)
df['AnyComp_gold'] = ((to_bin(df['Stage1_MinorComp_gold'])==1) | (to_bin(df['Stage1_MajorComp_gold'])==1)).astype(int)
outcomes['AnyComp'] = ('AnyComp_pred', 'AnyComp_gold')

def wilson_ci(p, n, z=1.96):
    if n == 0: return (0,0)
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return (round(center-margin,3), round(center+margin,3))

def kappa(p, g):
    mask = p.notna() & g.notna()
    p, g = p[mask], g[mask]
    n = len(p)
    if n == 0: return np.nan
    po = (p == g).sum() / n
    pe = ((p==1).sum()/n * (g==1).sum()/n) + ((p==0).sum()/n * (g==0).sum()/n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else np.nan

def mcnemar(p, g):
    mask = p.notna() & g.notna()
    p, g = p[mask].astype(int), g[mask].astype(int)
    b = ((p==1) & (g==0)).sum()
    c = ((p==0) & (g==1)).sum()
    if (b+c) == 0: return 1.0
    chi2_stat = (abs(b-c)-1)**2 / (b+c)
    from scipy.stats import chi2 as chi2dist
    return round(chi2dist.sf(chi2_stat, 1), 4)

print('=== TABLE 2 + TABLE 3 INPUTS ===')
print()
for name, (pc, gc) in outcomes.items():
    p = to_bin(df[pc]) if 'AnyComp' not in pc else df[pc].apply(lambda x: float(x) if str(x) not in ['','nan'] else np.nan)
    g = to_bin(df[gc]) if 'AnyComp' not in gc else df[gc].apply(lambda x: float(x) if str(x) not in ['','nan'] else np.nan)
    mask = g.notna()
    p2, g2 = p[mask], g[mask]
    n = len(g2)
    tp = ((p2==1)&(g2==1)).sum()
    fp = ((p2==1)&(g2==0)).sum()
    fn = ((p2==0)&(g2==1)).sum()
    tn = ((p2==0)&(g2==0)).sum()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    ppv  = tp/(tp+fp) if (tp+fp)>0 else 0
    npv  = tn/(tn+fn) if (tn+fn)>0 else 0
    f1   = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
    manual_rate = g2.sum()/n
    nlp_rate = p2.sum()/n
    k = kappa(p2, g2)
    mc = mcnemar(p2, g2)
    print(f'{name}: n={n}, TP={int(tp)}, FP={int(fp)}, FN={int(fn)}, TN={int(tn)}')
    print(f'  Manual rate={round(manual_rate,3)} NLP rate={round(nlp_rate,3)} Kappa={k} McNemar_p={mc}')
    print(f'  Sens={round(sens,3)} {wilson_ci(sens,tp+fn)} Spec={round(spec,3)} {wilson_ci(spec,tn+fp)}')
    print(f'  PPV={round(ppv,3)} {wilson_ci(ppv,tp+fp)} NPV={round(npv,3)} {wilson_ci(npv,tn+fn)} F1={round(f1,3)}')
    print()
