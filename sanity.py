import pandas as pd
import numpy as np

df = pd.read_csv('_outputs/validation_merged_patient_level.csv', dtype=str)

def to_bin(s):
    s = s.fillna('').astype(str).str.strip().str.lower()
    return s.map(lambda x: 1 if x in ['1','true','yes'] else (0 if x in ['0','false','no'] else np.nan))

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
    po = float((p == g).sum()) / n
    pe = ((p==1).sum()/n * (g==1).sum()/n) + ((p==0).sum()/n * (g==0).sum()/n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else np.nan

def mcnemar_p(p, g):
    mask = p.notna() & g.notna()
    p2 = p[mask].astype(int)
    g2 = g[mask].astype(int)
    b = int(((p2==1) & (g2==0)).sum())
    c = int(((p2==0) & (g2==1)).sum())
    if (b+c) == 0: return 1.0
    # chi2 with continuity correction, 1 df
    chi2_stat = float((abs(b-c)-1)**2) / float(b+c)
    # p-value from chi2 CDF approximation (no scipy)
    # use incomplete gamma approximation
    import math
    k = 0.5  # df/2
    x = chi2_stat / 2.0
    # regularized upper incomplete gamma via series
    def upper_inc_gamma_half(x):
        # P(chi2 > chi2_stat) for df=1 = erfc(sqrt(chi2_stat/2))
        return math.erfc(math.sqrt(x))
    return round(upper_inc_gamma_half(chi2_stat/2.0), 4)

# Derive AnyComp from gold and pred
mc_pred = to_bin(df['Stage1_MinorComp_pred'])
mc_gold = to_bin(df['Stage1_MinorComp_gold'])
mj_pred = to_bin(df['Stage1_MajorComp_pred'])
mj_gold = to_bin(df['Stage1_MajorComp_gold'])

df['AnyComp_pred'] = ((mc_pred==1)|(mj_pred==1)).astype(float)
df['AnyComp_gold'] = ((mc_gold==1)|(mj_gold==1)).astype(float)

# Only evaluate where gold outcome data exists (stage1 labeled)
stage1_mask = mc_gold.notna()

outcomes = [
    ('AnyComp',              'AnyComp_pred',                    'AnyComp_gold'),
    ('MajorComp',            'Stage1_MajorComp_pred',           'Stage1_MajorComp_gold'),
    ('Reoperation',          'Stage1_Reoperation_pred',         'Stage1_Reoperation_gold'),
    ('Rehospitalization',    'Stage1_Rehospitalization_pred',   'Stage1_Rehospitalization_gold'),
    ('Failure',              'Stage1_Failure_pred',             'Stage1_Failure_gold'),
]

print('=== TABLE 2 + TABLE 3 ===')
print()
for name, pc, gc in outcomes:
    p = to_bin(df[pc]) if 'AnyComp' not in pc else df[pc]
    g = to_bin(df[gc]) if 'AnyComp' not in gc else df[gc]
    mask = g.notna() & stage1_mask
    p2 = p[mask]
    g2 = g[mask]
    n = int(len(g2))
    tp = int(((p2==1)&(g2==1)).sum())
    fp = int(((p2==1)&(g2==0)).sum())
    fn = int(((p2==0)&(g2==1)).sum())
    tn = int(((p2==0)&(g2==0)).sum())
    sens = tp/float(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/float(tn+fp) if (tn+fp)>0 else 0.0
    ppv  = tp/float(tp+fp) if (tp+fp)>0 else 0.0
    npv  = tn/float(tn+fn) if (tn+fn)>0 else 0.0
    f1   = 2*tp/float(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0.0
    manual_rate = float(g2.sum())/n
    nlp_rate    = float(p2.sum())/n
    k  = kappa(p2, g2)
    mc = mcnemar_p(p2, g2)
    print('{}: n={} TP={} FP={} FN={} TN={}'.format(name,n,tp,fp,fn,tn))
    print('  Manual_rate={} NLP_rate={} Kappa={} McNemar_p={}'.format(
        round(manual_rate,3), round(nlp_rate,3), k, mc))
    print('  Sens={} CI={} Spec={} CI={}'.format(
        round(sens,3), wilson_ci(sens,tp+fn),
        round(spec,3), wilson_ci(spec,tn+fp)))
    print('  PPV={} CI={} NPV={} CI={} F1={}'.format(
        round(ppv,3), wilson_ci(ppv,tp+fp),
        round(npv,3), wilson_ci(npv,tn+fn),
        round(f1,3)))
    print()
