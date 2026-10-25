"""Recompute the Field Note #06 aggregates from the public per-frame CSV files.

Usage: python summarize_from_csv.py data/stereo_rig_frames.csv data/phone_frames.csv [results_fn06.json]
If a reference results file is given, every recomputed value is compared with it.
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def bins(df, col, edges):
    g = df.groupby(pd.cut(df[col], edges), observed=True)
    return [dict(lo=float(i.left), hi=float(i.right), n=int(len(x)), rel_sh=float(x.rel_sharpness.median()),
                 q25=float(x.rel_sharpness.quantile(.25)), q75=float(x.rel_sharpness.quantile(.75)),
                 rel_fast=float(x.rel_fast_corners.median()), w=float(x.rotation_dps.median())) for i, x in g]


def rules(score, y, thresholds):
    return {str(t): dict(flag=float((score > t).mean()),
                         recall=float(((score > t) & (y == 1)).sum() / max(1, y.sum())),
                         precision=float(((score > t) & (y == 1)).sum() / max(1, (score > t).sum()))) for t in thresholds}


S = pd.read_csv(sys.argv[1]); P = pd.read_csv(sys.argv[2]); R = {}
ok = S.rel_sharpness.notna(); y = (S.rel_sharpness[ok] < 0.6).astype(int)
R['stereo'] = dict(frames=len(S),
    exposure_ms_p1_p50_p99=[float(x) for x in np.percentile(S.exposure_ms, [1, 50, 99])],
    spearman_blur_rel=float(spearmanr(S.blur_px_pred, S.rel_sharpness, nan_policy='omit').statistic),
    blur_bins=bins(S, 'blur_px_pred', [0, .25, .5, .75, 1, 1.5, 2, 3, 6]),
    rot_bins=bins(S, 'rotation_dps', [0, 5, 10, 20, 40, 80, 250]),
    blur_frac={str(t): float((S.blur_px_pred > t).mean()) for t in (.5, 1, 2, 3)},
    rot_frac={str(t): float((S.rotation_dps > t).mean()) for t in (10, 20, 40, 80)},
    rot_pct_50_90_95_99=[float(x) for x in np.percentile(S.rotation_dps, [50, 90, 95, 99])],
    auc_flag_rel_lt_0p6=float(roc_auc_score(y, S.blur_px_pred[ok])),
    rule=rules(S.blur_px_pred[ok], y, (1.0, 1.5, 2.0)))
for name, d in P.groupby('recording', sort=False):
    v = d.dropna(subset=['rel_sharpness', 'rotation_dps']); yy = (v.rel_sharpness < 0.6).astype(int)
    sev = v.rel_sharpness < 0.25; nt = v.non_task == 1
    R[name] = dict(frames=len(d), rot_bins=bins(d, 'rotation_dps', [0, 5, 10, 20, 40, 80, 250]),
        rot_pct_50_90_95_99=[float(x) for x in np.nanpercentile(d.rotation_dps, [50, 90, 95, 99])],
        rot_frac={str(t): float((d.rotation_dps > t).mean()) for t in (10, 20, 40, 80)},
        spearman_w_rel=float(spearmanr(d.rotation_dps, d.rel_sharpness, nan_policy='omit').statistic),
        auc_flag_rel_lt_0p6=float(roc_auc_score(yy, v.rotation_dps)), rule=rules(v.rotation_dps, yy, (20, 40)),
        frames_without_reference=int(d.rel_sharpness.isna().sum()),
        nontask=dict(frames_nt=float(nt.mean()), severe=int(sev.sum()), severe_in_nt=float((sev & nt).sum() / max(1, sev.sum())),
                     rel_nt=float(v.rel_sharpness[nt].median()), rel_task=float(v.rel_sharpness[~nt].median())))
json.dump(R, open('summary_from_csv.json', 'w'), indent=1)
print('stereo  spearman %.4f  auc %.4f' % (R['stereo']['spearman_blur_rel'], R['stereo']['auc_flag_rel_lt_0p6']))
for k in ('phone_knife_prep', 'phone_seasoning'):
    print(k, 'auc %.4f' % R[k]['auc_flag_rel_lt_0p6'], 'frames>20dps %.4f' % R[k]['rot_frac']['20'])

if len(sys.argv) > 3:
    ref = json.load(open(sys.argv[3])); bad = []; n = 0
    def walk(a, b, path):
        global n
        if isinstance(a, dict):
            for k in a:
                if k in b: walk(a[k], b[k], path + '.' + k)
        elif isinstance(a, list):
            for i, (x, z) in enumerate(zip(a, b)): walk(x, z, f'{path}[{i}]')
        elif isinstance(a, (int, float)):
            n += 1
            if abs(a - b) > 1e-3 * max(1, abs(b)): bad.append((path, a, b))
    walk(R, ref, '')
    print(f'compared {n} values with reference, mismatches: {len(bad)}')
    for x in bad[:20]: print(x)
