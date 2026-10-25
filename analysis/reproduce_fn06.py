"""Field Note #06 — end-to-end reproduction from source recordings.

Recomputes, from the original video, timestamp and IMU files:
  * per-frame sharpness (Laplacian variance) and FAST corner counts on the image centre
  * gyro-predicted blur for the stereo rig (gyro→image map fitted to optical flow)
  * camera rotation rate per frame for two phone recordings
  * relative sharpness against nearby low-motion frames
  * every aggregate number quoted in the field note

Usage:  python reproduce_fn06.py config.json OUTPUT_DIR
The config lists input paths (see config.example.json). Source recordings are NOT public.
"""
import sys, os, json, subprocess
import numpy as np, pandas as pd, cv2
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

CFG = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
FAST = cv2.FastFeatureDetector_create(threshold=20)
CAM_IMU_OFFSET_US = -1100.0   # stereo rig, t_imu = t_cam + offset (Field Note #03)
REL_WIN, REL_MIN = 90, 6      # ±90 frames (±3 s) reference window, ≥6 reference frames


def frames_gray(path, W, H, vf=None):
    cmd = ['ffmpeg', '-v', 'error', '-i', path] + (['-vf', vf] if vf else []) + ['-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W)


def centre_metrics(path, W, H, crop):
    y0, y1, x0, x1 = crop; out = []
    for g in frames_gray(path, W, H):
        c = np.ascontiguousarray(g[y0:y1, x0:x1])
        out.append((cv2.Laplacian(c, cv2.CV_64F).var(), len(FAST.detect(c, None))))
    return np.array(out)


def flow_motion(path, W=400, H=300):
    prev = None; out = []
    for g in frames_gray(path, W, H, vf=f'scale={W}:{H}'):
        row = [np.nan] * 3
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8)
            if pts is not None and len(pts) > 20:
                nx, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3); ok = st.ravel() == 1
                M, _ = cv2.estimateAffinePartial2D(pts[ok], nx[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None:
                    row = [M[0, 2] + (M[0, 0] - 1) * W / 2 + M[0, 1] * H / 2, M[1, 2] + M[1, 0] * W / 2 + (M[1, 1] - 1) * H / 2, np.arctan2(M[1, 0], M[0, 0])]
        out.append(row); prev = g
    return np.array(out)


def relative(x, key, low, win=REL_WIN, minn=REL_MIN):
    n = len(x); r = np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - win), min(n, i + win + 1); s = x[a:b][key[a:b] < low]
        if len(s) >= minn:
            r[i] = x[i] / np.median(s)
    return r


def binstats(df, col, bins):
    g = df.groupby(pd.cut(df[col], bins), observed=True)
    return [dict(lo=float(i.left), hi=float(i.right), n=int(len(x)), rel_sh=float(x.rel_sh.median()),
                 q25=float(x.rel_sh.quantile(.25)), q75=float(x.rel_sh.quantile(.75)), rel_fast=float(x.rel_fast.median()),
                 w=float(x.w.median())) for i, x in g]


RES = {}
# ---------------- stereo rig ----------------
s = CFG['stereo']
ft = pd.read_csv(s['frame_timestamps']); im = pd.read_csv(s['imu'])
q = centre_metrics(s['video'], 1600, 1200, (300, 900, 400, 1200))
ti = im.timestamp_us.values.astype(float); G = im[['gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s']].values
st = ft.left_exposure_start_us.values + CAM_IMU_OFFSET_US; ex = ft.left_exposure_time_us.values.astype(float)
wexp = np.array([G[(ti >= a) & (ti <= a + e)].mean(0) for a, e in zip(st, ex)])
ctr = ft.left_exposure_centre_us.values + CAM_IMU_OFFSET_US
rot = np.array([G[(ti >= ctr[i - 1]) & (ti < ctr[i])].mean(0) * (ctr[i] - ctr[i - 1]) * 1e-6 if i > 0 else [np.nan] * 3 for i in range(len(ctr))])
M = flow_motion(s['video']) * 4.0            # 400×300 → 1600×1200 pixels
ok = ~np.isnan(M[:, 0]) & ~np.isnan(rot[:, 0])
A, *_ = np.linalg.lstsq(rot[ok], M[ok, :2], rcond=None)
pred = rot[ok] @ A
r2 = 1 - ((M[ok, :2] - pred) ** 2).sum() / ((M[ok, :2] - M[ok, :2].mean(0)) ** 2).sum()
blur = np.linalg.norm(wexp @ A, axis=1) * ex * 1e-6
d = pd.DataFrame(dict(frame=np.arange(len(q)), time_s=ft.left_t_rel_s.values, exposure_ms=ex / 1000, sharpness=q[:, 0], fast=q[:, 1],
                      blur_px_pred=blur, w=np.degrees(np.linalg.norm(wexp, axis=1))))
d['rel_sh'] = relative(d.sharpness.values, d.blur_px_pred.values, 0.35)
d['rel_fast'] = relative(d.fast.values.astype(float), d.blur_px_pred.values, 0.35)
d.to_csv(f'{OUT}/stereo_frames_full.csv', index=False)
y = (d.rel_sh < 0.6).astype(int); okr = d.rel_sh.notna()
RES['stereo'] = dict(frames=len(d), map_r2=float(r2), map_px_per_rad=A.T.tolist(),
    exposure_ms_p1_p50_p99=[float(x) for x in np.percentile(d.exposure_ms, [1, 50, 99])],
    spearman_blur_rel=float(spearmanr(d.blur_px_pred, d.rel_sh, nan_policy='omit').statistic),
    blur_bins=binstats(d, 'blur_px_pred', [0, .25, .5, .75, 1, 1.5, 2, 3, 6]),
    rot_bins=binstats(d, 'w', [0, 5, 10, 20, 40, 80, 250]),
    blur_frac={str(t): float((d.blur_px_pred > t).mean()) for t in (.5, 1, 2, 3)},
    rot_frac={str(t): float((d.w > t).mean()) for t in (10, 20, 40, 80)},
    rot_pct_50_90_95_99=[float(x) for x in np.percentile(d.w, [50, 90, 95, 99])],
    auc_flag_rel_lt_0p6=float(roc_auc_score(y[okr], d.blur_px_pred[okr])),
    rule={str(t): dict(flag=float((d.blur_px_pred[okr] > t).mean()),
                       recall=float(((d.blur_px_pred[okr] > t) & (y[okr] == 1)).sum() / max(1, y[okr].sum())),
                       precision=float(((d.blur_px_pred[okr] > t) & (y[okr] == 1)).sum() / max(1, (d.blur_px_pred[okr] > t).sum()))) for t in (1.0, 1.5, 2.0)})
print('stereo done', RES['stereo']['map_r2'], RES['stereo']['spearman_blur_rel'])

# ---------------- phones ----------------
for name, p in CFG['phones'].items():
    off = json.load(open(p['offset_json']))[p['offset_key']]['best_offset_s']
    pts = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'packet=pts_time', '-of', 'csv=p=0', p['video']], capture_output=True, text=True).stdout.split()
    pts = np.sort(np.array([float(x) for x in pts if x and x != 'N/A']))
    q = centre_metrics(p['video'], 1920, 1080, (270, 810, 480, 1440)); n = min(len(q), len(pts)); q = q[:n]; pts = pts[:n]
    im = pd.read_csv(p['imu']); g = im[im.sensor == 'gyroscope']; ti = g.timestamp_relative_sec.values - off; G = g[['x', 'y', 'z']].values
    w = np.array([np.degrees(np.linalg.norm(G[(ti >= t - 1 / 60) & (ti < t + 1 / 60)].mean(0))) if ((ti >= t - 1 / 60) & (ti < t + 1 / 60)).sum() > 3 else np.nan for t in pts])
    d = pd.DataFrame(dict(frame=np.arange(n), time_s=pts, sharpness=q[:, 0], fast=q[:, 1], w=w))
    key = d.w.fillna(999).values
    d['rel_sh'] = relative(d.sharpness.values, key, 5); d['rel_fast'] = relative(d.fast.values.astype(float), key, 5)
    ks = json.load(open(p['keysteps']))['keysteps']; nt = np.zeros(n, bool)
    for k in ks:
        if k['verb'] == 'non_task':
            nt |= ((d.time_s >= k['t_start']) & (d.time_s < k['t_end'])).values
    d['non_task'] = nt.astype(int)
    d.to_csv(f'{OUT}/{name}_frames_full.csv', index=False)
    v = d.dropna(subset=['rel_sh', 'w']); y = (v.rel_sh < 0.6).astype(int); sev = v.rel_sh < 0.25; vnt = v.non_task == 1
    RES[name] = dict(frames=n, offset_s=off, offset_applied=True,
        rot_bins=binstats(d, 'w', [0, 5, 10, 20, 40, 80, 250]),
        rot_pct_50_90_95_99=[float(x) for x in np.nanpercentile(d.w, [50, 90, 95, 99])],
        rot_frac={str(t): float((d.w > t).mean()) for t in (10, 20, 40, 80)},
        spearman_w_rel=float(spearmanr(d.w, d.rel_sh, nan_policy='omit').statistic),
        auc_flag_rel_lt_0p6=float(roc_auc_score(y, v.w)),
        rule={str(t): dict(flag=float((v.w > t).mean()), recall=float(((v.w > t) & (y == 1)).sum() / max(1, y.sum())),
                           precision=float(((v.w > t) & (y == 1)).sum() / max(1, (v.w > t).sum()))) for t in (20, 40)},
        frames_without_reference=int(d.rel_sh.isna().sum()),
        nontask=dict(frames_nt=float(vnt.mean()), severe=int(sev.sum()), severe_in_nt=float((sev & vnt).sum() / max(1, sev.sum())),
                     rel_nt=float(v.rel_sh[vnt].median()), rel_task=float(v.rel_sh[~vnt].median())))
    print(name, 'done')
json.dump(RES, open(f'{OUT}/results_fn06.json', 'w'), indent=1)
print('ok')
