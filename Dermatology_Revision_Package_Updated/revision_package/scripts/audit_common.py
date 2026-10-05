"""Shared analysis utilities. No model training, network access, or unsafe eval."""
from pathlib import Path
import ast
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd
from scipy.stats import beta, norm, t


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def new_output(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise ValueError(f'Output directory is not empty: {p}. Use a new directory.')
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def wilson(k, n, alpha=.05):
    if n == 0: return (np.nan, np.nan)
    if not 0 <= k <= n: raise ValueError('Invalid count')
    z = norm.ppf(1-alpha/2)
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0., center-half), min(1., center+half)


def clopper_pearson(k, n, alpha=.05):
    if n == 0: return (np.nan, np.nan)
    return (0. if k == 0 else beta.ppf(alpha/2, k, n-k+1),
            1. if k == n else beta.ppf(1-alpha/2, k+1, n-k))


def parse_saved_counts(text):
    # Original CSVs contain Python reprs of numpy integers, not JSON.
    # Only strip this exact numeric wrapper, then use literal_eval (never eval).
    text = re.sub(r'np\.int(?:32|64)\((-?\d+)\)', r'\1', text)
    obj = ast.literal_eval(text)
    if not isinstance(obj, dict): raise ValueError('Expected counts dictionary')
    for attr, groups in obj.items():
        if not isinstance(groups, dict): raise ValueError('Invalid group dictionary')
        for name, pair in groups.items():
            if len(pair) != 2 or any(type(x) is not int for x in pair):
                raise ValueError(f'Invalid counts for {attr}/{name}')
            if not 0 <= pair[0] <= pair[1] or pair[1] < 1:
                raise ValueError(f'Impossible counts for {attr}/{name}')
    return obj


def corrected_fold_comparison(differences, test_train_ratio):
    d = np.asarray(differences, dtype=float)
    if len(d) < 3 or not np.isfinite(d).all() or test_train_ratio <= 0:
        raise ValueError('Need >=3 finite paired differences and a positive test/train ratio')
    se = float(d.std(ddof=1)*np.sqrt(1/len(d)+test_train_ratio))
    mean = float(d.mean())
    if se == 0:
        # Do not claim infinite certainty from zero observed fold variance.
        return dict(mean_difference=mean, corrected_se=0., ci_low=None, ci_high=None,
                    two_sided_p=None, note='Zero variance; inferential summary not reported')
    q = t.ppf(.975, len(d)-1)
    return dict(mean_difference=mean, sample_sd=float(d.std(ddof=1)), corrected_se=se,
                ci_low=mean-q*se, ci_high=mean+q*se,
                two_sided_p=float(2*t.sf(abs(mean/se), len(d)-1)),
                test_train_ratio=test_train_ratio,
                note='Approximate dependence-corrected CV sensitivity; not a selection correction or equivalence test')


def normalize_text(x):
    if pd.isna(x) or not str(x).strip(): return 'missing'
    return str(x).strip().lower()


def audit_proxy_columns(df):
    """NEW audit labels only. Never pass these silently to an old checkpoint."""
    d = df.copy()
    age = pd.to_numeric(d.get('age', pd.Series(np.nan, index=d.index)), errors='coerce')
    age = age.where(age.between(0, 120))
    d['age_proxy'] = pd.cut(age, [-.001, 17, 59, 120],
                              labels=['0-17', '18-59', '60+']).astype('object').fillna('missing')
    raw_gender = d.get('dominant_gender', d.get('gender', pd.Series('missing', index=d.index)))
    g = raw_gender.map(normalize_text)
    d['gender_proxy'] = g.map({'man':'male', 'male':'male', 'woman':'female', 'female':'female',
                             'other/unclear':'unclear', 'unclear':'unclear', 'unknown':'unknown',
                             'missing':'missing'}).fillna('unmapped')
    raw_app = d.get('dominant_race', d.get('skin_tone', pd.Series('missing', index=d.index)))
    app = raw_app.map(normalize_text)
    allowed = {'white','black','asian','indian','latino hispanic','middle eastern',
               'unclear','unknown','missing'}
    # These are original source label strings, NOT race measurements or skin-tone levels.
    d['appearance_proxy'] = app.where(app.isin(allowed), 'unmapped')
    d['appearance_raw_normalized'] = app
    d['gender_raw_normalized'] = g
    for c in ['annotation_route', 'source_status', 'body_site']:
        if c not in d: d[c] = 'unknown'
        else: d[c] = d[c].map(normalize_text)
    d['composite_proxy'] = d[['age_proxy','gender_proxy','appearance_proxy']].astype(str).agg('|'.join, axis=1)
    return d


def build_flat_image_index(root):
    """Index an explicitly flat/flattened image directory by unique basename.

    Basename fallback is never implicit.  Callers must opt in with
    ``--flat-images`` and this function refuses ambiguous duplicate names.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Image directory not found: {root}')
    index = {}
    collisions = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.name.casefold()
        if key in index:
            collisions.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path.resolve()
    if collisions:
        example = next(iter(collisions.values()))
        raise ValueError(
            'Flat-image mode is ambiguous because image basenames repeat. '
            f'Example: {[str(path) for path in example]}'
        )
    if not index:
        raise ValueError(f'No supported image files found under: {root}')
    return index


def resolve_image(row, root, strip_prefix='', flat_index=None):
    root = Path(root).resolve()
    raw = str(row.get('filepath', '')).replace('\\', '/')
    if raw in ('', 'nan'):
        # Use the ORIGINAL split only, before cross-validation overwrites it.
        if any(str(row.get(c, '')) in ('', 'nan') for c in ['split','label','filename']):
            raise ValueError('Need filepath or ORIGINAL split + label + filename')
        raw = '/'.join(str(row[c]) for c in ['split','label','filename'])
    prefix = strip_prefix.replace('\\', '/').rstrip('/')
    if prefix:
        if not raw.startswith(prefix+'/'): raise ValueError(f'Path lacks explicit prefix: {raw}')
        raw = raw[len(prefix)+1:]
    if flat_index is not None:
        key = Path(raw).name.casefold()
        if key not in flat_index:
            raise FileNotFoundError(
                f'No unique flat-directory image named {Path(raw).name!s} under {root}'
            )
        p = Path(flat_index[key]).resolve()
    else:
        p = Path(raw)
        p = p.resolve() if p.is_absolute() else (root/p).resolve()
    if not p.is_file(): raise FileNotFoundError(f'{p}; preserve original paths or use --strip-prefix')
    if not p.is_relative_to(root): raise ValueError(f'Image is outside data root: {p}')
    return p, p.relative_to(root).as_posix()
