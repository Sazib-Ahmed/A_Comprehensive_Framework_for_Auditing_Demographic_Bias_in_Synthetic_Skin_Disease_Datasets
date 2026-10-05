"""Scientific failure-case tests; artificial data only, never manuscript evidence."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_common import (parse_saved_counts,wilson,audit_proxy_columns,
                          build_flat_image_index,resolve_image)


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


ev=module('03_evaluate_predictions')
data=module('02_audit_data')


class MetricsTests(unittest.TestCase):
    def test_safe_count_parser(self):
        self.assertEqual(parse_saved_counts("{'a': {'x': (np.int64(2), np.int64(3))}}")['a']['x'],(2,3))
        with self.assertRaises((ValueError,SyntaxError)):
            parse_saved_counts("__import__('os').system('false')")
    def test_impossible_count_rejected(self):
        with self.assertRaises(ValueError):parse_saved_counts("{'a': {'x': (3,2)}}")
    def test_tiny_perfect_group_has_wide_interval(self):
        lo,hi=wilson(3,3);self.assertLess(lo,.45);self.assertAlmostEqual(hi,1.)
    def test_age_sixty_not_in_adult_audit_bin(self):
        d=audit_proxy_columns(pd.DataFrame({'age':[17,18,59,60,64,65,None], 'dominant_race':['white']*7}))
        self.assertEqual(d.age_proxy.tolist(),['0-17','18-59','18-59','60+','60+','60+','missing'])
    def test_undefined_rates_are_nan(self):
        d=pd.DataFrame({'g':['x','x'],'y_true':['a','a'],'y_pred':['a','a']})
        r=ev.rate_table(d,'g',['a','b'])
        self.assertTrue(np.isnan(r[r.disease=='b'].tpr.iloc[0]))
        eo=ev.eod_from_rates(r,['a','b'],1)
        self.assertTrue(all(np.isnan(z['eod']) for z in eo))
    def test_equal_tpr_different_fpr_is_not_equalized_odds(self):
        d=pd.DataFrame({'g':['x']*4+['y']*4,
                        'y_true':['a','a','b','b']*2,
                        'y_pred':['a','a','b','b','a','a','a','a']})
        r=ev.rate_table(d,'g',['a','b']);z=ev.eod_from_rates(r,['a','b'],1)[0]
        self.assertEqual(z['tpr_gap'],0.);self.assertEqual(z['fpr_gap'],1.);self.assertEqual(z['eod'],1.)
    def test_single_retained_group_not_zero_eod(self):
        d=pd.DataFrame({'g':['x']*4+['y']*2,'y_true':['a','b','a','b','a','b'],'y_pred':['a','b','a','b','a','b']})
        r=ev.rate_table(d,'g',['a','b']);eo=ev.eod_from_rates(r,['a','b'],3)
        self.assertTrue(all(np.isnan(z['eod']) for z in eo))
    def test_probability_class_order_checked(self):
        d=pd.DataFrame([dict(sample_id='a',model='m',fold=1,y_true='a',y_pred='b',p_0=.9,p_1=.1)])
        with self.assertRaises(ValueError):ev.validate_predictions(d,['a','b'])
    def test_repeated_oof_ids_rejected(self):
        row=dict(sample_id='a',model='m',fold=1,y_true='a',y_pred='a')
        with self.assertRaises(ValueError):ev.validate_predictions(pd.DataFrame([row,row]),['a','b'])
    def test_cross_role_duplicates(self):
        p=pd.DataFrame({'sample_id_a':['a'],'sample_id_b':['b']})
        m=pd.DataFrame({'sample_id':['a','b'],'fold':[1,1],'role':['Train','Test']})
        self.assertEqual(len(data.cross_split_pairs(p,m)),1)
    def test_group_transitivity(self):
        m=module('06_make_grouped_splits');u=m.UnionFind();u.union('a','b');u.union('b','c')
        self.assertEqual(u.find('a'),u.find('c'))
    def test_hierarchical_metadata_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'Train'/'Herpes'/'image.png'
            path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
            resolved,sample_id=resolve_image({'filepath':'Train/Herpes/image.png'},root)
            self.assertEqual(resolved,path.resolve());self.assertEqual(sample_id,'Train/Herpes/image.png')
    def test_flat_fallback_requires_unique_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'a').mkdir();(root/'b').mkdir()
            (root/'a'/'same.png').write_bytes(b'a');(root/'b'/'same.png').write_bytes(b'b')
            with self.assertRaises(ValueError):build_flat_image_index(root)


if __name__=='__main__':unittest.main()
