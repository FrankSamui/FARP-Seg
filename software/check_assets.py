"""Validate every packaged inference fold, not just the demonstration fold."""
from pathlib import Path
import hashlib,json
import numpy as np
import lightgbm as lgb
from full_pipeline import Assets,MODELS

def main():
    root=Path(__file__).resolve().parent/'resources/models'
    original=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    verified=[]
    for fold in range(3):
        a14=root/f'a14_fold{fold}.txt'
        assert hashlib.sha256(a14.read_bytes()).hexdigest()==original['fold_artifacts'][str(fold)]['model_sha256']
        assert lgb.Booster(model_file=str(a14)).num_feature()==14
        assets=Assets(fold,1)
        for name in assets.manifest['sha256']:
            if name.startswith(f'fold{fold}/'):assets.path(name.split('/',1)[1]);verified.append(name)
        assert assets.model('region_ranker.txt').num_feature()==14
        names=assets.json('repartition/columns.json')['names']
        for kind in ('gain','q20','cat','pos'):assert assets.model(f'repartition/{kind}.txt').num_feature()==len(names)
        assert assets.model('stitch_gain.txt').num_feature()==34
        cal=assets.json('repartition/calibration.json')
        assert set(cal['by_budget'])=={'8','16','32'}
        with np.load(assets.path('f_morphology_bank.npz')) as bank:
            assert bank['anchors'].shape[1]==7 and len(bank['anchors'])==len(bank['loo'])
            assert np.all(bank['scale']>0) and np.isfinite(bank['anchors']).all()
    print(f'PASS: all three A14/R/stitch/morphology folds; {len(verified)} full-stage assets SHA256 verified')

if __name__=='__main__':main()
