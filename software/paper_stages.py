"""Traceable segmentation stage snapshots."""
from pathlib import Path
import hashlib
import json
import numpy as np

STAGES=(('atoms','原始 atom'),('no_f','无 F（重分割前）'),
        ('with_f','有 F（重分割前）'),('before_p','P 前'),('after_p','P 后'))
DEFINITIONS={
    'atoms':'Original watershed A atoms, before F or merging.',
    'no_f':'Independent A14 merge on the original A-only graph, using the same fold and threshold; before repartition.',
    'with_f':'Actual A14 merge on the A+F graph; before repartition.',
    'before_p':'After risk-aware repartition and residual stitching, before PF_PA or exterior cleanup.',
    'after_p':'Final output after PF_PA and exterior cleanup, using the recorded parameters.'}

def a_only_mapping(atoms,prediction,cfg):
    """Identical to the production F-disabled A14 branch; never modify live atoms."""
    import historical_core as old
    import lightgbm as lgb
    counts,centers=old.compute_node_geometry(atoms)
    try:edges,features,_=old.build_rag_edge_features(atoms,prediction,counts,centers)
    except RuntimeError as exc:
        if 'no adjacency edges' not in str(exc):raise
        return np.arange(int(atoms.max())+1,dtype=np.int32)
    root=Path(__file__).parent/'resources/models'
    model=root/f'a14_fold{cfg["fold"]}.txt'
    expected=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['fold_artifacts'][str(cfg['fold'])]['model_sha256']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=expected:raise ValueError('A14 model hash mismatch')
    probability=lgb.Booster(model_file=str(model)).predict(features,num_threads=cfg['threads'])
    return old.decode_merge_edges(int(atoms.max()),edges,probability>=cfg['threshold'])

def write_manifest(output,metadata):
    output=Path(output)
    source=Path(metadata['input'])
    digest=hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    manifest=dict(version=1,input=metadata['input'],prediction_sha256=digest.hexdigest(),
                  atoms_file='viewer_atoms.npy',maps_file='stage_cluster_maps.npz',
                  result_file=output.name,metadata_file=output.with_suffix('.json').name,
                  stage_definitions=DEFINITIONS,no_f_is_independently_computed=True)
    path=output.parent/'viewer_manifest.json';temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(manifest,indent=2),encoding='utf-8');temporary.replace(path)

class LazyFinal:
    def __init__(self,source):
        self.source=source;self.shape=source.shape;self.dtype=np.dtype('uint32')
    def __len__(self):return self.shape[0]
    def __getitem__(self,z):return self.source.plane('after_p',z)


class StageSource:
    def __init__(self,manifest):
        self.path=Path(manifest).resolve();self.root=self.path.parent
        self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        def asset(key):
            name=self.manifest[key]
            if Path(name).name!=name:raise ValueError('Invalid snapshot asset path')
            return self.root/name
        self.meta=json.loads(asset('metadata_file').read_text(encoding='utf-8'))
        self.na=int(self.meta['original_atoms']);self.shape=tuple(self.meta['shape'])
        self.cfg=self.meta['parameters'];self.box=[slice(a,b) for a,b in self.meta['bbox_processing_grid']]
        self.h5=None;self.prediction=None
        if self.manifest.get('storage')=='hdf5':
            import h5py
            self.h5=h5py.File(asset('result_file'),'r')
            self.atoms=self.h5['atoms'];self.prediction=self.h5['prediction'];self.final=LazyFinal(self)
            self.maps={k:self.h5['maps/'+k][:] for k in ('A_NO_F','A14','R_PLUS_ABS','PF_PA')}
            if self.h5['final'].shape!=self.atoms.shape or self.prediction.shape!=self.shape:raise ValueError('Invalid compressed snapshot dimensions')
        else:
            self.atoms=np.load(asset('atoms_file'),mmap_mode='r',allow_pickle=False)
            self.final=np.load(asset('result_file'),mmap_mode='r',allow_pickle=False)
            with np.load(asset('maps_file'),allow_pickle=False) as maps:
                self.maps={k:np.array(maps[k]) for k in ('A_NO_F','A14','R_PLUS_ABS','PF_PA')}
        if not self.cfg.get('f_enabled'):raise ValueError('Five-stage snapshots require an F-enabled run and its independent A-only comparison')
        if self.final.shape!=self.shape:raise ValueError('Snapshot result shape mismatch')
        if tuple(s.stop-s.start for s in self.box)!=self.atoms.shape:raise ValueError('Snapshot atom shape mismatch')
        n=int(self.meta['atoms']) if self.h5 is not None else int(self.atoms.max())
        if len(self.maps['A_NO_F'])!=self.na+1:raise ValueError('Invalid A-only mapping')
        if any(len(self.maps[k])!=n+1 for k in ('A14','R_PLUS_ABS','PF_PA')):raise ValueError('Invalid stage mapping')
        self.anchors={key:self.color_anchors(mapping) for key,mapping in self.maps.items()}

    def color_anchors(self,mapping):
        # Anchor a component to its smallest A atom, or its smallest F atom if pure F.
        # Stable anchors preserve colors for unchanged cells across stage switches.
        sentinel=np.iinfo(np.uint32).max
        anchors=np.full(int(mapping.max())+1,sentinel,np.uint32)
        np.minimum.at(anchors,mapping[1:],np.arange(1,len(mapping),dtype=np.uint32))
        anchors[anchors==sentinel]=0;anchors[0]=0
        return anchors

    def close(self):
        if self.h5 is not None:self.h5.close();return
        for array in (self.atoms,self.final):
            handle=getattr(array,'_mmap',None)
            if handle is not None:handle.close()

    def plane(self,key,z,color_ids=False):
        if key not in dict(STAGES):raise ValueError('Unknown stage')
        if not 0<=z<self.shape[0]:raise IndexError(z)
        if key=='after_p' and self.h5 is None:
            labels=np.asarray(self.final[z])
            return self.anchors['PF_PA'][labels] if color_ids else labels
        step=self.cfg['xy_step'];height=(self.shape[1]+step-1)//step;width=(self.shape[2]+step-1)//step
        result=np.zeros((height,width),np.uint32)
        if self.box[0].start<=z<self.box[0].stop:
            if key=='after_p':
                labels=self.h5['final'][z-self.box[0].start]
                if color_ids:labels=self.anchors['PF_PA'][labels]
                result[self.box[1],self.box[2]]=labels
                if self.cfg['flip_y']:result=result[::-1]
                if step>1:result=np.repeat(np.repeat(result,step,axis=0),step,axis=1)
                return result[:self.shape[1],:self.shape[2]]
            atoms=np.array(self.atoms[z-self.box[0].start])
            if key in ('atoms','no_f'):atoms[atoms>self.na]=0
            if key=='atoms':labels=atoms
            else:
                name={'no_f':'A_NO_F','with_f':'A14','before_p':'R_PLUS_ABS'}[key]
                labels=self.maps[name][atoms]
                if color_ids:labels=self.anchors[name][labels]
            result[self.box[1],self.box[2]]=labels
        if self.cfg['flip_y']:result=result[::-1]
        if step>1:result=np.repeat(np.repeat(result,step,axis=0),step,axis=1)
        return result[:self.shape[1],:self.shape[2]]
