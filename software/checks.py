"""I/O, geometry, and segmentation checks."""
import tempfile
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from engine import DEFAULTS, ROOT, read_volume, export_volume, clean_exterior, infer
import historical_core as old
import full_pipeline as full


def main():
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as work:
        work = Path(work)
        a = np.zeros((5,31,35),np.uint32); a[:,4:15,3:14]=65537; a[:,16:25,18:29]=80001
        for ext in ('.npy','.tif','.h5'):
            dest=work/('segment'+ext); export_volume(a,dest,{'axes':'ZYX'})
            b=read_volume(dest, 'segment' if ext=='.h5' else None)
            assert np.array_equal(a,b) and b.dtype==np.uint32, ext
        np.save(work/'bad.npy',np.zeros((2,3,4,5)))
        try: read_volume(work/'bad.npy')
        except ValueError: pass
        else: raise AssertionError('4D input accepted')
        # Transitive merge and isolated background; arbitrary label IDs preserved.
        edges=np.array([[1,2],[2,3],[3,4]])
        m=old.decode_merge_edges(4,edges,np.array([True,True,False]))
        assert m[0]==0 and m[1]==m[2]==m[3] and m[4]!=m[3]
        # Fixed initial outline: increasing width cannot resurrect deleted voxels.
        atoms=np.zeros((7,51,51),np.int32); atoms[:,6:45,6:45]=1
        atoms[:,6:9,6:45]=2; atoms[:,9:16,6:45]=3
        mapping=np.arange(4,dtype=np.uint32)
        previous=None
        for width in (4,6,8):
            cfg=dict(DEFAULTS,width=width)
            out=clean_exterior(atoms,mapping,cfg,lambda *a:None,lambda:None)
            assert out.shape==atoms.shape and out.dtype==np.uint32
            if previous is not None: assert np.all((previous>0)|(out==0))
            previous=out
        # RAG must produce the exact fourteen inputs expected by the trained model.
        atoms=np.ones((6,20,20),np.int32); atoms[:,:,10:]=2
        pred=np.zeros(atoms.shape,np.float32); pred[:,:,9:11]=.9
        count,center=old.compute_node_geometry(atoms)
        edges,features,raw=old.build_rag_edge_features(atoms,pred,count,center)
        assert edges.tolist()==[[1,2]] and features.shape==(1,14) and np.isfinite(features).all()
        assert int(raw[0,0])==120
        # Missing enclosed tissue must become one nonbranching F track. Existing
        # A labels are immutable, and a true image frame alone cannot create F.
        original=np.zeros((8,64,64),np.int32);original[:,12:52,12:52]=1;original[:,25:39,25:39]=0
        unified,fv,stats=full.f_atoms(original,dict(top=False,bottom=False,left=False,right=False),{},lambda *a:None,lambda:None)
        assert stats['n_F_atoms']==1 and stats['n_F_links']==7
        assert np.array_equal(unified[original>0],original[original>0])
        assert np.all(unified[:,25:39,25:39]==2)
        assert np.allclose(fv[0],[np.log1p(8*196),np.log1p(8),np.log1p(196),np.log1p(196),0,1,1])
        empty=np.zeros((2,32,32),np.int32)
        _,_,stats=full.f_atoms(empty,dict(top=True,bottom=True,left=True,right=True),{},lambda *a:None,lambda:None)
        assert stats['n_F_atoms']==0
        # P-A's median is undefined on a one-cell crop: preserve the cell and
        # record that condition instead of inventing a size threshold.
        dummy=dict(mass=np.array([0,100]),edges=np.empty((0,2),np.int32),faces=np.empty(0))
        class NoBank:
            def path(self,*a):raise AssertionError('Empty F needs no morphology bank')
        one=np.ones((1,10,10),np.int32)
        result,stats=full.postprocess(one,np.array([0,1]),dummy,1,np.empty((0,7)),NoBank(),{})
        assert result.tolist()==[0,1] and 'reason' in stats['PA']
        # The no-adjacency case is valid and must restore odd source dimensions.
        src=work/'single_cell.npy'; np.save(src,np.full((3,31,35),64,np.uint8))
        dest=work/'single_cell_segment.npy'
        meta=infer(src,dest,dict(xy_step=2,cleanup=False,flip_y=True,f_enabled=False,r_enabled=False,stitch_enabled=False,post_enabled=False))
        restored=np.load(dest,allow_pickle=False)
        assert restored.shape==(3,31,35) and restored.dtype==np.uint32
        assert meta['instances']==1 and meta['edges']==0
    print('PASS: uint32 TIFF/NPY/HDF5 round trips, invalid input, merge, cleanup monotonicity, A14 feature contract, F geometry/tracks, empty-F abstention, undefined P-A scale, odd-size restoration')


if __name__=='__main__': main()
