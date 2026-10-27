"""FARP-Seg filling, agglomeration, repartitioning, stitching, and cleanup."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from kernels import fill, rays, tracks, repartition as rp, geometry, post
from kernels.pf_graph import PFComponentGraph
import historical_core as old

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'resources/models/full'


class Assets:
    def __init__(self,fold,threads):
        self.fold=fold;self.threads=threads;self.root=MODELS/f'fold{fold}'
        self.manifest=json.loads((MODELS/'assets.json').read_text(encoding='utf-8'))
        self.loaded={}

    def path(self,name):
        path=self.root/name;key=path.relative_to(MODELS).as_posix()
        expected=self.manifest['sha256'][key]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'模型/参数校验失败：{key}')
        return path

    def json(self,name):return json.loads(self.path(name).read_text(encoding='utf-8'))

    def model(self,name):
        if name not in self.loaded:
            import lightgbm as lgb
            self.loaded[name]=lgb.Booster(model_file=str(self.path(name)))
        return self.loaded[name]

    def predict(self,name,x):return self.model(name).predict(x,num_threads=self.threads)


def f_atoms(original,frame_sides,cfg,progress,check):
    """NECK85/RAY50 2D chambers + reciprocal adjacent-Z tracks, no A edits."""
    na=int(original.max());nz=len(original)
    fill.ENCLOSURE_THRESHOLD=cfg.get('f_enclosure',.5)
    paths=rays.build_ray_paths(original.shape[1:],64)
    patches=np.zeros(original.shape,np.int32)
    node_z=[];node_label=[];node_area=[];links=[];previous={}
    for z,aa in enumerate(original):
        check()
        basins,depth,_,_=fill.initial_watershed_basins(aa>0,frame_sides)
        edges=fill.basin_adjacencies(basins,depth)
        run=fill.run_configuration(aa,frame_sides,paths,basins,depth,edges,cfg.get('f_neck',.85))
        current=run.repair_labels;patches[z]=current
        labels,areas=np.unique(current[current>0],return_counts=True)
        nodes={}
        for label,area in zip(labels,areas):
            nodes[int(label)]=len(node_z);node_z.append(z);node_label.append(int(label));node_area.append(int(area))
        if z:
            accepted,_=tracks.reciprocal_best_links(patches[z-1],current)
            links.extend((previous[a],nodes[b]) for a,b,*_ in accepted)
        previous=nodes
        if z%8==0:progress(27+int(10*(z+1)/nz),'F 补区与跨层连接',f'切片 {z+1} / {nz} · {len(node_z):,} 个二维补区')
    n=len(node_z)
    if not n:return original,np.empty((0,7)),dict(n_A_atoms=na,n_F_atoms=0,n_F_patches=0,n_F_links=0)
    uf=old.UnionFind(n)
    for u,v in links:uf.union(u,v)
    roots=[uf.find(i) for i in range(n)]
    # Nodes were added in (Z, local repair ID) order, same order as frozen tracks.
    unique={};track=np.empty(n,np.int32)
    for i,root in enumerate(roots):
        if root not in unique:unique[root]=len(unique)+1
        track[i]=unique[root]
    nf=len(unique);unified=original.copy()
    zvalues=np.asarray(node_z);areas=np.asarray(node_area,float)
    labels=np.asarray(node_label)
    for z in np.unique(zvalues):
        ids=zvalues==z;lut=np.zeros(int(patches[z].max())+1,np.int32)
        lut[labels[ids]]=na+track[ids];mask=patches[z]>0
        assert not np.any(mask&(original[z]>0))
        unified[z][mask]=lut[patches[z][mask]]
    count=np.bincount(track,minlength=nf+1);mass=np.bincount(track,weights=areas,minlength=nf+1)
    square=np.bincount(track,weights=areas**2,minlength=nf+1)
    peak=np.zeros(nf+1);np.maximum.at(peak,track,areas)
    lo=np.full(nf+1,nz);hi=np.full(nf+1,-1);np.minimum.at(lo,track,zvalues);np.maximum.at(hi,track,zvalues)
    first=np.zeros(nf+1);last=np.zeros(nf+1)
    for t,z,area in zip(track,zvalues,areas):
        if z==lo[t]:first[t]=area
        if z==hi[t]:last[t]=area
    mean=mass[1:]/count[1:];variance=np.maximum(square[1:]/count[1:]-mean**2,0)
    features=np.column_stack([np.log1p(mass[1:]),np.log1p(count[1:]),np.log1p(mean),np.log1p(peak[1:]),
                              np.sqrt(variance)/np.maximum(mean,1e-12),peak[1:]/np.maximum(mean,1e-12),
                              .5*(first[1:]+last[1:])/np.maximum(peak[1:],1)])
    assert np.array_equal(unified[original>0],original[original>0])
    return unified,features,dict(n_A_atoms=na,n_F_atoms=nf,n_F_patches=n,n_F_links=len(links))


def region_pool(a):
    """Same 14 region descriptors and whole-cell candidate pool as Exp1-3-1."""
    mapping,edges,prob,mass,faces=(a[k] for k in ('mapping','edges','prob','mass','faces'))
    n=int(mapping.max());u,w=mapping[edges].T;same=u==w
    counts=np.bincount(mapping,minlength=n+1);vol=np.bincount(mapping,weights=mass,minlength=n+1)
    pressure=np.bincount(u[same],weights=(1-prob[same])*np.log1p(faces[same]),minlength=n+1)
    mx=np.zeros(n+1);np.maximum.at(mx,u[same],1-prob[same])
    seeds=list(dict.fromkeys(int(x) for x in np.r_[np.argsort(pressure)[::-1][:96],np.argsort(vol)[::-1][:24]] if x>0))
    key=(np.minimum(u[~same],w[~same]).astype(np.int64)<<32)|np.maximum(u[~same],w[~same])
    keys,inv=np.unique(key,return_inverse=True);weight=np.bincount(inv,weights=faces[~same]*prob[~same]);nei={}
    for key,value in zip(keys,weight):
        x,y=int(key>>32),int(key&0xffffffff);nei.setdefault(x,[]).append((value,y));nei.setdefault(y,[]).append((value,x))
    for x in nei:nei[x].sort(reverse=True)
    order=np.argsort(mapping,kind='stable');start=np.r_[0,np.cumsum(np.bincount(mapping,minlength=n+1))]
    groups={i:np.sort(order[start[i]:start[i+1]]) for i in range(1,n+1)}
    seen=set();rows=[];regions=[]
    for seed in seeds:
        expanded=[seed]
        for step in range(4):
            if step:
                choices=[y for _,y in nei.get(seed,[]) if y not in expanded]
                if not choices:break
                expanded.append(choices[0])
            ids=tuple(sorted(expanded));na=int(counts[list(ids)].sum())
            if na>8000:break
            if ids in seen or na<2:continue
            seen.add(ids);atoms=np.sort(np.concatenate([groups[c] for c in ids]));eids=np.flatnonzero(np.isin(u,ids)&np.isin(w,ids))
            if not len(eids):continue
            ss=same[eids];pp=prob[eids];ff=faces[eids];bm=a['base'][eids]
            values=[len(ids),np.log1p(na),np.log1p(vol[list(ids)].sum()),np.log1p(len(eids)),np.log1p(pressure[list(ids)].sum()),float(mx[list(ids)].max()),float(np.mean(pp)),float(np.std(pp)),float(np.mean(pp[ss])) if ss.any() else 0.,float(np.mean(pp[~ss])) if (~ss).any() else 0.,float(np.mean(bm[:,0])),float(np.std(bm[:,0])),float(np.average(pp,weights=ff)),float(np.mean(ss))]
            rows.append(dict(region=len(rows),clusters=list(ids),heuristic=float(pressure[list(ids)].sum()),features=values));regions.append((atoms,eids))
    return rows,regions


def node_slabs(atoms,check=lambda:None):
    """Exact per-atom per-Z moments used by the trained partition selector."""
    n=int(atoms.max())+1;nz,ny,nx=atoms.shape;stride=nz+1
    yy,xx=np.indices((ny,nx),dtype=np.float64)
    weights=(yy.ravel(),xx.ravel(),(yy*yy).ravel(),(xx*xx).ravel(),(yy*xx).ravel())
    keys=[];slabs=[]
    for z,aa in enumerate(atoms):
        check();flat=aa.ravel();count=np.bincount(flat,minlength=n).astype(float);present=np.flatnonzero(count[1:])+1
        sy,sx,syy,sxx,syx=[np.bincount(flat,weights=w,minlength=n)[present] for w in weights];vol=count[present]
        perimeter=np.zeros(n,float)
        for a,b in ((aa[:-1],aa[1:]),(aa[:,:-1],aa[:,1:])):
            mask=a!=b;perimeter+=np.bincount(a[mask],minlength=n)+np.bincount(b[mask],minlength=n)
        for boundary in (aa[0],aa[-1],aa[:,0],aa[:,-1]):perimeter+=np.bincount(boundary,minlength=n)
        overlap=np.zeros(n,float)
        if z+1<nz:
            bb=atoms[z+1];same=(aa==bb)&(aa>0);overlap=np.bincount(aa[same],minlength=n).astype(float)
        slabs.append(np.column_stack([vol,vol*z,sy,sx,vol*z*z,syy,sxx,sy*z,sx*z,syx,perimeter[present],overlap[present]]));keys.append(present.astype(np.int64)*stride+z)
    keys=np.concatenate(keys);stats=np.concatenate(slabs);order=np.argsort(keys,kind='stable')
    return keys[order],stats[order]


def partition_geometry(labels,slab,slab_atom,contacts,pairs):
    nc=int(labels.max());parent=labels[slab_atom];moment=np.zeros((nc+1,10));np.add.at(moment,parent,slab[:,:10])
    surface=np.bincount(parent,weights=slab[:,10]+2*slab[:,0]-2*slab[:,11],minlength=nc+1)
    same=labels[pairs[:,0]]==labels[pairs[:,1]];np.add.at(surface,labels[pairs[same,0]],-2*contacts[same])
    assert np.all(surface[1:]>0)
    desc=geometry.descriptors(moment[1:],surface[1:]);weight=moment[1:,0]/moment[1:,0].sum()
    return np.r_[np.average(desc,axis=0,weights=weight),np.min(desc,axis=0),np.max(desc,axis=0)]


def partition_features(atom_ids,eids,a,slabs,nz):
    maps,families=rp.proposal_bank(atom_ids,eids,a)
    edges=a['edges'][eids];pairs=np.searchsorted(atom_ids,edges);prob=a['prob'][eids];faces=a['faces'][eids]
    previous=maps[0][pairs[:,0]]==maps[0][pairs[:,1]];base=a['base'][eids]
    keys,stats=slabs;left=np.searchsorted(keys,atom_ids*(nz+1));right=np.searchsorted(keys,(atom_ids+1)*(nz+1));length=right-left
    ix=np.repeat(left-np.r_[0,np.cumsum(length)[:-1]],length)+np.arange(int(length.sum()))
    slab=stats[ix];slab_atom=np.repeat(np.arange(len(atom_ids)),length)
    rows=[];geos=[];names=None
    for labels,family in zip(maps,families):
        keep=labels[pairs[:,0]]==labels[pairs[:,1]];changed=keep!=previous;cut=previous&~keep;merge=~previous&keep
        mass=np.bincount(labels,weights=a['mass'][atom_ids])[1:];fraction=mass/mass.sum();entropy=-np.sum(fraction*np.log(np.maximum(fraction,1e-12)))
        values=[np.log1p(len(atom_ids)),np.log1p(len(edges)),np.log1p(mass.sum()),len(mass),entropy,float(fraction.max()),float(fraction.min()),float(np.mean(changed)),float(np.mean(cut)),float(np.mean(merge)),float(np.sum(faces[cut])/faces.sum()),float(np.sum(faces[merge])/faces.sum()),float(np.mean(-np.log(np.clip(np.where(keep,prob,1-prob),1e-6,1))))]
        names=['B_log_atoms','B_log_edges','B_log_mass','B_components','B_entropy','B_largest','B_smallest','B_changed','B_cut','B_merge','B_cut_faces','B_merge_faces','B_energy']
        for fam in ('hierarchy','gaec','mst','geodesic','geodesic_gaec'):values.append(float(family==fam));names.append('B_family_'+fam)
        for state,mask in [('cut',~keep),('merge',keep),('changed',changed)]:
            values.extend(np.mean(base[mask],axis=0) if mask.any() else np.zeros(base.shape[1]));names.extend(f'B_{state}_edge{j}' for j in range(base.shape[1]))
        rows.append(np.asarray(values));geos.append(partition_geometry(labels,slab,slab_atom,faces,pairs))
    features=np.asarray([np.r_[v,v-rows[0],g,g-geos[0]] for v,g in zip(rows,geos)],np.float32)
    names=names+['D'+n for n in names]+[f'M_{j}' for j in range(len(geos[0]))]+[f'DM_{j}' for j in range(len(geos[0]))]
    return maps,families,features,names


def risk_repartition(a,slabs,nz,assets,cfg,progress,check):
    rows,regions=region_pool(a);mapping=a['mapping'].copy()
    if not rows:return mapping,set(),dict(regions=0,actions=0,proposals=0)
    score=assets.predict('region_ranker.txt',np.asarray([r['features'] for r in rows],np.float32))
    selected=rp.select_regions(rows,score,32)[:cfg.get('r_budget',32)]
    budget=str(cfg.get('r_budget',32));margin=assets.json('repartition/calibration.json')['by_budget'][budget]['margin']
    expected=assets.json('repartition/columns.json')['names'];occupied=set();chosen=[];total_proposals=0
    for i,rid in enumerate(selected):
        check();ids,eids=regions[rid]
        maps,families,x,names=partition_features(ids,eids,a,slabs,nz)
        if names!=expected:raise ValueError('重划分特征顺序与正式模型不一致。')
        if len(maps)>1:
            prediction={kind:assets.predict(f'repartition/{kind}.txt',x[1:]) for kind in ('gain','q20','cat','pos')}
            score=.7*prediction['gain']+.3*prediction['q20']-.12*prediction['cat']+.02*(prediction['pos']-.5)
            j=int(np.argmax(score))+1
            if score[j-1]>margin:
                offset=int(mapping.max())+1;mapping[ids]=maps[j]+offset
                occupied.update(rows[rid]['clusters']);chosen.append(dict(region=rid,proposal=j,family=families[j],score=float(score[j-1])))
        total_proposals+=len(maps)
        progress(56+int(13*(i+1)/max(len(selected),1)),'风险感知重划分',f'区域 {i+1}/{len(selected)} · 已采纳 {len(chosen)} 项')
    return mapping,occupied,dict(regions=len(rows),selected_regions=selected,actions=len(chosen),chosen=chosen,proposals=total_proposals,margin=margin)


def stitch_graph(a):
    mapping,atom_edges=a['mapping'],a['edges'];u,w=mapping[atom_edges].T;cross=u!=w
    low=np.minimum(u[cross],w[cross]);high=np.maximum(u[cross],w[cross]);key=(low.astype(np.int64)<<32)|high
    keys,inv=np.unique(key,return_inverse=True);edges=np.column_stack([keys>>32,keys&0xffffffff]).astype(np.int64);n=len(edges)
    if not n:return edges,np.empty((0,34),np.float32),np.bincount(mapping,weights=a['mass'])
    prob=a['prob'][cross].astype(float);faces=a['faces'][cross];count=np.bincount(inv,minlength=n);area=np.bincount(inv,weights=faces,minlength=n)
    mean=np.bincount(inv,weights=prob,minlength=n)/count;weighted=np.bincount(inv,weights=prob*faces,minlength=n)/area
    sd=np.sqrt(np.maximum(np.bincount(inv,weights=prob**2,minlength=n)/count-mean**2,0))
    lo=np.full(n,np.inf);hi=np.full(n,-np.inf);np.minimum.at(lo,inv,prob);np.maximum.at(hi,inv,prob)
    nc=int(mapping.max())+1;mass=np.bincount(mapping,weights=a['mass'],minlength=nc);atom_count=np.bincount(mapping,minlength=nc)
    small=np.where(mass[edges[:,0]]<=mass[edges[:,1]],edges[:,0],edges[:,1]);large=np.where(small==edges[:,0],edges[:,1],edges[:,0])
    degree=np.bincount(edges.ravel(),minlength=nc);contact=np.bincount(edges.ravel(),weights=np.repeat(area,2),minlength=nc)
    best=np.zeros(nc);np.maximum.at(best,edges[:,0],weighted);np.maximum.at(best,edges[:,1],weighted)
    same=~cross;internal=np.bincount(u[same],weights=(1-a['prob'][same])*np.log1p(a['faces'][same]),minlength=nc)
    columns=[np.log1p(mass[small]),np.log1p(mass[large]),mass[small]/np.maximum(mass[large],1),np.log1p(atom_count[small]),np.log1p(atom_count[large]),np.log1p(count),np.log1p(area),mean,weighted,sd,lo,hi,np.log1p(degree[small]),np.log1p(degree[large]),area/np.maximum(contact[small],1),area/np.maximum(contact[large],1),weighted-best[small],weighted-best[large],np.log1p(internal[small]),np.log1p(internal[large])]
    base=a['base'][cross]
    for j in range(base.shape[1]):columns.append(np.bincount(inv,weights=base[:,j],minlength=n)/count)
    return edges,np.column_stack(columns).astype(np.float32),mass


def stitch(a,rmap,occupied,assets):
    edges,x,mass=stitch_graph(a)
    if not len(edges):return rmap,dict(actions=0,candidate_edges=0)
    gain=assets.predict('stitch_gain.txt',x);score=gain*mass[edges[:,0]]*mass[edges[:,1]]
    margin=assets.json('stitch_calibration.json')['selected']['margin']
    busy=set(occupied);chosen=[]
    if margin is not None:
        # Match first, then apply the calibrated margin, exactly as historical t.greedy/selected.
        for j in np.argsort(-score,kind='stable'):
            if score[j]<=0:continue
            u,v=map(int,edges[j])
            if u in busy or v in busy:continue
            busy.update((u,v))
            if score[j]>margin:chosen.append(int(j))
    remap=np.arange(int(rmap.max())+1,dtype=np.int32)
    selected=edges[chosen];remap[selected[:,1]]=selected[:,0]
    return remap[rmap],dict(actions=len(chosen),candidate_edges=len(edges),margin=margin,edge_indices=chosen)


class AtomReader:
    def __init__(self,data):self.data=data;self.shape=data.shape
    def frame(self,z):return self.data[z]
    def max(self):return int(self.data.max())


def z_face_counts(atoms,edges):
    keys=(edges[:,0].astype(np.int64)<<32)|edges[:,1];counts=np.zeros(len(edges),float)
    for z in range(1,len(atoms)):
        a,b=atoms[z-1],atoms[z];mask=(a>0)&(b>0)&(a!=b)
        pair=(np.minimum(a[mask],b[mask]).astype(np.int64)<<32)|np.maximum(a[mask],b[mask])
        unique,n=np.unique(pair,return_counts=True);counts[np.searchsorted(keys,unique)]+=n
    return counts


def track_similarity(features,assets):
    if not len(features):return np.empty(0)
    with np.load(assets.path('f_morphology_bank.npz'),allow_pickle=False) as bank:
        query=(features-bank['center'])/bank['scale'];distance=cKDTree(bank['anchors']).query(query,k=1,workers=assets.threads)[0]
        loo=bank['loo'];return np.clip((len(loo)-np.searchsorted(loo,distance,side='left'))/len(loo),0,1)


def pf(mapping,mass,edges,faces,zfaces,ext,na,similarity,threshold):
    volume,acount,pure=post.current_state(mapping,mass,na)
    score=np.bincount(mapping[na+1:],weights=mass[na+1:]*similarity,minlength=len(volume))/np.maximum(volume,1)
    candidates=set(np.flatnonzero(pure&(score<threshold)));retained=set(np.flatnonzero(pure))-candidates
    ce,area=post.contacts(mapping,edges,faces);ze,za=post.contacts(mapping,edges,zfaces)
    exterior=set(np.unique(mapping[ext]))-{0};adj=defaultdict(list)
    for a,b in ce:
        if a in candidates and b in candidates:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
    deleted=candidates&exterior;queue=list(deleted)
    for a in queue:
        for b in adj[a]:
            if b not in deleted:deleted.add(b);queue.append(b)
    protected=candidates-deleted;surviving=retained|protected
    ff={(int(a),int(b)):int(w) for (a,b),w in zip(ce,area) if a in surviving and b in surviving};fa={}
    for (a,b),w in zip(ze,za):
        if a in surviving and acount[b]>0:fa[(int(a),int(b))]=int(w)
        elif b in surviving and acount[a]>0:fa[(int(b),int(a))]=int(w)
    graph=PFComponentGraph(surviving,protected,retained,ff,fa);graph.run()
    lut=np.arange(len(volume),dtype=np.int32)
    for c in deleted:lut[c]=0
    for c in surviving:
        root=graph.track_to_root[c];terminal=graph.terminal[root];lut[c]=min(graph.members[root]) if terminal is None else terminal[1]
    return lut[mapping],dict(candidates=len(candidates),deleted=len(deleted),protected=len(protected),merged=int(np.sum((lut>0)&(lut!=np.arange(len(lut))))))


def postprocess(atoms,mapping,a,na,f_features,assets,cfg):
    reader=AtomReader(atoms);identity=np.arange(len(mapping),dtype=np.int32)
    ext=post.exterior_for(reader,identity)
    zfaces=z_face_counts(atoms,a['edges'])
    similarity=track_similarity(f_features,assets)
    after,stats=pf(mapping,a['mass'],a['edges'],a['faces'],zfaces,ext,na,similarity,cfg.get('pf_threshold',.10))
    if stats['deleted']:ext=post.exterior_for(reader,after)
    # P-A's scale is undefined if no clean interior current cell exists (e.g. a
    # one-slice input). Explicitly record that case instead of inventing a median.
    try:result,pa_stats,actions=post.pa(after,a['mass'],a['edges'],a['faces'],ext,na)
    except RuntimeError as e:
        if 'No clean interior' not in str(e):raise
        result=after;pa_stats=dict(pa_candidates=0,pa_merged=0,pa_deleted=0,reason='no clean interior A-containing cells; scale undefined');actions=[]
    return result,dict(PF=stats,PA=pa_stats,pa_actions=actions)
