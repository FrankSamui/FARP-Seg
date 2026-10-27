"""Prediction-only discovery for the original database directory layout."""
from pathlib import Path
import json
import hashlib

DATABASE = Path(__file__).resolve().parent / 'resources/sample'
SOFTWARE = Path(__file__).resolve().parent
FOLDERS = SOFTWARE/'sample_folders.json'
VOLUME_SUFFIXES={'.tif','.tiff','.npy','.npz','.h5','.hdf5'}


def same_input(first,second):
    first,second=Path(first),Path(second)
    if not first.is_file() or not second.is_file():return False
    if first.samefile(second):return True
    if first.name.casefold()!=second.name.casefold() or first.stat().st_size!=second.stat().st_size:return False
    def digest(path):
        result=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''):result.update(chunk)
        return result.digest()
    return digest(first)==digest(second)


def sample_roots():
    roots=[SOFTWARE/'resources',DATABASE]
    try:roots.extend(Path(p) for p in json.loads(FOLDERS.read_text(encoding='utf-8')))
    except (OSError,ValueError,TypeError):pass
    return list(dict.fromkeys(p.resolve() for p in roots if p.is_dir()))


def add_sample_folder(folder):
    folder=Path(folder).resolve()
    if not folder.is_dir():raise ValueError('Sample folder does not exist.')
    try:folders=json.loads(FOLDERS.read_text(encoding='utf-8'))
    except (OSError,ValueError):folders=[]
    if str(folder) not in folders:folders.append(str(folder))
    FOLDERS.write_text(json.dumps(folders,ensure_ascii=False,indent=2),encoding='utf-8')


def available_samples(roots=None):
    """Scan inputs only; never interpret references, results or model assets as samples."""
    found={};identities=set();names={}
    for root in sample_roots() if roots is None else map(Path,roots):
        if not root.is_dir():continue
        for path in root.rglob('*'):
            if path.suffix.lower() not in VOLUME_SUFFIXES or not path.is_file():continue
            relative=path.relative_to(root)
            if any(part.lower() in ('models','outputs','figures','.ipynb_checkpoints','__pycache__') for part in relative.parts[:-1]):continue
            name=path.stem.lower()
            if any(word in name for word in ('segmented','segment','reference','label','mask','preprocessed')):continue
            if name in ('atoms','viewer_atoms','stage_cluster_maps','review'):continue
            if path.suffix.lower() in ('.npy','.npz') and root.resolve()==(SOFTWARE/'resources').resolve() and 'prediction' not in name:continue
            stats=path.stat();identity=(stats.st_dev,stats.st_ino) if stats.st_ino else str(path.resolve()).casefold()
            if identity in identities:continue
            identities.add(identity)
            key=(path.name.casefold(),stats.st_size)
            if any(same_input(path,other) for other in names.get(key,[])):continue
            names.setdefault(key,[]).append(path)
            found[str(path.resolve()).casefold()]=path.resolve()
    return sorted(found.values(),key=lambda p:str(p).casefold())

def is_prediction(path):
    path = Path(path)
    return path.suffix.lower() in ('.tif', '.tiff') and path.stem.lower().endswith(('_prediction','_predictions'))

def predictions(root=DATABASE):
    root = Path(root)
    return sorted((p for p in root.rglob('*') if p.is_file() and is_prediction(p)
                   and '.ipynb_checkpoints' not in p.parts), key=lambda p:str(p).lower()) if root.exists() else []

def resolve_input(path):
    path = Path(path)
    if path.is_dir():
        matches = available_samples([path])
        if len(matches) != 1:
            raise ValueError(f'该目录包含 {len(matches)} 个 prediction TIFF，请选择单个样本目录或直接选择 prediction 文件。')
        return matches[0]
    if path.suffix.lower() == '.csv':
        raise ValueError('无需导入 CSV。请选择同一样本的 *_Prediction.tif / .tiff。')
    if path.stem.lower().endswith('_segmented_cells'):
        raise ValueError('这是数据库参考标签。请选择同一样本的 *_Prediction.tif / .tiff。')
    return path

def database_defaults(path):
    path=Path(path)
    if not is_prediction(path):return {}
    volume_id='/'.join(path.parent.parts[-2:])
    manifest=Path(__file__).parent/'resources/models/manifest.json'
    folds=json.loads(manifest.read_text(encoding='utf-8'))['folds']
    if volume_id not in folds:return {}
    return dict(volume_id=volume_id,fold=folds[volume_id],xy_step=2,input_is_crop=False,
                defaults_source='official held-out model manifest')
