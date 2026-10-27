"""Database import checks: original TIFF alone, mixed naming, directory selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import tifffile
from inputs import predictions, resolve_input, database_defaults
from engine import read_volume

with TemporaryDirectory() as temp:
    root=Path(temp)
    for group,volume,name in [('1-I_1-II','396_6x','396_predictions.tif'),('3-VI','555','555_Prediction.tiff')]:
        folder=root/group/volume;folder.mkdir(parents=True)
        path=folder/name
        array=np.arange(3*12*14,dtype=np.uint16).reshape(3,12,14)
        tifffile.imwrite(path,array,photometric='minisblack',metadata={'axes':'ZYX'})
        assert resolve_input(folder)==path
        np.testing.assert_array_equal(read_volume(resolve_input(folder)),array)
    assert len(predictions(root))==2
    # CSV and reference labels are not required and must never be selected by discovery.
    (folder/'555_Attributes.csv').write_text('unused',encoding='utf-8')
    tifffile.imwrite(folder/'555_Segmented_Cells.tif',array,photometric='minisblack')
    assert len(predictions(root))==2 and resolve_input(folder)==path
    for rejected in [folder/'555_Attributes.csv',folder/'555_Segmented_Cells.tif',root]:
        try:resolve_input(rejected)
        except ValueError:pass
        else:raise AssertionError(f'Should reject: {rejected}')
    assert database_defaults(root/'1-I_1-II/397_4x/397_4x_predictions.tif')['fold']==1
    assert database_defaults(root/'3-VI/555/555_Prediction.tif')['fold']==0
    assert database_defaults(root/'new/new/Prediction.tif')=={}
print('PASS: original TIFF-only import, singular/plural/case names, sample folders, CSV/reference exclusion, ambiguous folder rejection')
