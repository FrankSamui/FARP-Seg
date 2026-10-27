"""Small reversible UI catalogue; switching language never rebuilds the workspace."""
import re
from string import Formatter
from PyQt5 import QtCore, QtWidgets as W

LANGUAGE='en'
MESSAGES={
    '取消前处理':'Cancel preprocessing', '取消分割':'Cancel segmentation', '正在取消…':'Cancelling…',
    '可用内存可能不足，任务将继续运行。若实际内存不足，任务可能失败。':'Available memory may be insufficient. The task will continue and may fail if memory runs out.',
    '导出全部视图':'Export all views', '全部视图已导出':'All views exported',
    '打开':'Open', '没有已准备的审阅结果。':'No prepared review results for this sample.',
    '请选择已有审阅结果的样本':'Select a sample with prepared review results',
    '导入文件夹':'Import folder', '选择样本':'Select sample',
    '文件夹内没有可用样本。':'No usable samples found in this folder.',
    '前处理（可选）':'Preprocessing (optional)',
    '在预测较模糊暗淡时开启，预计花费10分钟':'Enable when predictions are blurry or dim. Estimated time: 10 minutes.',
    '前处理并行数':'Preprocessing workers',
    '运行前处理':'Run preprocessing', '使用前处理结果':'Use preprocessed input',
    '请先运行前处理。':'Run preprocessing first.',
    '输入设置已变化，请重新运行前处理。':'Input settings changed. Run preprocessing again.',
    '正在前处理…':'Preprocessing…', '前处理完成':'Preprocessing complete',
    '前处理':'Preprocessing', '增强膜信号':'Enhance membrane signal',
    '原始输入':'Original input', '前处理结果':'Preprocessed input',
    '正在前处理 · {done} / {total} 层 · {percent}%':'Preprocessing · {done} / {total} slices · {percent}%',
    '原始 atom':'Atoms', '无 F（重分割前）':'Without F · before repartition',
    '有 F（重分割前）':'With F · before repartition', 'P 前':'Before P', 'P 后':'After P',
    '参数面板':'Controls', '导出五个视图':'Export all five', '全屏（F11）':'Full screen (F11)',
    '生成阶段视图':'Generate stage views', '导出配图 ↗':'Export figure ↗',
    '打开预定样本':'Open selected sample', '仅导出可见区域':'Visible region only',
    '阶段视图已就绪':'Stage views ready', '请生成阶段视图':'Generate stage views to begin',
    '导出配图':'Export figure', '配图已导出':'Figure exported', '五个视图已导出':'All five views exported',
    '导出当前视图':'Export current stage',
    '阶段预览':'Stage view',
    'FARP-Seg · 结果查看器':'FARP-Seg · Results Viewer',
    'FARP-Seg · 结果查看器 · {filename}':'FARP-Seg · Results Viewer · {filename}',
    '输入数据':'Input', '分割参数':'Parameters', '清理强度':'Cleanup strength',
    '高级设置':'Advanced settings', '更多参数':'Additional parameters',
    '背景伪细胞处理强度':'Background cleanup strength',
    '边缘伪细胞处理强度':'Edge cleanup strength',
    '数值越大，对可疑背景区域的处理越强。':'Higher values apply stronger cleanup to suspected background regions.',
    '＋  导入 prediction TIFF':'＋  Import prediction', '从数据库选择样本':'Browse samples',
    '打开示例':'Open example', '尚未导入文件':'No file selected', '恢复推荐':'Reset defaults',
    '开始分割':'Run segmentation', '取消运行':'Cancel', '准备就绪':'Ready',
    '切片工作台':'Slice viewer', '适应窗口':'Fit view', '导出 segment ↗':'Export segmentation ↗',
    '彩色实例':'Instance colors', '边界叠加':'Boundary overlay', '半透明叠加':'Blended overlay',
    '预测':'Prediction', '分割':'Segmentation',
    '导入 prediction 或打开示例':'Import a prediction', '运行后在这里查看分割结果':'Segmentation preview',
    '等待分割结果':'Segmentation preview', 'XY 采样':'XY sampling', '推理模型':'Model',
    '原始尺寸  /  1×':'Original / 1×', '推荐采样  /  XY÷2':'Standard / XY÷2',
    '快速预览  /  XY÷4':'Preview / XY÷4',
    'A14 · 模型 {id}':'A14 · Model {id}', '合并阈值':'Merge threshold',
    '完整流程':'Pipeline', 'F 补区 / 跨层连接':'F filling / Z linking',
    '风险感知重划分':'Risk-aware repartition', '残余碎片连接':'Fragment stitching',
    'P-F / P-A 清理':'P-F / P-A cleanup', '重划分预算':'Region budget', '{count} 个区域':'{count} regions',
    '启用外缘清理':'Enable exterior cleanup', '条带占比':'Band fraction', 'Z 局部窗口':'Z window',
    '薄片半径':'Thin-region radius', 'HDF5 数据集':'HDF5 dataset',
    'HDF5 数据集（自动识别）':'Auto-detect', '种子 Z 间隔':'Z seed spacing',
    '种子 XY 间隔':'XY seed spacing', '响应值范围':'Response range',
    '推理时翻转 Y，输出还原':'Flip Y during inference',
    '输入为人工裁剪，保护裁剪边界':'Protect crop boundaries',
    'CPU 线程':'CPU threads', 'F 颈部阈值':'F neck threshold',
    'F 射线闭合比例':'F enclosure ratio', 'P-F 形态阈值':'P-F shape threshold',
    '有多个 HDF5 数据集时，先填写名称再导入。':'Specify a dataset before importing an HDF5 file with multiple datasets.',
    '正式三折模型。未知新样本默认模型 1；已知训练体积请使用其 held-out 折。':'Known samples use their held-out model. New samples default to Model 1.',
    '使用对应模型在验证集选定的阈值；调低会合并更多。':'Lower values merge more regions.',
    '推荐 6；轻 4 / 强 8。单位为处理网格 XY 像素。0 关闭清理。':'Default 6; light 4; strong 8. Measured in processing-grid pixels. Set 0 to disable.',
    '滚轮缩放；拖动平移；Ctrl + 滚轮切层。':'Scroll to zoom; drag to pan; Ctrl + scroll to change slices.',
    '导入 prediction':'Import prediction', '导出 segment':'Export segmentation',
    'Prediction TIFF (*.tif *.tiff);;其他体积格式 (*.npy *.npz *.h5 *.hdf5)':'Prediction TIFF (*.tif *.tiff);;Other volumes (*.npy *.npz *.h5 *.hdf5)',
    '数据库 · 选择 prediction':'Select a sample', '样本 / prediction':'Sample / prediction',
    '文件大小':'Size', '{count} 个样本':'{count} samples', '取消':'Cancel',
    '正在读取文件…':'Loading…', '已导入':'Loaded', '正在导出…':'Exporting…', '已导出':'Exported',
    '启动本机推理…':'Starting segmentation…', '正在分割 · {percent}%':'Segmenting · {percent}%',
    '参数已修改 · 当前预览仍为上次结果，重新运行后生效':'Parameters changed · Run again to update the preview',
    '预计算示例 · {count} 个实例':'Precomputed example · {count} instances',
    '完成 · {count} 个实例 · {seconds} 秒':'Completed · {count} instances · {seconds} s',
    ' · 实例 {id}':' · Instance {id}',
    '已取消 · 上次结果仍可查看':'Cancelled · Previous result retained',
    '操作未完成 · 请查看提示':'Operation incomplete',
    '正在读写文件，请稍候完成再关闭。':'Please wait for file operations to finish before closing.',
    '默认示例清单中的文件名无效。':'The example manifest contains an invalid filename.',
    '示例尚未下载。请导入自己的 prediction，或将真实示例放入 resources/sample。':'The example is unavailable. Import a prediction instead.',
    '示例结果的尺寸、标签类型或来源说明不匹配。':'The example result does not match its input or metadata.',
    'Z 局部窗口需为奇数，例如推荐的 5 层。':'Z window must be odd. The default is 5.',
    '缺少 A14 模型。请将模型放入 resources/models 后重试。':'The A14 model is missing from resources/models.',
    '缺少完整流程模型资源，请先安装 resources/models/full 下的模型与校准参数。':'Pipeline models or calibration files are missing from resources/models/full.',
    '预计内存不足。请提高 XY 采样倍数或裁剪输入后重试。':'Insufficient memory. Increase XY sampling or import a smaller crop.',
    '推理进程无法启动。请确认 Python 和依赖已安装。':'Unable to start segmentation. Check the Python environment and dependencies.',
    '推理退出，退出码 {code}':'Segmentation exited with code {code}',
    '输出不能覆盖输入 prediction。':'The output cannot overwrite the input prediction.',
    '无需导入 CSV。请选择同一样本的 *_Prediction.tif / .tiff。':'Select the prediction TIFF for this sample.',
    '这是数据库参考标签。请选择同一样本的 *_Prediction.tif / .tiff。':'This is a reference label image. Select the prediction TIFF.',
    '该目录包含 {count} 个 prediction TIFF，请选择单个样本目录或直接选择 prediction 文件。':'This folder contains {count} prediction TIFFs. Select a single sample or a prediction file.',
    'TIFF 含颜色/通道轴，请先选择单个 prediction 通道后导入。':'Select a single grayscale prediction channel before importing this TIFF.',
    '支持 TIFF、NPY、NPZ、HDF5 文件。':'Supported formats: TIFF, NPY, NPZ and HDF5.',
    '输入为空或 XY 尺寸小于 2。':'The input is empty or its XY dimensions are smaller than 2.',
    '需要灰度 Z×Y×X 体积，实际 shape={shape}, dtype={dtype}。多通道数据请先选择通道。':'Expected a grayscale Z×Y×X volume; got shape={shape}, dtype={dtype}. Select a single channel for multi-channel data.',
    'HDF5 含多个数据集，请填写数据集名称：{names}':'Multiple HDF5 datasets found. Specify one of: {names}',
}

def set_language(language):
    global LANGUAGE
    LANGUAGE='zh' if language=='zh' else 'en'

class ChineseDialogs(QtCore.QTranslator):
    """Fallback for Windows PyQt distributions without qtbase_zh_CN.qm."""
    labels={
        'Look in:':'查找范围：','&Look in:':'查找范围：','File name:':'文件名：',
        'File &name:':'文件名：','&File name:':'文件名：','Files of type:':'文件类型：',
        'File type:':'文件类型：','Save in:':'保存位置：','Directory:':'目录：',
        'Open':'打开','&Open':'打开','Save':'保存','&Save':'保存',
        'Cancel':'取消','&Cancel':'取消','Close':'关闭','&Close':'关闭','OK':'确定',
        '&Yes':'是','&No':'否','Yes':'是','No':'否','Apply':'应用','Reset':'重置',
        'Help':'帮助','Show Details...':'显示详情…','Hide Details...':'隐藏详情…',
        'Name':'名称','Size':'大小','Type':'类型','Date Modified':'修改日期',
        'Date modified':'修改日期','Folder':'文件夹','File Folder':'文件夹',
        'Computer':'计算机','My Computer':'计算机','New Folder':'新建文件夹',
        'Back':'后退','Forward':'前进','Parent Directory':'上级目录',
        'Create New Folder':'新建文件夹','List View':'列表视图','Detail View':'详细视图',
        'Show hidden files':'显示隐藏文件','Show Hidden Files':'显示隐藏文件',
        'Rename':'重命名','&Rename':'重命名','Delete':'删除','&Delete':'删除',
        'All Files (*)':'所有文件 (*)','Select':'选择','Choose':'选择',
        '&Undo':'撤销','&Redo':'重做','Cu&t':'剪切','&Copy':'复制','&Paste':'粘贴',
        'Select All':'全选','Select &All':'全选','Step up':'增加','Step down':'减少',
    }

    def isEmpty(self):return False

    def translate(self,context,sourceText,disambiguation=None,n=-1):
        return self.labels.get(sourceText,sourceText)

def _pattern(template):
    parts=[]
    for literal,name,_,_ in Formatter().parse(template):
        parts.append(re.escape(literal))
        if name is not None:parts.append(f'(?P<{name}>.+?)')
    return re.compile(''.join(parts),re.DOTALL)

_REVERSE={value:key for key,value in MESSAGES.items()}
_TEMPLATES=[(cn,en,_pattern(cn),_pattern(en)) for cn,en in MESSAGES.items() if '{' in cn]

def tr(text,**values):
    text=str(text)
    if LANGUAGE=='en':result=MESSAGES.get(text,text)
    else:result=_REVERSE.get(text,text)
    if result==text:
        for cn,en,cn_pattern,en_pattern in _TEMPLATES:
            match=(cn_pattern if LANGUAGE=='en' else en_pattern).fullmatch(text)
            if match:
                result=(en if LANGUAGE=='en' else cn).format(**match.groupdict());break
    if result==text and text.strip()!=text:
        trimmed=text.strip();translated=tr(trimmed)
        if translated!=trimmed:result=text[:len(text)-len(text.lstrip())]+translated+text[len(text.rstrip()):]
    return result.format(**values) if values else result

def translate_widgets(root):
    """Update text only, preserving widget identities, values and signal connections."""
    for widget in [root]+root.findChildren(W.QWidget):
        widget.setToolTip(tr(widget.toolTip()))
        if isinstance(widget,(W.QLabel,W.QAbstractButton)):widget.setText(tr(widget.text()))
        if isinstance(widget,W.QGroupBox):widget.setTitle(tr(widget.title()))
        if isinstance(widget,W.QLineEdit):widget.setPlaceholderText(tr(widget.placeholderText()))
        if isinstance(widget,W.QComboBox):
            blocked=widget.blockSignals(True)
            for index in range(widget.count()):widget.setItemText(index,tr(widget.itemText(index)))
            widget.blockSignals(blocked)
        if isinstance(widget,W.QTreeWidget):
            for column in range(widget.columnCount()):
                widget.headerItem().setText(column,tr(widget.headerItem().text(column)))
        if isinstance(widget,(W.QDialog,W.QMainWindow)):widget.setWindowTitle(tr(widget.windowTitle()))
        if isinstance(widget,W.QStatusBar):widget.showMessage(tr(widget.currentMessage()))
    for action in root.findChildren(W.QAction):
        action.setText(tr(action.text()));action.setToolTip(tr(action.toolTip()))
