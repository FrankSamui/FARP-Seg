from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback
import uuid
from collections import OrderedDict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets as W
from engine import DEFAULTS, ROOT, read_volume, export_volume
from inputs import DATABASE, predictions, resolve_input, database_defaults, available_samples, add_sample_folder
from i18n import tr, set_language, translate_widgets, ChineseDialogs

STYLE = """QWidget { background:#101216; color:#d5d9e2; font-family:'Microsoft YaHei UI','Segoe UI'; font-size:12px; }
QMainWindow { background:#101216; }
QFrame#sidebar { background:#171a20; border:1px solid #292d36; border-radius:12px; }
QLabel { background:transparent; }
QLabel#brand { color:#f5f6fa; font-size:25px; font-weight:700; }
QLabel#muted { color:#8d95a5; font-size:11px; }
QLabel#section { color:#a5b0c2; font-size:11px; font-weight:600; }
QLabel#badge { background:#173229; color:#83d6ae; padding:5px 10px; border-radius:9px; }
QLabel#notice { color:#aeb6c7; background:#1f252f; border-radius:6px; padding:8px; font-size:11px; }
QPushButton { background:#262b35; border:1px solid #363c49; border-radius:7px; padding:8px 13px; color:#e0e5ee; }
QPushButton:hover { background:#343b49; border-color:#727f97; }
QPushButton:pressed { background:#1f2530; }
QPushButton:disabled { color:#606777; background:#20242c; border-color:#292e38; }
QPushButton#primary { background:#8bddbb; color:#10281f; border:0; font-weight:700; padding:12px; }
QPushButton#primary:hover { background:#a3ebcd; }
QPushButton#primary:disabled { background:#29483d; color:#698a7d; }
QPushButton#subtle { border:0; background:transparent; color:#92a3bd; padding:5px; }
QToolButton { border:0; background:transparent; color:#a5b0c2; padding:10px 0; text-align:left; }
QToolButton:hover { color:#d5f0e4; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#101318; border:1px solid #343b47; border-radius:5px; padding:6px; min-height:19px; selection-background-color:#44645b; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color:#7bc4a7; }
QComboBox QAbstractItemView { background:#1d222b; selection-background-color:#365448; }
QCheckBox { spacing:7px; background:transparent; }
QCheckBox::indicator { width:15px; height:15px; border:1px solid #5a6576; border-radius:3px; background:#11151b; }
QCheckBox::indicator:checked { background:#8bddbb; border-color:#8bddbb; image:none; }
QSlider::groove:horizontal { height:4px; background:#303744; border-radius:2px; }
QSlider::sub-page:horizontal { background:#7dc4aa; border-radius:2px; }
QSlider::handle:horizontal { width:12px; margin:-5px 0; background:#c5f4df; border-radius:6px; }
QProgressBar { background:#232936; border:0; border-radius:3px; height:5px; color:transparent; }
QProgressBar::chunk { background:#8bddbb; border-radius:3px; }
QScrollArea { border:0; background:transparent; }
QScrollBar:vertical { background:#181c23; width:7px; }
QScrollBar::handle:vertical { background:#3b4453; min-height:30px; border-radius:3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QSplitter::handle { background:#101216; width:10px; }
QToolTip { background:#252d39; color:#e3e9f2; border:1px solid #546175; padding:6px; }
QStatusBar { color:#8d95a5; }
"""


class Task(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent); self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e))


class ImageView(W.QGraphicsView):
    changed = QtCore.pyqtSignal(object)
    pixel = QtCore.pyqtSignal(int, int)
    sliceRequested = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setScene(W.QGraphicsScene(self))
        self.item = self.scene().addPixmap(QtGui.QPixmap())
        self.setBackgroundBrush(QtGui.QColor('#090b0e'))
        self.setFrameShape(W.QFrame.NoFrame)
        self.setDragMode(W.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(W.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(W.QGraphicsView.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.placeholder = self.scene().addText('')
        self.placeholder.setDefaultTextColor(QtGui.QColor('#616c7e'))
        self.placeholder.setFont(QtGui.QFont('Microsoft YaHei UI', 13))
        self.fitted = False

    def empty(self, message):
        self.item.setPixmap(QtGui.QPixmap())
        self.placeholder.setData(0,message)
        self.placeholder.setPlainText(tr(message))
        self.scene().setSceneRect(0, 0, 500, 500)
        self.placeholder.setPos(50, 200)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def set_rgb(self, rgb, fit=False):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[:2]
        q = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.placeholder.setPlainText('')
        self.item.setPixmap(QtGui.QPixmap.fromImage(q))
        self.scene().setSceneRect(0, 0, w, h)
        if fit or not self.fitted:
            self.fit_image()

    def fit_image(self):
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, QtCore.Qt.KeepAspectRatio)
            self.fitted = True
            self.changed.emit(self)

    def wheelEvent(self, e):
        if e.modifiers() & QtCore.Qt.ControlModifier:
            self.sliceRequested.emit(1 if e.angleDelta().y() > 0 else -1)
        else:
            amount = 1.18 if e.angleDelta().y() > 0 else 1/1.18
            scale = self.transform().m11() * amount
            if .03 < scale < 40:
                self.scale(amount, amount)
                self.changed.emit(self)
        e.accept()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        p = self.mapToScene(e.pos())
        self.pixel.emit(int(p.x()), int(p.y()))
        if e.buttons() & QtCore.Qt.LeftButton:
            self.changed.emit(self)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fitted:
            self.fit_image()


def text_label(text, name=None):
    label = W.QLabel(text)
    if name: label.setObjectName(name)
    return label


def button(text, callback, name=None):
    b = W.QPushButton(text); b.clicked.connect(callback)
    if name: b.setObjectName(name)
    return b


def colors(labels):
    # Same label always gets the same color across slices, with stronger mixing than
    # the old low-byte-only palette (which repeated every 256 labels).
    x = labels.astype(np.uint64)
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xffffffff
    x = ((x ^ (x >> 16)) * 0x45d9f3b) & 0xffffffff
    x ^= x >> 16
    rgb = np.stack([65 + ((x >> shift) & 191) for shift in (0,8,16)], axis=-1).astype(np.uint8)
    rgb[labels == 0] = 0
    return rgb


def sample_metadata(path):
    path=Path(path)
    if path.parent.resolve()!=(ROOT/'resources/sample').resolve():return database_defaults(path)
    metadata=path.with_suffix('.json')
    if not metadata.exists():return {}
    try:
        data=json.loads(metadata.read_text(encoding='utf-8'))
        return data if isinstance(data,dict) else {}
    except (ValueError,OSError):return {}


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FARP-Seg')
        self.resize(1440, 920); self.setMinimumSize(1100, 730)
        self.raw = None; self.segment = None; self.path = None; self.meta = None
        self.demo_meta={}; self.processed=None
        self.prepared_path=None;self.prepared_metadata={};self.process_kind='segmentation'
        self.process = None; self.jobs = []; self.buffer = ''; self.errors = ''
        self.cache = OrderedDict(); self.syncing = False; self.output_path = None
        self.setAcceptDrops(True)
        central = W.QWidget(); self.setCentralWidget(central)
        outer = W.QVBoxLayout(central); outer.setContentsMargins(24,18,24,12); outer.setSpacing(16)
        header = W.QHBoxLayout()
        title_box = W.QVBoxLayout(); title_box.setSpacing(3)
        title_box.addWidget(text_label('FARP-Seg', 'brand'))
        header.addLayout(title_box); header.addStretch()
        self.language=W.QComboBox();self.language.addItem('English','en');self.language.addItem('中文','zh')
        self.language.setFixedWidth(112);self.language.setToolTip('Language / 语言');header.addWidget(self.language)
        outer.addLayout(header)
        content = W.QHBoxLayout(); content.setSpacing(16)
        sidebar = W.QFrame(); sidebar.setObjectName('sidebar'); sidebar.setFixedWidth(300)
        side_outer = W.QVBoxLayout(sidebar); side_outer.setContentsMargins(0,0,0,0)
        scroll = W.QScrollArea(); scroll.setWidgetResizable(True)
        panel = W.QWidget(); panel.setStyleSheet('background:transparent;')
        side = W.QVBoxLayout(panel); side.setContentsMargins(18,18,18,18); side.setSpacing(12)
        side.addWidget(text_label('输入数据', 'section'))
        self.folder_button=button('导入文件夹',self.choose_folder);side.addWidget(self.folder_button)
        self.database_button = button('选择样本', self.choose_database)
        side.addWidget(self.database_button)
        self.file_label = text_label('尚未导入文件', 'muted'); self.file_label.setWordWrap(True); side.addWidget(self.file_label)
        self.dataset = W.QLineEdit(); self.dataset.setPlaceholderText(tr('HDF5 数据集（自动识别）'))
        self.dataset.setToolTip('有多个 HDF5 数据集时，先填写名称再导入。')
        self.separator(side)
        side.addWidget(text_label('前处理（可选）', 'section'))
        preprocess_note=text_label('在预测较模糊暗淡时开启，预计花费10分钟','muted')
        preprocess_note.setWordWrap(True);side.addWidget(preprocess_note)
        self.preprocess_button=button('运行前处理',self.run_preprocessing);side.addWidget(self.preprocess_button)
        self.preprocessing_enabled=W.QCheckBox('使用前处理结果');side.addWidget(self.preprocessing_enabled)
        self.preprocess_progress=W.QProgressBar();self.preprocess_progress.setRange(0,100)
        self.preprocess_progress.setFixedHeight(18)
        self.preprocess_progress.setStyleSheet('QProgressBar {color:#d5d9e2;height:18px;text-align:center;} QProgressBar::chunk {background:#365e50;}')
        self.preprocess_progress.setFormat('%p%');self.preprocess_progress.hide();side.addWidget(self.preprocess_progress)
        self.preprocessing_enabled.toggled.connect(self.select_preprocessing)
        self.separator(side)
        row = W.QHBoxLayout(); row.addWidget(text_label('分割参数', 'section')); row.addStretch()
        self.reset_button = button('恢复推荐', self.reset_parameters, 'subtle'); row.addWidget(self.reset_button); side.addLayout(row)
        self.width = self.integer(0,24)
        self.width.setToolTip('推荐 6；轻 4 / 强 8。单位为处理网格 XY 像素。0 关闭清理。')
        self.pf_threshold=self.number(0,1,.01,2)
        self.pf_threshold.setToolTip('数值越大，对可疑背景区域的处理越强。')
        primary=W.QFormLayout();primary.setHorizontalSpacing(10);primary.setVerticalSpacing(14)
        for caption,control in [('背景伪细胞处理强度',self.pf_threshold),('边缘伪细胞处理强度',self.width)]:
            label=text_label(caption);label.setWordWrap(True);label.setFixedWidth(150)
            control.setMinimumWidth(80);control.setMaximumWidth(94);primary.addRow(label,control)
        side.addLayout(primary)
        self.advanced_toggle=W.QToolButton();self.advanced_toggle.setText(tr('高级设置'))
        self.advanced_toggle.setCheckable(True);self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.RightArrow);self.advanced_toggle.setSizePolicy(W.QSizePolicy.Expanding,W.QSizePolicy.Fixed);side.addWidget(self.advanced_toggle)
        self.advanced_panel=W.QWidget();self.advanced_panel.setVisible(False)
        advanced_layout=W.QVBoxLayout(self.advanced_panel);advanced_layout.setContentsMargins(0,8,0,0);advanced_layout.setSpacing(14)
        def show_advanced(checked):
            self.advanced_panel.setVisible(checked)
            self.advanced_toggle.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.advanced_toggle.toggled.connect(show_advanced)
        side.addWidget(self.advanced_panel)
        self.controls = {}
        self.form = W.QFormLayout(); self.form.setHorizontalSpacing(10); self.form.setVerticalSpacing(10)
        self.xy = W.QComboBox()
        for title, value in [('原始尺寸  /  1×',1),('推荐采样  /  XY÷2',2),('快速预览  /  XY÷4',4)]: self.xy.addItem(title,value)
        self.form.addRow('XY 采样',self.xy)
        self.fold = W.QComboBox()
        for i in range(3): self.fold.addItem(f'A14 · 模型 {i+1}',i)
        self.fold.setToolTip('正式三折模型。未知新样本默认模型 1；已知训练体积请使用其 held-out 折。')
        self.form.addRow('推理模型', self.fold)
        self.threshold = self.number(.5,.9999,.001,3)
        self.threshold.setToolTip('使用对应模型在验证集选定的阈值；调低会合并更多。')
        self.form.addRow('合并阈值',self.threshold)
        advanced_layout.addLayout(self.form)
        flow=W.QGroupBox('完整流程'); ff=W.QFormLayout(flow)
        self.f_enabled=W.QCheckBox('F 补区 / 跨层连接');ff.addRow(self.f_enabled)
        self.r_enabled=W.QCheckBox('风险感知重划分');ff.addRow(self.r_enabled)
        self.r_budget=W.QComboBox()
        for k in (8,16,32):self.r_budget.addItem(f'{k} 个区域',k)
        ff.addRow('重划分预算',self.r_budget)
        self.stitch_enabled=W.QCheckBox('残余碎片连接');ff.addRow(self.stitch_enabled)
        self.post_enabled=W.QCheckBox('P-F / P-A 清理');ff.addRow(self.post_enabled)
        advanced_layout.addWidget(flow)
        self.cleanup = W.QCheckBox('启用外缘清理');advanced_layout.addWidget(self.cleanup)
        cleanup_form = W.QFormLayout(); cleanup_form.setVerticalSpacing(10)
        self.fraction = self.number(.1,1,.05,2); cleanup_form.addRow('条带占比',self.fraction)
        self.window = self.integer(1,15,2); cleanup_form.addRow('Z 局部窗口',self.window)
        self.radius = self.number(0,10,.5,1); cleanup_form.addRow('薄片半径',self.radius)
        advanced_layout.addLayout(cleanup_form)
        advanced = W.QGroupBox('更多参数'); advanced.setStyleSheet('QGroupBox {border:1px solid #303744;border-radius:6px;margin-top:8px;padding-top:12px;}')
        af = W.QFormLayout(advanced)
        af.addRow('HDF5 数据集', self.dataset)
        self.marker_z = self.integer(1,64); af.addRow('种子 Z 间隔',self.marker_z)
        self.marker_xy = self.integer(2,128); af.addRow('种子 XY 间隔',self.marker_xy)
        self.normalization = W.QComboBox(); self.normalization.addItems(['auto','1','255','4095','65535']); af.addRow('响应值范围',self.normalization)
        self.flip = W.QCheckBox('推理时翻转 Y，输出还原'); af.addRow(self.flip)
        self.input_is_crop=W.QCheckBox('输入为人工裁剪，保护裁剪边界');af.addRow(self.input_is_crop)
        self.threads = self.integer(1,32); af.addRow('CPU 线程',self.threads)
        self.preprocessing_workers=self.integer(1,32);af.addRow('前处理并行数',self.preprocessing_workers)
        self.f_neck=self.number(.1,1,.05,2);af.addRow('F 颈部阈值',self.f_neck)
        self.f_enclosure=self.number(.1,1,.05,2);af.addRow('F 射线闭合比例',self.f_enclosure)
        advanced_layout.addWidget(advanced)
        side.addStretch()
        scroll.setWidget(panel); side_outer.addWidget(scroll)
        bottom = W.QVBoxLayout(); bottom.setContentsMargins(18,8,18,16)
        self.run_button = button('开始分割', self.run, 'primary'); bottom.addWidget(self.run_button)
        self.cancel_button = button('取消运行', self.cancel); self.cancel_button.hide(); bottom.addWidget(self.cancel_button)
        side_outer.addLayout(bottom); content.addWidget(sidebar)
        workspace = W.QVBoxLayout(); workspace.setSpacing(12)
        tools = W.QHBoxLayout(); tools.addWidget(text_label('切片工作台', 'section')); tools.addStretch()
        self.mode = W.QComboBox(); self.mode.addItems(['彩色实例','边界叠加','半透明叠加']); self.mode.setMinimumWidth(130)
        tools.addWidget(self.mode)
        self.input_view=W.QComboBox();self.input_view.addItems(['原始输入','前处理结果'])
        self.input_view.hide();tools.addWidget(self.input_view)
        self.input_view.currentIndexChanged.connect(lambda *_:(self.cache.clear(),self.render()))
        tools.addWidget(button('适应窗口', self.fit_views))
        tools.addWidget(button('1 : 1', self.actual_size))
        self.export_button = button('导出 segment ↗',self.export); self.export_button.setEnabled(False); tools.addWidget(self.export_button)
        workspace.addLayout(tools)
        self.views = []
        splitter = W.QSplitter(QtCore.Qt.Horizontal)
        for i, name in enumerate(['Prediction','Segmentation']):
            card = W.QFrame(); card.setStyleSheet('QFrame {background:#13171e;border:1px solid #2a303b;border-radius:9px;}')
            cv = W.QVBoxLayout(card); cv.setContentsMargins(1,10,1,1); cv.setSpacing(10)
            label = text_label('   '+name,'section'); label.setStyleSheet('border:0; background:transparent;'); cv.addWidget(label)
            v = ImageView(); cv.addWidget(v)
            v.empty('导入 prediction 或打开示例' if i==0 else '运行后在这里查看分割结果')
            v.changed.connect(self.sync_views); v.pixel.connect(self.pixel_info)
            v.sliceRequested.connect(lambda d:self.slice_slider.setValue(self.slice_slider.value()+d))
            self.views.append(v); splitter.addWidget(card)
        splitter.setSizes([500,500]); workspace.addWidget(splitter,1)
        navigation = W.QHBoxLayout(); navigation.addWidget(text_label('Z', 'section'))
        navigation.addWidget(button('‹',lambda:self.slice_slider.setValue(self.slice_slider.value()-1)))
        self.slice_slider = W.QSlider(QtCore.Qt.Horizontal); self.slice_slider.setRange(0,0); navigation.addWidget(self.slice_slider,1)
        navigation.addWidget(button('›',lambda:self.slice_slider.setValue(self.slice_slider.value()+1)))
        self.slice_spin = self.integer(0,0); self.slice_spin.setFixedWidth(70); navigation.addWidget(self.slice_spin)
        self.slice_total = text_label('/ —', 'muted'); navigation.addWidget(self.slice_total)
        workspace.addLayout(navigation)
        self.info = text_label('','muted'); workspace.addWidget(self.info)
        self.progress = W.QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0); self.progress.setFixedHeight(5); workspace.addWidget(self.progress)
        self.stage = text_label('准备就绪','muted'); workspace.addWidget(self.stage)
        content.addLayout(workspace,1); outer.addLayout(content,1)
        for view in self.views:view.setToolTip('滚轮缩放；拖动平移；Ctrl + 滚轮切层。')
        self.slice_slider.valueChanged.connect(self.on_slice)
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)
        self.mode.currentIndexChanged.connect(self.render)
        self.fold.currentIndexChanged.connect(self.fold_changed)
        self.reset_parameters()
        self.run_button.setEnabled(False)
        self.preprocess_button.setEnabled(False)
        self.param_widgets = [self.preprocessing_enabled,self.xy,self.fold,self.threshold,self.cleanup,self.width,self.fraction,self.window,self.radius,self.marker_z,self.marker_xy,self.normalization,self.flip,self.input_is_crop,self.threads,self.reset_button,self.dataset,self.f_enabled,self.r_enabled,self.r_budget,self.stitch_enabled,self.post_enabled,self.f_neck,self.f_enclosure,self.pf_threshold]
        self.param_widgets.append(self.preprocessing_workers)
        for widget in self.param_widgets:
            if isinstance(widget,(W.QSpinBox,W.QDoubleSpinBox)): widget.valueChanged.connect(self.parameters_changed)
            elif isinstance(widget,W.QComboBox): widget.currentIndexChanged.connect(self.parameters_changed)
            elif isinstance(widget,W.QCheckBox): widget.toggled.connect(self.parameters_changed)
        W.QShortcut(QtGui.QKeySequence('Ctrl+O'), self, self.choose_folder)
        W.QShortcut(QtGui.QKeySequence('Ctrl+S'), self, self.export)
        W.QShortcut(QtGui.QKeySequence('Left'), self, lambda:self.slice_slider.setValue(self.slice_slider.value()-1))
        W.QShortcut(QtGui.QKeySequence('Right'), self, lambda:self.slice_slider.setValue(self.slice_slider.value()+1))
        self.language.currentIndexChanged.connect(self.change_language)
        self.change_language()

    def change_language(self):
        set_language(self.language.currentData())
        application=W.QApplication.instance()
        if hasattr(self,'qt_translation'):application.removeTranslator(self.qt_translation)
        self.qt_translation=QtCore.QTranslator(self)
        if self.language.currentData()=='zh':
            translations=QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.TranslationsPath)
            if not self.qt_translation.load('qtbase_zh_CN',translations) and not self.qt_translation.load('qt_zh_CN',translations):
                self.qt_translation=ChineseDialogs(self)
            application.installTranslator(self.qt_translation)
        translate_widgets(self)
        self.info.clear()
        for view in self.views:
            source=view.placeholder.data(0)
            if view.placeholder.toPlainText() and source:view.placeholder.setPlainText(tr(source))

    def separator(self, layout):
        line = W.QFrame(); line.setFrameShape(W.QFrame.HLine); line.setStyleSheet('background:#2a303b;max-height:1px;border:0;'); layout.addWidget(line)

    def integer(self, lo, hi, step=1):
        box = W.QSpinBox(); box.setRange(lo,hi); box.setSingleStep(step); box.setButtonSymbols(W.QAbstractSpinBox.PlusMinus); return box

    def number(self, lo, hi, step, decimals):
        box = W.QDoubleSpinBox(); box.setRange(lo,hi); box.setSingleStep(step); box.setDecimals(decimals); return box

    def reset_parameters(self):
        cfg = DEFAULTS
        self.preprocessing_enabled.setChecked(cfg['preprocessing_enabled'])
        self.preprocessing_workers.setValue(cfg['preprocessing_workers'])
        self.xy.setCurrentIndex(1); self.fold.setCurrentIndex(cfg['fold'])
        for k in ['threshold','width','fraction','window','radius','marker_z','marker_xy','threads','f_neck','f_enclosure','pf_threshold']: getattr(self,k).setValue(cfg[k])
        for k in ['f_enabled','r_enabled','stitch_enabled','post_enabled']:getattr(self,k).setChecked(cfg[k])
        self.r_budget.setCurrentIndex(2)
        self.cleanup.setChecked(cfg['cleanup']); self.flip.setChecked(False); self.normalization.setCurrentIndex(0)
        self.input_is_crop.setChecked(False)
        if self.path:self.apply_sample_defaults()

    def apply_sample_defaults(self):
        data=self.demo_meta
        if not data:return
        step=data.get('xy_step',1 if data.get('xy_grid_stride')==2 else 2)
        index=self.xy.findData(step)
        if index>=0:self.xy.setCurrentIndex(index)
        fold=data.get('fold',0)
        if fold in (0,1,2):
            self.fold.setCurrentIndex(fold);self.fold_changed()
        artificial=data.get('input_is_crop',bool(data.get('frame_sides')) and not any(data['frame_sides'].values()))
        self.input_is_crop.setChecked(bool(artificial))

    def fold_changed(self):
        self.threshold.setValue([.942,.972,.955][self.fold.currentIndex()])

    def parameters_changed(self):
        if self.segment is not None and not self.busy():
            self.stage.setText(tr('参数已修改 · 当前预览仍为上次结果，重新运行后生效'))

    def config(self):
        cfg = {k:getattr(self,k).value() for k in ['threshold','width','fraction','window','radius','marker_z','marker_xy','threads','f_neck','f_enclosure','pf_threshold']}
        cfg.update({k:getattr(self,k).isChecked() for k in ['f_enabled','r_enabled','stitch_enabled','post_enabled']})
        cfg['r_budget']=self.r_budget.currentData()
        cfg.update(xy_step=self.xy.currentData(), fold=self.fold.currentData(), cleanup=self.cleanup.isChecked(), flip_y=self.flip.isChecked(), normalization=self.normalization.currentText(),dataset=self.dataset.text().strip())
        cfg['input_is_crop']=self.input_is_crop.isChecked()
        cfg['preprocessing_enabled']=self.preprocessing_enabled.isChecked()
        cfg['preprocessing_workers']=self.preprocessing_workers.value()
        if cfg['preprocessing_enabled'] and self.prepared_path:
            cfg['preprocessing_input']=str(self.prepared_path)
            cfg['preprocessing_metadata']=self.prepared_metadata
        return cfg

    def select_preprocessing(self,enabled):
        if self.processed is not None:self.input_view.setCurrentIndex(1 if enabled else 0)

    def busy(self): return self.process is not None or any(j.isRunning() for j in self.jobs)

    def set_busy(self,value):
        running=self.process is not None
        cancelling=running and getattr(self,'cancelled',False)
        segmenting=running and self.process_kind=='segmentation'
        preprocessing=running and self.process_kind=='preprocessing'
        self.database_button.setEnabled(not value)
        self.folder_button.setEnabled(not value)
        self.run_button.setText(tr('正在取消…' if cancelling and segmenting else '取消分割' if segmenting else '开始分割'))
        self.preprocess_button.setText(tr('正在取消…' if cancelling and preprocessing else '取消前处理' if preprocessing else '运行前处理'))
        self.run_button.setEnabled((segmenting and not cancelling) or (not value and self.raw is not None))
        self.preprocess_button.setEnabled((preprocessing and not cancelling) or (not value and self.raw is not None))
        self.export_button.setEnabled(not value and self.segment is not None)
        for w in self.param_widgets: w.setEnabled(not value)

    def task(self, fn, done):
        job = Task(fn,self); self.jobs.append(job); self.set_busy(True)
        job.done.connect(done); job.failed.connect(self.error)
        def finish():
            self.jobs.remove(job); self.set_busy(False); job.deleteLater()
        job.finished.connect(finish); job.start()

    def choose_input(self):
        if self.busy(): return
        folder=self.path.parent if self.path else (DATABASE if DATABASE.exists() else ROOT/'resources/sample')
        path,_ = W.QFileDialog.getOpenFileName(self,tr('导入 prediction'),str(folder),tr('Prediction TIFF (*.tif *.tiff);;其他体积格式 (*.npy *.npz *.h5 *.hdf5)'),options=W.QFileDialog.DontUseNativeDialog)
        if path: self.load(Path(path))

    def choose_folder(self):
        if self.busy():return
        folder=W.QFileDialog.getExistingDirectory(self,tr('导入文件夹'),str(self.path.parent if self.path else ROOT/'resources'),options=W.QFileDialog.DontUseNativeDialog)
        if folder:
            files=self.selectable_samples(Path(folder))
            if not files:self.error('文件夹内没有可用样本。');return
            add_sample_folder(folder)
            if len(files)==1:self.load(files[0])
            else:self.choose_database(Path(folder))

    def selectable_samples(self,root=None):
        return available_samples([root] if isinstance(root,(str,Path)) else None)

    def choose_database(self,root=None):
        if self.busy(): return
        dialog=W.QDialog(self);dialog.setWindowTitle(tr('选择样本'));dialog.resize(680,560)
        layout=W.QVBoxLayout(dialog)
        tree=W.QTreeWidget();tree.setHeaderLabels(['样本 / prediction','文件大小']);tree.setColumnWidth(0,460)
        layout.addWidget(tree)
        groups={};files=self.selectable_samples(root)
        for path in files:
            try:group=path.relative_to(DATABASE).parts[0]
            except ValueError:group=path.parent.name
            volume=path.parent.name
            if path.parent.resolve()==(ROOT/'resources/sample').resolve():
                identifier=sample_metadata(path).get('volume_id','')
                if '/' in identifier:group,volume=identifier.rsplit('/',1)
            if group not in groups:groups[group]=W.QTreeWidgetItem(tree,[group])
            item=W.QTreeWidgetItem(groups[group],[f'{volume}  /  {path.name}',f'{path.stat().st_size/2**20:.1f} MB'])
            item.setData(0,QtCore.Qt.UserRole,str(path));item.setToolTip(0,str(path))
        tree.expandAll()
        layout.addWidget(text_label(tr('{count} 个样本',count=len(files)),'muted'))
        actions=W.QDialogButtonBox(W.QDialogButtonBox.Open|W.QDialogButtonBox.Cancel)
        actions.button(W.QDialogButtonBox.Open).setText(tr('打开'));actions.button(W.QDialogButtonBox.Open).setEnabled(False)
        actions.button(W.QDialogButtonBox.Cancel).setText(tr('取消'));layout.addWidget(actions)
        tree.itemSelectionChanged.connect(lambda:actions.button(W.QDialogButtonBox.Open).setEnabled(bool(tree.currentItem() and tree.currentItem().data(0,QtCore.Qt.UserRole))))
        def accept(*_):
            item=tree.currentItem()
            if item and item.data(0,QtCore.Qt.UserRole):dialog.accept()
        actions.accepted.connect(accept);actions.rejected.connect(dialog.reject);tree.itemDoubleClicked.connect(accept)
        translate_widgets(dialog)
        if dialog.exec_()==W.QDialog.Accepted:self.load(Path(tree.currentItem().data(0,QtCore.Qt.UserRole)))

    def open_sample(self):
        if self.busy(): return
        folder=ROOT/'resources/sample'
        selection=json.loads((folder/'default.json').read_text(encoding='utf-8'))
        name=selection['filename']
        if Path(name).name!=name:raise ValueError('Invalid sample filename')
        source=folder/name
        if source.exists():self.load(source)
        else:self.choose_database()

    def load(self,path):
        if self.busy():return
        if Path(path).is_dir():
            matches=available_samples([path])
            if len(matches)!=1:
                add_sample_folder(path);self.choose_database(Path(path));return
        try:path=resolve_input(path)
        except ValueError as exc:self.error(str(exc));return
        self.stage.setText(tr('正在读取文件…')); self.progress.setRange(0,0)
        key = (self.dataset.text().strip() or None) if path.suffix.lower() in ('.h5','.hdf5','.npz') else None
        def read():
            arr = read_volume(path,key)
            import historical_core as old
            scale = old.infer_prediction_scale_from_canonical(arr)
            demo=sample_metadata(path);segment=None;metadata=None
            return arr, scale, demo, segment, metadata
        def ready(data):
            previous_demo=bool(self.demo_meta)
            self.processed=None;self.prepared_path=None;self.prepared_metadata={}
            self.preprocessing_enabled.setChecked(False)
            self.input_view.setCurrentIndex(0);self.input_view.hide()
            self.preprocess_progress.setValue(0);self.preprocess_progress.hide()
            self.raw,self.display_scale,self.demo_meta,self.segment,self.meta=data; self.path=Path(path); self.cache.clear()
            self.output_path=None
            if previous_demo and not self.demo_meta:
                self.xy.setCurrentIndex(1);self.fold.setCurrentIndex(DEFAULTS['fold']);self.fold_changed()
            if not self.demo_meta:self.input_is_crop.setChecked(False)
            label=self.demo_meta.get('volume_id',path.name)
            self.file_label.setText(f'{label}\n{self.raw.shape[0]} × {self.raw.shape[1]} × {self.raw.shape[2]}  ·  {self.raw.dtype}')
            self.file_label.setToolTip(str(path)); self.slice_slider.setRange(0,len(self.raw)-1); self.slice_spin.setRange(0,len(self.raw)-1)
            self.slice_total.setText(f'/ {len(self.raw)-1}')
            self.slice_slider.setValue(int(np.clip(self.demo_meta.get('initial_z',len(self.raw)//2),0,len(self.raw)-1)))
            self.apply_sample_defaults()
            if self.segment is None:self.views[1].empty('等待分割结果')
            self.render(fit=True); self.progress.setRange(0,100); self.progress.setValue(100 if self.segment is not None else 0)
            if self.segment is not None:
                self.output_path=Path(path).parent/self.demo_meta['result_filename']
                self.stage.setText(tr('预计算示例 · {count} 个实例',count=f'{self.meta["instances"]:,}'))
                self.statusBar().clearMessage()
            else:self.stage.setText(tr('已导入'))
            self.setWindowTitle(f'FARP-Seg · {path.name}')
        self.task(read, ready)

    def on_slice(self,z):
        self.slice_spin.blockSignals(True); self.slice_spin.setValue(z); self.slice_spin.blockSignals(False); self.render()

    def gray(self,z):
        if z not in self.cache:
            processed=self.processed is not None and self.input_view.currentIndex()==1
            arr = np.nan_to_num(np.asarray((self.processed if processed else self.raw)[z],np.float32)/(1 if processed else max(self.display_scale,1e-9)))
            self.cache[z] = np.repeat(np.rint(np.clip(arr,0,1)*255).astype(np.uint8)[...,None],3,axis=-1)
            while len(self.cache)>8: self.cache.popitem(last=False)
        self.cache.move_to_end(z); return self.cache[z]

    def render(self, *_args, fit=False):
        if self.raw is None: return
        z = self.slice_slider.value(); gray=self.gray(z)
        self.views[0].set_rgb(gray,fit=fit)
        if self.segment is not None:
            labels = np.asarray(self.segment[z]); rgb=colors(labels)
            if self.mode.currentIndex()==1:
                edge=np.zeros(labels.shape,bool)
                edge[1:] |= labels[1:]!=labels[:-1]; edge[:,1:] |= labels[:,1:]!=labels[:,:-1]
                edge &= labels>0
                rgb = gray.copy(); rgb[edge]=[122,238,192]
            elif self.mode.currentIndex()==2:
                mask=labels>0; overlay=gray.copy(); overlay[mask]=(gray[mask]*.45+rgb[mask]*.55).astype(np.uint8); rgb=overlay
            self.views[1].set_rgb(rgb,fit=fit)
        self.sync_views(self.views[0])

    def sync_views(self, source):
        if self.syncing: return
        self.syncing=True
        try:
            target=self.views[1] if source is self.views[0] else self.views[0]
            target.setTransform(source.transform())
            target.centerOn(source.mapToScene(source.viewport().rect().center()))
        finally: self.syncing=False

    def fit_views(self):
        self.views[0].fit_image(); self.sync_views(self.views[0])

    def actual_size(self):
        self.views[0].resetTransform(); self.sync_views(self.views[0])

    def pixel_info(self,x,y):
        if self.raw is None: return
        z=self.slice_slider.value()
        if 0<=y<self.raw.shape[1] and 0<=x<self.raw.shape[2]:
            label=tr(' · 实例 {id}',id=int(self.segment[z,y,x])) if self.segment is not None else ''
            self.info.setText(f'Z {z}   X {x}   Y {y}   ·   {float(self.raw[z,y,x]):.4g}{label}')

    def run(self):
        if self.process is not None and self.process_kind=='segmentation':self.cancel();return
        if self.busy() or self.raw is None: return
        cfg=self.config()
        if cfg['preprocessing_enabled']:
            if not self.prepared_path or not self.prepared_path.exists():
                self.error('请先运行前处理。');return
            if (self.prepared_metadata.get('normalization')!=cfg['normalization'] or
                    self.prepared_metadata.get('dataset','')!=(cfg.get('dataset') or '')):
                self.error('输入设置已变化，请重新运行前处理。');return
        if cfg['window']%2==0: self.error('Z 局部窗口需为奇数，例如推荐的 5 层。'); return
        model=ROOT/'resources/models'/f'a14_fold{cfg["fold"]}.txt'
        if not model.exists(): self.error('缺少 A14 模型。请将模型放入 resources/models 后重试。'); return
        if any(cfg[k] for k in ('r_enabled','stitch_enabled','post_enabled')) and not (ROOT/'resources/models/full/assets.json').exists():
            self.error('缺少完整流程模型资源，请先安装 resources/models/full 下的模型与校准参数。');return
        # Input/output arrays can dominate RAM at full acquisition dimensions.
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total',ctypes.c_ulonglong),('available',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),('total_virtual',ctypes.c_ulonglong),('avail_virtual',ctypes.c_ulonglong),('avail_extended',ctypes.c_ulonglong)]
        if sys.platform=='win32':
            mem=Memory(); mem.length=ctypes.sizeof(mem); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            estimate=self.raw.size*(48/cfg['xy_step']**2+8)
            if estimate>mem.available*.8:
                self.warn_memory()
        work=ROOT/'outputs'/uuid.uuid4().hex[:12]; work.mkdir(parents=True)
        self.output_path=work/'segment.npy'
        request=work/'request.json'; request.write_text(json.dumps(dict(input=str(self.path.resolve()),output=str(self.output_path),config=cfg)),encoding='utf-8')
        self.process_kind='segmentation'
        self.start_process('engine.py',request)
        self.stage.setText(tr('启动本机推理…'))

    def run_preprocessing(self):
        if self.process is not None and self.process_kind=='preprocessing':self.cancel();return
        if self.busy() or self.raw is None:return
        work=ROOT/'outputs'/('prep_'+uuid.uuid4().hex[:12]);work.mkdir(parents=True)
        self.preprocess_output=work/'preprocessed.npy'
        request=work/'request.json'
        request.write_text(json.dumps(dict(input=str(self.path.resolve()),output=str(self.preprocess_output),
            dataset=self.dataset.text().strip(),normalization=self.normalization.currentText(),workers=self.preprocessing_workers.value())),encoding='utf-8')
        self.process_kind='preprocessing';self.preprocess_progress.setValue(0);self.preprocess_progress.show()
        self.start_process('preprocessing.py',request)
        self.stage.setText(tr('正在前处理…'))

    def start_process(self,module,request):
        self.process=QtCore.QProcess(self); self.buffer=''; self.errors=''; self.cancelled=False
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self.read_progress)
        self.process.readyReadStandardError.connect(self.read_errors)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        executable=Path(sys.executable)
        if executable.name.lower()=='pythonw.exe': executable=executable.with_name('python.exe')
        self.set_busy(True); self.progress.setRange(0,100); self.progress.setValue(1)
        self.process.start(str(executable),['-u',str(ROOT/module),str(request)])

    def read_progress(self):
        if self.process is None: return
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try: item=json.loads(line)
            except ValueError: continue
            if 'percent' in item:
                self.progress.setValue(item['percent'])
                if item.get('stage')=='R3 preprocessing':
                    done,total=map(int,item['detail'].split('/'))
                    percent=round(100*done/total);self.preprocess_progress.setValue(percent)
                    self.stage.setText(tr('正在前处理 · {done} / {total} 层 · {percent}%',done=done,total=total,percent=percent))
                else:self.stage.setText(tr('正在分割 · {percent}%',percent=item['percent']))

    def read_errors(self):
        if self.process is not None: self.errors+=bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace')

    def process_error(self,error):
        if error==QtCore.QProcess.FailedToStart:
            self.errors='推理进程无法启动。请确认 Python 和依赖已安装。'; self.finished(1,0)

    def finished(self,code,_status):
        if self.process is None: return
        self.read_progress(); self.read_errors(); proc=self.process; self.process=None; proc.deleteLater()
        self.cancel_button.hide(); self.set_busy(False)
        if self.process_kind=='preprocessing':
            if self.cancelled:
                self.preprocess_output.with_name(self.preprocess_output.name+'.partial').unlink(missing_ok=True)
                self.preprocess_progress.setValue(100 if self.processed is not None else 0)
                self.progress.setValue(0);self.stage.setText(tr('已取消 · 上次结果仍可查看'));return
            if code or not self.preprocess_output.with_suffix('.json').exists():
                self.error(self.errors[-4000:] or 'Preprocessing failed.');return
            info=json.loads(self.preprocess_output.with_suffix('.json').read_text(encoding='utf-8'))
            self.load_processed(self.preprocess_output.parent,{'preprocessing':info})
            self.preprocessing_enabled.setChecked(True)
            self.preprocess_progress.setValue(100);self.progress.setValue(100)
            self.render();self.stage.setText(tr('前处理完成'));return
        if self.cancelled:
            self.output_path.with_name(self.output_path.name+'.partial').unlink(missing_ok=True)
            (self.output_path.parent/'preprocessed.npy.partial').unlink(missing_ok=True)
            self.progress.setValue(0); self.stage.setText(tr('已取消 · 上次结果仍可查看')); return
        if code or not self.output_path.exists():
            self.error(self.errors[-4000:] or f'推理退出，退出码 {code}'); return
        self.segment=np.load(self.output_path,mmap_mode='r',allow_pickle=False)
        self.meta=json.loads(self.output_path.with_suffix('.json').read_text(encoding='utf-8'))
        if self.meta.get('preprocessing',{}).get('enabled'):
            self.load_processed(self.output_path.parent,self.meta)
        self.render(); self.progress.setValue(100); self.set_busy(False)
        self.stage.setText(tr('完成 · {count} 个实例 · {seconds} 秒',count=f'{self.meta["instances"]:,}',seconds=f'{self.meta["seconds"]:.1f}'))
        self.statusBar().showMessage(str(self.output_path))

    def load_processed(self,folder,metadata):
        info=metadata.get('preprocessing',{})
        self.processed=None
        self.prepared_path=None;self.prepared_metadata={}
        if info.get('enabled'):
            name=info['file']
            if Path(name).name!=name:raise ValueError('Invalid preprocessing asset path')
            self.processed=np.load(Path(folder)/name,mmap_mode='r',allow_pickle=False)
            if self.processed.shape!=self.raw.shape:raise ValueError('Preprocessing shape mismatch')
            self.prepared_path=(Path(folder)/name).resolve();self.prepared_metadata=info
        self.input_view.setVisible(self.processed is not None)
        self.input_view.setCurrentIndex(1 if self.processed is not None else 0)
        self.cache.clear()

    def cancel(self):
        if self.process is not None:
            self.cancelled=True;self.set_busy(True)
            self.stage.setText(tr('正在取消…'));self.process.kill()

    def warn_memory(self):
        self.memory_warning_box=W.QMessageBox(W.QMessageBox.Warning,'FARP-Seg',
            tr('可用内存可能不足，任务将继续运行。若实际内存不足，任务可能失败。'),parent=self)
        self.memory_warning_box.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        translate_widgets(self.memory_warning_box)
        self.memory_warning_box.show()

    def export(self):
        if self.busy() or self.segment is None: return
        path,selected=W.QFileDialog.getSaveFileName(self,tr('导出 segment'),str(ROOT/'outputs'/f'{self.path.stem}_segment.tif'),'TIFF (*.tif);;NumPy (*.npy);;HDF5 (*.h5)',options=W.QFileDialog.DontUseNativeDialog)
        if not path: return
        if not Path(path).suffix: path += '.npy' if 'NumPy' in selected else '.h5' if 'HDF5' in selected else '.tif'
        if Path(path).resolve()==self.path.resolve(): self.error('输出不能覆盖输入 prediction。'); return
        self.stage.setText(tr('正在导出…')); self.progress.setRange(0,0)
        def done(_):
            self.progress.setRange(0,100); self.progress.setValue(100); self.stage.setText(tr('已导出'));self.statusBar().showMessage(path)
        self.task(lambda:export_volume(self.segment,path,self.meta),done)

    def error(self,message):
        self.progress.setRange(0,100); self.stage.setText(tr('操作未完成 · 请查看提示'))
        translated=tr(message)
        box=W.QMessageBox(W.QMessageBox.Warning,'FARP-Seg',translated,parent=self)
        if self.language.currentData()=='en' and any('\u4e00'<=c<='\u9fff' for c in translated):
            box.setText('Unable to complete this operation.');box.setDetailedText(message)
        translate_widgets(box);box.exec_()

    def dragEnterEvent(self,event):
        if event.mimeData().hasUrls() and not self.busy(): event.acceptProposedAction()

    def dropEvent(self,event):
        if not self.busy():
            urls=event.mimeData().urls()
            if urls and urls[0].isLocalFile(): self.load(Path(urls[0].toLocalFile()))

    def closeEvent(self,event):
        if any(j.isRunning() for j in self.jobs):
            self.statusBar().showMessage(tr('正在读写文件，请稍候完成再关闭。')); event.ignore(); return
        if self.process is not None:
            self.process.kill(); self.process.waitForFinished(3000)
        event.accept()


def main():
    W.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    W.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app=W.QApplication(sys.argv); app.setStyle('Fusion'); app.setStyleSheet(STYLE)
    app.setApplicationName('FARP-Seg')
    window=MainWindow(); window.show()
    if '--database' in sys.argv: QtCore.QTimer.singleShot(200,window.choose_database)
    elif '--sample' in sys.argv: QtCore.QTimer.singleShot(200,window.open_sample)
    if '--run' in sys.argv:
        startup=QtCore.QTimer(window)
        def start_when_ready():
            if window.raw is not None and not window.busy():
                startup.stop(); window.run()
        startup.timeout.connect(start_when_ready); startup.start(300)
    sys.exit(app.exec_())


if __name__=='__main__': main()
