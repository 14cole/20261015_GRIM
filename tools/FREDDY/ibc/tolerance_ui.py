"""Dedicated sensitivity/yield workspace using the shared FREDDY job contract."""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QLineEdit, QComboBox, QPushButton, QTableWidget,
                              QTableWidgetItem, QHeaderView, QDoubleSpinBox, QScrollArea,
                              QAbstractItemView)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from .tolerance_config import (SETUP_DEFAULTS, DEFAULT_SPEC, PARAMETERS, DISTRIBUTIONS,
                               MODES, ENVELOPE_MODE, layer_parameters, validate_setup)
from .tolerance_analysis import (parameters_from_layers, study_workload, run_tolerance_study,
                                 export_tolerance_report, StopToleranceAnalysis)
from .compute import layer_material_label
from .thickness_envelopes import envelope_workload, run_thickness_envelopes, export_envelope_csv, thickness_axes
from .io import layer_config_to_dict
from .plot import show_grid_value_on_hover, style_axis, style_colorbar
from .ui_controls import messagebox, filedialog

VIEWS = ('Sensitivity ranking', 'Parameter tolerance sweep', 'Margin distribution',
         'Failure map', 'Yield convergence')

SETTING_HELP = {
    'mode': 'Sensitivity only varies one parameter at a time. Statistical trials also vary all active parameters together to estimate yield. Thickness envelopes plot minimum/maximum reflection versus frequency for each layer and the complete selected thickness grid.',
    'f_start': 'First frequency in the sweep, in GHz; must be greater than zero.',
    'f_stop': 'Upper frequency limit in GHz. The sweep includes this value only when it falls on a step from the start frequency.',
    'f_step': 'Frequency spacing in GHz; must be positive. Smaller steps help resolve narrow resonances but increase calculation time.',
    'a_start': 'First incidence angle in degrees, measured from the surface normal: 0° is normal incidence. Use angles from 0° to less than 90°.',
    'a_stop': 'Upper incidence-angle limit in degrees from the surface normal, less than 90°. Included only when it falls on a step from the start angle.',
    'a_step': 'Incidence-angle spacing in degrees; must be positive. Smaller steps sample angular performance more closely.',
    'polarization': 'TE has electric field perpendicular to the plane of incidence; TM has magnetic field perpendicular to it. Both evaluates both polarizations. Sensitivity/yield requires all selected points and polarizations to pass; thickness plots let you view TE and TM separately.',
    'target': 'Maximum allowed PEC-backed reflection in dB. More negative means less reflection. Sensitivity/yield passes when every sampled frequency, angle and polarization meets this limit; thickness envelopes display it as a reference line.',
    'points': 'Odd number of values (5–101) across each parameter’s ± tolerance, including nominal and both endpoints. Each sweep varies one parameter while holding all others nominal. More points can resolve nonmonotonic responses.',
    'samples': 'Number of joint manufacturing trials in each independent repeat. Each trial evaluates the entire selected frequency/angle/polarization grid. Use a power of two; total trials = samples per repeat × repeats, limited to 65,536.',
    'seed': 'Integer from 0 to 4,294,967,295 that controls statistical sampling. Identical inputs and seed reproduce a study; independent repeats use successive seeds.',
    'repeats': 'Number of statistical studies (1–16) with different seeds. Reports the spread in modeled yield and, with multiple repeats, sampling standard error. This measures sampling variability, not material-model accuracy.',
}

PARAMETER_HELP = (
    'Layer and parameter being varied. Layer numbers follow the current stack order.',
    'Symmetric ± tolerance around the nominal value. Zero holds this parameter fixed. Thickness and resistance must remain positive throughout their bounds.',
    'Percent uses the magnitude of the nominal value at each frequency. Absolute uses inches for thickness, Ω/sq for sheet resistance, or relative units for ε/μ components.',
    'Statistical trials only: Uniform samples within the entered bounds. Truncated normal uses bound/3 as the underlying standard deviation and truncates at the bounds.',
    'Statistical trials only: leave blank for independence. Matching group names share a manufacturing factor; the loading controls its strength and direction.',
    'Statistical trials only: value from −1 to +1. Zero removes the shared factor. +1/+1 move together; +1/−1 move oppositely. Latent correlation equals the product of the two loadings; transformed-variable correlation can differ.',
    'Physical unit used when Units is Absolute: inches, Ω/sq, or dimensionless relative ε/μ.',
)


class ToleranceWidget(QWidget):
    retain_layer_metrics = False
    study_name = 'Sensitivity & Yield'
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.result = None
        self.colors = None
        self.active = False
        self._stop = threading.Event()
        self._progress = None
        self._refreshing = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setup_page = QWidget()
        scroll.setWidget(self.setup_page)
        self.tabs.addTab(scroll, 'Setup')
        layout = QVBoxLayout(self.setup_page)
        top = QHBoxLayout()
        intro = QLabel('Analyze the current PEC-backed stack. Apply an inverse candidate first to study that design.')
        intro.setWordWrap(True)
        top.addWidget(intro, 1)
        self.edit_stack = QPushButton('Edit stack…')
        self.edit_stack.clicked.connect(lambda: host._open_guide_workflow('Impedance'))
        top.addWidget(self.edit_stack)
        help_button = QPushButton('Workflow help')
        help_button.clicked.connect(lambda: host._show_guide('tolerance'))
        top.addWidget(help_button)
        layout.addLayout(top)
        self.stack_summary = QLabel()
        self.stack_summary.setWordWrap(True)
        layout.addWidget(self.stack_summary)
        self.fields = {}
        self.field_labels = {}
        self.setup_grid = grid = QGridLayout()
        self._visible_fields = None
        self.field_rows = field_rows = [
            [('mode', 'Analysis')],
            [('f_start', 'Frequency start (GHz)'), ('f_stop', 'Stop (GHz)'), ('f_step', 'Step (GHz)')],
            [('a_start', 'Angle start (°)'), ('a_stop', 'Stop (°)'), ('a_step', 'Step (°)')],
            [('polarization', 'Polarization'), ('target', 'Reflection Target (dB)'), ('points', 'Points per parameter')],
            [('samples', 'Samples per repeat'), ('seed', 'Random seed'), ('repeats', 'Independent repeats')],
        ]
        options = {'polarization': ['TE', 'TM', 'Both'], 'mode': MODES,
                   'samples': [str(2**i) for i in range(4, 17)], 'repeats': [str(i) for i in range(1, 17)]}
        for row, entries in enumerate(field_rows):
            for col, (key, label) in enumerate(entries):
                caption = QLabel(label)
                caption.setToolTip(SETTING_HELP[key])
                self.field_labels[key] = caption
                grid.addWidget(caption, row, 2 * col)
                if key in options:
                    edit = QComboBox()
                    edit.addItems(options[key])
                    edit.setCurrentText(SETUP_DEFAULTS[key])
                    edit.currentTextChanged.connect(self.refresh_workload)
                else:
                    edit = QLineEdit(SETUP_DEFAULTS[key])
                    edit.textChanged.connect(self.refresh_workload)
                edit.setToolTip(SETTING_HELP[key])
                edit.setMinimumWidth(85)
                self.fields[key] = edit
                grid.addWidget(edit, row, 2 * col + 1)
        layout.addLayout(grid)
        self.copy_inverse = QPushButton('Copy inverse frequencies, angles and requirement')
        self.copy_inverse.setToolTip('Copy the inverse band sweep, incidence-angle sweep, polarization and reflection target. For a discrete-frequency inverse study, enter the desired sweep here. Does not apply an inverse candidate or reuse calculated results.')
        self.copy_inverse.clicked.connect(self.use_inverse_setup)
        layout.addWidget(self.copy_inverse)
        note = QLabel('Set ± bounds below; zero disables a parameter. Absolute units: thickness in inches, resistance in Ω/sq, ε/μ relative units. '
                      'Percent bounds use the magnitude of the nominal component at each frequency. The same trial deviation spans the whole curve.')
        note.setWordWrap(True)
        self.input_note = note
        layout.addWidget(note)
        self.parameters = QTableWidget(0, 7)
        self.parameters.setHorizontalHeaderLabels(['Layer / parameter', '± bound', 'Units', 'Distribution', 'Shared group', 'Loading', 'Absolute unit'])
        for col, help_text in enumerate(PARAMETER_HELP):
            self.parameters.horizontalHeaderItem(col).setToolTip(help_text)
        self.parameters.setMinimumHeight(245)
        self.parameters.setAlternatingRowColors(True)
        self.parameters.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col, width in ((1, 120), (2, 95), (3, 190), (4, 115), (5, 95), (6, 110)):
            self.parameters.setColumnWidth(col, width)
        layout.addWidget(self.parameters, 1)
        self.thickness_ranges = QTableWidget(0, 6)
        self.thickness_ranges.setHorizontalHeaderLabels(['Layer', 'Nominal (in)', 'Minimum (in)', 'Maximum (in)', 'Step (in)', 'Defined values'])
        self.thickness_ranges.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.thickness_ranges.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thickness_ranges.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.thickness_ranges.setSelectionMode(QAbstractItemView.SingleSelection)
        self.thickness_ranges.verticalHeader().hide()
        self.thickness_ranges.setMinimumHeight(245)
        self.thickness_ranges.setToolTip('Uses the same layer minimum, maximum and step as Inverse Design. Values are minimum + k × step, never exceeding maximum. A layer without limits stays at its nominal thickness. Nominal is plotted separately and may lie outside the bounds.')
        layout.addWidget(self.thickness_ranges, 1)
        self.edit_range = QPushButton('Edit selected layer range…')
        self.edit_range.setToolTip('Open the existing layer editor. Changes to minimum, maximum and step apply to both Inverse Design and thickness envelopes.')
        self.edit_range.clicked.connect(self.edit_selected_range)
        layout.addWidget(self.edit_range)
        group_note = QLabel('Statistical groups: blank = independent. Matching names share a Gaussian manufacturing factor. '
                            'Latent correlation = loading₁ × loading₂; +1 moves together, opposite signs move oppositely. '
                            'Uniform or normal-distribution Pearson correlations can differ after the bounded transform. '
                            'Truncated normal uses nominal ±3σ as the entered hard bounds.')
        group_note.setWordWrap(True)
        self.group_note = group_note
        layout.addWidget(group_note)
        self.workload = QLabel()
        self.workload.setWordWrap(True)
        layout.addWidget(self.workload)
        buttons = QHBoxLayout()
        self.run_button = QPushButton('Run sensitivity study')
        self.run_button.clicked.connect(self.run)
        buttons.addWidget(self.run_button)
        self.stop_button = QPushButton('Stop study')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop.set)
        buttons.addWidget(self.stop_button)
        self.progress_label = QLabel('Ready')
        buttons.addWidget(self.progress_label, 1)
        layout.addLayout(buttons)

        results_page = QWidget()
        results = QVBoxLayout(results_page)
        self.tabs.addTab(results_page, 'Results')
        self.result_status = QLabel('Run a study from Setup to inspect sensitivity and modeled yield.')
        self.result_status.setWordWrap(True)
        results.addWidget(self.result_status)
        bar = QHBoxLayout()
        self.view = QComboBox()
        self.view.setToolTip('Choose how to display the captured results. Changing a view does not rerun the analysis.')
        self.view.addItems(VIEWS)
        self.view.currentTextChanged.connect(self.draw)
        bar.addWidget(self.view)
        self.parameter_choice = QComboBox()
        self.parameter_choice.setToolTip('Choose a captured parameter sweep or thickness-envelope panel. All plots shows every individual layer and the combined envelope.')
        self.parameter_choice.currentIndexChanged.connect(self.draw)
        bar.addWidget(self.parameter_choice, 1)
        self.pol_choice = QComboBox()
        self.pol_choice.setToolTip('Choose the polarization to display from the captured study. Does not change which polarizations were evaluated.')
        self.pol_choice.currentIndexChanged.connect(self.draw)
        bar.addWidget(self.pol_choice)
        self.angle_choice = QComboBox()
        self.angle_choice.setToolTip('Select one incidence angle to isolate thickness effects. All selected angles pools their minimum/maximum bounds and shows the nominal angular range as well.')
        self.angle_choice.currentIndexChanged.connect(self.draw)
        bar.addWidget(self.angle_choice)
        self.csv_button = QPushButton('Export envelope CSV…')
        self.csv_button.clicked.connect(self.export_csv)
        self.csv_button.setVisible(False)
        bar.addWidget(self.csv_button)
        self.export_button = QPushButton('Export study JSON…')
        self.export_button.clicked.connect(self.export)
        self.export_button.setEnabled(False)
        bar.addWidget(self.export_button)
        results.addLayout(bar)
        self.figure = Figure(figsize=(9, 4), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        results.addWidget(NavigationToolbar2QT(self.canvas, self))
        results.addWidget(self.canvas, 1)
        self.summary = QTableWidget(0, 5)
        self.summary.setHorizontalHeaderLabels(['Parameter', 'Margin loss (dB)', 'Worst margin (dB)', 'First negative miss', 'First positive miss'])
        self.summary.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary.setMaximumHeight(185)
        self.summary.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary.itemSelectionChanged.connect(self.select_summary)
        results.addWidget(self.summary)
        caveat = QLabel('Positive margin passes every sampled frequency/angle/polarization. Individual sweeps hold other inputs fixed; '
                        'first-miss brackets are sampled bounds, not guaranteed continuous limits. Statistical pass fraction depends on '
                        'the captured distributions and correlations. Refine the operating grid and repeat seeds to assess convergence.')
        caveat.setWordWrap(True)
        self.caveat = caveat
        self.standard_caveat = caveat.text()
        results.addWidget(caveat)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.show_progress)
        self.refresh_layers()

    def capture_setup(self):
        return {key: edit.currentText() if isinstance(edit, QComboBox) else edit.text() for key, edit in self.fields.items()}

    def restore_setup(self, setup):
        self._refreshing = True
        try:
            for key, value in setup.items():
                if key not in self.fields: continue  # Ignore retired controls in older setups.
                edit = self.fields[key]
                edit.setCurrentText(value) if isinstance(edit, QComboBox) else edit.setText(value)
        finally:
            self._refreshing = False
        self.clear_result()
        self.refresh_workload()

    def use_inverse_setup(self):
        discrete = self.host.inv_freq_mode_var.get() != 'Band sweep'
        if discrete:
            messagebox.showwarning(self.study_name + ' setup', 'This workspace uses a frequency band. Enter the desired band here for a discrete-frequency inverse study.')
            return
        mapping = {'f_start': 'inv_target_start', 'f_stop': 'inv_target_stop', 'f_step': 'inv_target_step',
                   'a_start': 'inv_angle_start', 'a_stop': 'inv_angle_stop', 'a_step': 'inv_angle_step',
                   'target': 'inv_requirement_db', 'polarization': 'inv_wave_pol'}
        for key, source in mapping.items():
            edit = self.fields[key]
            value = getattr(self.host, source + '_var').get()
            edit.setCurrentText('Both' if value.lower()=='both' else value.upper()) if isinstance(edit, QComboBox) else edit.setText(value)

    def refresh_layers(self):
        self._refreshing = True
        self.parameters.setRowCount(0)
        descriptions = []
        for index, layer in enumerate(self.host.layers):
            descriptions.append(f'{index+1}. ' + (f'Sheet {layer.sheet_resistance:g} Ω/sq' if layer.is_sheet
                                else f'{layer_material_label(layer)} · {layer.thickness_in:g} in'))
            for key in layer_parameters(layer.is_sheet):
                spec = {**DEFAULT_SPEC, **layer.tolerances.get(key, {})}
                row = self.parameters.rowCount()
                self.parameters.insertRow(row)
                item = QTableWidgetItem(f'{index+1}. {PARAMETERS[key]}')
                item.setData(Qt.UserRole, key)
                item.setToolTip(PARAMETER_HELP[0])
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.parameters.setItem(row, 0, item)
                bound = QDoubleSpinBox()
                bound.setDecimals(6)
                bound.setRange(0, 1e9)
                bound.setValue(spec['bound'])
                units = QComboBox()
                units.addItems(['%', 'Absolute'])
                units.setCurrentText(spec['units'])
                distribution = QComboBox()
                distribution.addItems(DISTRIBUTIONS)
                distribution.setCurrentText(spec['distribution'])
                group = QLineEdit(spec['group'])
                group.setMaxLength(80)
                loading = QDoubleSpinBox()
                loading.setRange(-1, 1)
                loading.setSingleStep(.1)
                loading.setDecimals(3)
                loading.setValue(spec['loading'])
                for col, widget in enumerate((bound, units, distribution, group, loading), 1):
                    widget.setToolTip(PARAMETER_HELP[col])
                    self.parameters.setCellWidget(row, col, widget)
                unit = 'in' if key == 'thickness' else ('Ω/sq' if layer.is_sheet else 'relative')
                item = QTableWidgetItem(unit)
                item.setToolTip(PARAMETER_HELP[6])
                item.setData(Qt.UserRole, key)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.parameters.setItem(row, 6, item)
                def changed(*_, layer=layer, key=key, widgets=(bound, units, distribution, group, loading)):
                    if self._refreshing: return
                    b, u, d, g, l = widgets
                    value = dict(bound=b.value(), units=u.currentText(), distribution=d.currentText(), group=g.text().strip(), loading=l.value())
                    if value == DEFAULT_SPEC: layer.tolerances.pop(key, None)
                    else: layer.tolerances[key] = value
                    self.refresh_workload()
                    self.host._schedule_inverse_work_count()
                bound.valueChanged.connect(changed)
                units.currentTextChanged.connect(changed)
                distribution.currentTextChanged.connect(changed)
                group.textChanged.connect(changed)
                loading.valueChanged.connect(changed)
        self.stack_summary.setText(' | '.join(descriptions) if descriptions else 'No layers. Use Edit stack to add materials or sheets.')
        self.refresh_ranges()
        self._refreshing = False
        self.refresh_workload()

    def refresh_ranges(self):
        self.thickness_ranges.setRowCount(0)
        for index, layer in enumerate(self.host.layers):
            if layer.is_sheet: continue
            row = self.thickness_ranges.rowCount()
            self.thickness_ranges.insertRow(row)
            low, high = layer.inv_t_min_in, layer.inv_t_max_in
            if low is None and high is None: low = high = layer.thickness_in
            try:
                _, axes = thickness_axes([layer])
                axis = axes[0]
                count = f'{axis.count:,}'
                detail = f'{axis.count:,} values; first {axis.value(0):.12g} in, last {axis.value(axis.count-1):.12g} in. Maximum is included only if reached by the step.'
            except ValueError as exc:
                count, detail = 'Invalid range', str(exc)
            values = [f'{index+1}. {layer_material_label(layer)}', layer.thickness_in,
                      low, high, layer.inv_t_accuracy_in, count]
            for col, value in enumerate(values):
                text = '—' if value is None else f'{value:.12g}' if isinstance(value, (float, int)) else value
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, index)
                item.setToolTip(detail)
                self.thickness_ranges.setItem(row, col, item)
        if self.thickness_ranges.rowCount(): self.thickness_ranges.selectRow(0)

    def edit_selected_range(self):
        if self.host.job_is_running(): return
        item = self.thickness_ranges.item(self.thickness_ranges.currentRow(), 0)
        if item is None: return
        self.host.layer_list.setCurrentRow(item.data(Qt.UserRole))
        self.host._edit_layer()

    def _refresh_setup_visibility(self, envelopes, statistical):
        hidden = set()
        if envelopes:
            hidden.add('points')
        if not statistical:
            hidden.update(('samples', 'seed', 'repeats'))
        visible = tuple(key for key in self.fields if key not in hidden)
        if visible == self._visible_fields:
            return
        self._visible_fields = visible
        # Repack the visible label/control pairs so hidden modes leave no gaps.
        for key, edit in self.fields.items():
            caption = self.field_labels[key]
            self.setup_grid.removeWidget(caption)
            self.setup_grid.removeWidget(edit)
            caption.setVisible(key not in hidden)
            edit.setVisible(key not in hidden)
        row = 0
        for entries in self.field_rows:
            keys = [key for key, _ in entries if key not in hidden]
            if not keys:
                continue
            for col, key in enumerate(keys):
                self.setup_grid.addWidget(self.field_labels[key], row, 2 * col)
                self.setup_grid.addWidget(self.fields[key], row, 2 * col + 1)
            row += 1

    def refresh_workload(self, *_):
        if self._refreshing or len(self.fields) != len(SETUP_DEFAULTS): return
        envelopes = self.fields['mode'].currentText() == ENVELOPE_MODE
        statistical = self.fields['mode'].currentText() == MODES[1]
        self._refresh_setup_visibility(envelopes, statistical)
        self.parameters.setVisible(not envelopes)
        self.thickness_ranges.setVisible(envelopes)
        self.edit_range.setVisible(envelopes)
        self.group_note.setVisible(statistical)
        self.parameters.verticalHeader().setVisible(not envelopes)
        for col in (3,4,5): self.parameters.setColumnHidden(col,not statistical)
        self.fields['points'].setEnabled(not envelopes)
        for row in range(self.parameters.rowCount()):
            self.parameters.setRowHidden(row,envelopes and self.parameters.item(row,0).data(Qt.UserRole)!='thickness')
            for col in (3,4,5): self.parameters.cellWidget(row,col).setEnabled(statistical)
        self.input_note.setText('Uses each layer’s existing minimum, maximum and step in inches. Every defined thickness is evaluated individually, and every combination is evaluated together. Layers without limits stay nominal. Nominal is a separate reference; material properties and sheets stay nominal. Statistical ± bounds do not apply.' if envelopes else 'Set ± bounds below; zero disables a parameter. Absolute units: thickness in inches, resistance in Ω/sq, ε/μ relative units. Percent bounds use nominal magnitude.')
        self.fields['samples'].setEnabled(statistical)
        self.fields['seed'].setEnabled(statistical)
        self.fields['repeats'].setEnabled(statistical)
        self.run_button.setText('Compute thickness envelopes' if envelopes else 'Run sensitivity + statistical study' if statistical else 'Run sensitivity study')
        try:
            if envelopes:
                setup = validate_setup(self.capture_setup())
                w = envelope_workload(self.host.layers,setup,retain_layer_metrics=self.retain_layer_metrics)
                params = range(w['active_thicknesses'])
            else:
                params = parameters_from_layers(self.host.layers)
                w = study_workload(self.capture_setup(), len(params), len(self.host.layers))
            description = (f'{w["joint_combinations"]:,} defined joint combinations · {w["evaluations"]:,} total evaluations including individual sweeps and nominal reference · ' if envelopes else
                           f'{len(params)} active parameter(s) · {w["evaluations"]:,} stack evaluations · ')
            self.workload.setText(description +
                                  f'{w["response_points"]:,} response points · about {w["array_mib"]:.2f} MiB of analysis arrays '
                                  '(excludes loaded files, GUI and Python overhead).')
        except ValueError as exc:
            self.workload.setText(str(exc))
        if self.result is not None:
            self.result_status.setText(self.result_summary() + ' Results retain the captured setup; rerun to evaluate edits.')

    def set_busy(self, busy):
        for widget in (self.parameters, self.thickness_ranges, self.edit_range, self.copy_inverse, self.edit_stack, *self.fields.values(), self.run_button):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy and self.active)
        self.export_button.setEnabled(not busy and self.result is not None)
        self.csv_button.setEnabled(not busy and self.result is not None)
        if not busy:
            self.timer.stop()
            self.active = False
            self.refresh_workload()

    def show_progress(self):
        if self._progress:
            done, total, phase = self._progress
            text = f'{done:,} / {total:,} · {phase}'
            self.progress_label.setText(text)
            self.host.status_var.set(self.study_name + ': ' + text)
            self.host.status_progress.setRange(0, 1000)
            self.host.status_progress.setValue(int(1000 * done / max(total, 1)))

    def run(self):
        if self.host.job_is_running(): return
        try:
            setup = validate_setup(self.capture_setup())
            snapshot = self.host._snapshot_layers()
            if setup['mode']==ENVELOPE_MODE: envelope_workload(snapshot,setup,retain_layer_metrics=self.retain_layer_metrics)
            else: parameters_from_layers(snapshot)
            captured_layers = [layer_config_to_dict(layer) for layer in snapshot]
        except ValueError as exc:
            messagebox.showerror(self.study_name + ' setup', str(exc))
            return
        self._stop.clear()
        self._progress = None
        self.active = True
        self.timer.start()
        def progress(done, total, phase):
            self._progress = done, total, phase
        def worker():
            try:
                loaded = self.host._load_layers(snapshot)
                analysis = run_thickness_envelopes if setup['mode']==ENVELOPE_MODE else run_tolerance_study
                options = dict(retain_layer_metrics=self.retain_layer_metrics) if setup['mode']==ENVELOPE_MODE else {}
                result = analysis(loaded, snapshot, setup, stop_requested=self._stop.is_set, progress=progress, **options)
                result['layers'] = captured_layers
                return result
            except StopToleranceAnalysis:
                raise StopToleranceAnalysis(self.study_name + ' study stopped; previous results retained.') from None
        def success(result):
            self.result = result
            self.publish()
            self.progress_label.setText(f'Completed {result["evaluated"]:,} evaluations')
            self.tabs.setCurrentIndex(1)
        self.host._run_background_task(self.study_name, worker, success, self.study_name + ' error')

    def clear_result(self):
        self.result = None
        self.summary.setRowCount(0)
        self.parameter_choice.clear()
        self.pol_choice.clear()
        self.export_button.setEnabled(False)
        self.result_status.setText('Run a study from Setup to inspect sensitivity and modeled yield.')
        self.progress_label.setText('Ready')
        self.tabs.setCurrentIndex(0)
        self.draw()

    def result_summary(self):
        r = self.result
        if r.get('kind')=='thickness_envelopes':
            return (f'Thickness-only envelopes · {len(r["panels"])-1} layer plots + all layers · '
                    f'{r["workload"]["joint_combinations"]:,} defined combinations fully evaluated · '
                    f'{len(r["frequencies_ghz"])} frequencies × {len(r["angles_deg"])} incidence angles · '
                    + '+'.join(p.upper() for p in r['polarizations']))
        text = f'Captured PEC limit {r["target_db"]:g} dB · nominal margin {r["nominal_margin_db"]:+.3f} dB'
        if r['trials']:
            text += f' · modeled pass fraction {r["pass_fraction_pct"]:.2f}% ({r["passing"]:,}/{r["trials"]:,} trials)'
            rep = r.get('repeatability')
            if rep and len(rep['repeat_pass_pct'])>1:
                text += f' · repeat range {rep["min_pass_pct"]:.2f}–{rep["max_pass_pct"]:.2f}% · sampling SE {rep["standard_error_percentage_points"]:.3f} percentage points (not a guaranteed bound)'
        else:
            text += ' · individual parameter sweeps only'
        text += (f'\n{r["frequencies_ghz"][0]:g}–{r["frequencies_ghz"][-1]:g} GHz '
                 f'({len(r["frequencies_ghz"])} points) · {r["angles_deg"][0]:g}–{r["angles_deg"][-1]:g}° '
                 f'({len(r["angles_deg"])} angles) · ' + ' + '.join(p.upper() for p in r['polarizations']))
        return text

    def publish(self):
        envelope = self.result.get('kind')=='thickness_envelopes'
        self.view.blockSignals(True)
        self.view.clear()
        self.view.addItems(['Thickness envelopes'] if envelope else VIEWS)
        self.view.blockSignals(False)
        if envelope:
            self.parameter_choice.blockSignals(True)
            self.parameter_choice.clear()
            self.parameter_choice.addItems(['All plots']+[p['label'] for p in self.result['panels']])
            if len(self.result['panels'])>6: self.parameter_choice.setCurrentIndex(1)
            self.parameter_choice.blockSignals(False)
            self.pol_choice.blockSignals(True)
            self.pol_choice.clear(); self.pol_choice.addItems([p.upper() for p in self.result['polarizations']])
            self.pol_choice.blockSignals(False)
            self.angle_choice.blockSignals(True)
            self.angle_choice.clear()
            self.angle_choice.addItems(['All selected angles']+[f'{a:g}° incidence' for a in self.result['angles_deg']])
            self.angle_choice.setCurrentIndex(1)
            self.angle_choice.blockSignals(False)
            self.result_status.setText(self.result_summary())
            self.summary.setRowCount(0)
            self.export_button.setEnabled(not self.host.job_is_running())
            self.csv_button.setEnabled(not self.host.job_is_running())
            self.draw()
            return
        self.parameter_choice.blockSignals(True)
        self.parameter_choice.clear()
        self.parameter_choice.addItems([s['label'] for s in self.result['sensitivities']])
        self.parameter_choice.blockSignals(False)
        self.pol_choice.blockSignals(True)
        self.pol_choice.clear()
        self.pol_choice.addItems([p.upper() for p in self.result['polarizations']])
        self.pol_choice.blockSignals(False)
        self.result_status.setText(self.result_summary())
        self.summary.setSortingEnabled(False)
        self.summary.setRowCount(len(self.result['sensitivities']))
        for row, s in enumerate(self.result['sensitivities']):
            def bracket(direction):
                if not s['crossings']['nominal_pass']: return 'Nominal misses'
                pair = s['crossings'][direction]
                if pair is None: return 'No sampled miss'
                return f'{100*pair[0]:g}% → {100*pair[1]:g}% of bound'
            values = [s['label'], s['margin_loss_db'], min(s['margins_db']), bracket('negative'), bracket('positive')]
            for col, value in enumerate(values):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, round(value, 5) if isinstance(value, float) else value)
                item.setData(Qt.UserRole, row)
                self.summary.setItem(row, col, item)
        self.summary.setSortingEnabled(True)
        self.summary.sortItems(1, Qt.DescendingOrder)
        self.export_button.setEnabled(not self.host.job_is_running())
        self.draw()

    def select_summary(self):
        row = self.summary.currentRow()
        item = self.summary.item(row, 0)
        if item is not None:
            self.parameter_choice.setCurrentIndex(item.data(Qt.UserRole))

    def apply_theme(self, colors):
        self.colors = colors
        self.draw()

    def draw(self, *_):
        if not hasattr(self, 'figure'): return
        envelope = self.result is not None and self.result.get('kind')=='thickness_envelopes'
        self.angle_choice.setVisible(envelope)
        self.csv_button.setVisible(envelope)
        self.summary.setVisible(not envelope)
        self.caveat.setText('Bounds cover every defined min/max/step combination, not the continuous intervals between values. Nominal is a separate reference and can lie outside the bounds. All selected angles also pools angular spread; select one incidence angle to isolate thickness effects.' if envelope else self.standard_caveat)
        if envelope:
            from .thickness_envelope_plot import draw_envelopes
            self.parameter_choice.setVisible(True);self.pol_choice.setVisible(True)
            draw_envelopes(self.figure,self.result,max(0,self.parameter_choice.currentIndex()),
                           self.pol_choice.currentIndex(),self.angle_choice.currentIndex(),self.colors)
            self.canvas.draw_idle()
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        r = self.result
        view = self.view.currentText()
        if r is not None:
            self.figure.suptitle(f'PEC limit {r["target_db"]:g} dB | '
                                f'{r["frequencies_ghz"][0]:g}–{r["frequencies_ghz"][-1]:g} GHz | '
                                f'{r["angles_deg"][0]:g}–{r["angles_deg"][-1]:g}° | '
                                + '+'.join(p.upper() for p in r['polarizations']),
                                fontsize=9, color=self.colors['plot_text'] if self.colors else 'black')
        self.parameter_choice.setVisible(view == 'Parameter tolerance sweep')
        self.pol_choice.setVisible(view == 'Failure map')
        if r is None:
            ax.text(.5, .5, 'Run a study to view captured results.', ha='center', va='center', transform=ax.transAxes)
        elif view == 'Sensitivity ranking':
            records = sorted(r['sensitivities'], key=lambda s: s['margin_loss_db'], reverse=True)[:20][::-1]
            ax.barh([s['label'] for s in records], [s['margin_loss_db'] for s in records], color='#529ce8')
            ax.set_xlabel('Largest margin loss within each individual sampled tolerance (dB)')
            ax.set_title('Most influential parameters' + (' · top 20' if len(r['sensitivities']) > 20 else ''))
        elif view == 'Parameter tolerance sweep':
            s = r['sensitivities'][max(0, self.parameter_choice.currentIndex())]
            p = s['parameter']
            ax.plot(np.asarray(r['offsets']) * p['bound'], s['margins_db'], '.-', color='#529ce8')
            ax.axhline(0, color='#c08060', linestyle='--')
            unit = '%' if p['units'] == '%' else ('in' if p['key'] == 'thickness' else ('Ω/sq' if p['key'] == 'sheet_resistance' else 'relative units'))
            ax.set_xlabel(f'Signed deviation ({unit}) · other parameters fixed')
            ax.set_ylabel('Whole-region requirement margin (dB)')
            ax.set_title(s['label'])
        elif not r['trials']:
            ax.text(.5, .5, 'Enable statistical trials in Setup and rerun for this view.', ha='center', va='center', transform=ax.transAxes)
        elif view == 'Margin distribution':
            ax.hist(r['trial_margins_db'], bins=min(40, max(5, int(np.sqrt(r['trials'])))), color='#529ce8')
            ax.axvline(0, color='#c08060', linestyle='--')
            ax.set_xlabel('Whole-region margin (dB) · ≥ 0 passes')
            ax.set_ylabel('Simulated stacks')
            ax.set_title(f'Modeled pass fraction {r["pass_fraction_pct"]:.2f}%')
        elif view == 'Failure map':
            pi = max(0, self.pol_choice.currentIndex())
            def edges(v, fallback):
                a = np.asarray(v)
                if len(a) == 1: return [a[0]-fallback, a[0]+fallback]
                mid = (a[:-1]+a[1:])/2
                return np.r_[a[0]-(mid[0]-a[0]), mid, a[-1]+(a[-1]-mid[-1])]
            cells = edges(r['frequencies_ghz'], .025), edges(r['angles_deg'], .5), 100 * r['failure_counts'][pi] / r['trials']
            mesh = ax.pcolormesh(*cells, vmin=0, vmax=100, cmap='magma')
            show_grid_value_on_hover(ax, *cells, edges=True)
            colorbar = self.figure.colorbar(mesh, ax=ax, label='Trials missing the point requirement (%)')
            if self.colors: style_colorbar(colorbar, self.colors)
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Incidence angle (°)')
            ax.set_title(r['polarizations'][pi].upper() + ' · pointwise failures')
            if len(r['frequencies_ghz']) > 1:
                ax.set_xlim(r['frequencies_ghz'][0], r['frequencies_ghz'][-1])
            if len(r['angles_deg']) > 1:
                ax.set_ylim(r['angles_deg'][0], r['angles_deg'][-1])
        else:
            samples, fractions = np.asarray(r['convergence']).T
            ax.semilogx(samples, fractions, '.-', base=2, color='#529ce8')
            ax.set_ylim(0, 100)
            ax.set_xlabel('Completed statistical trials')
            ax.set_ylabel('Modeled pass fraction (%)')
            ax.set_title('Sampling convergence · repeat with other seeds to check stability')
        if self.colors:
            self.figure.patch.set_facecolor(self.colors['plot_bg'])
            style_axis(ax, self.colors)
            for text in ax.texts: text.set_color(self.colors['plot_text'])
        self.canvas.draw_idle()

    def export(self):
        if self.result is None or self.host.job_is_running(): return
        result = self.result  # The report belongs to these captured results, not live Setup.
        path = filedialog.asksaveasfilename(title='Export ' + self.study_name + ' study', defaultextension='.json', filetypes=[('JSON report', '*.json')])
        if not path: return
        self.host._run_background_task('Export ' + self.study_name + ' study', lambda: export_tolerance_report(Path(path), result),
                                       lambda _: self.host.status_var.set(f'Saved {self.study_name} study: {path}'), 'Export error')

    def export_csv(self):
        if self.host.job_is_running() or self.result is None: return
        result = self.result
        path = filedialog.asksaveasfilename(title='Export thickness envelope curves', defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path: return
        from .export_paths import polarization_paths
        from .thickness_envelopes import export_envelope_files
        paths = list(polarization_paths(path, result['polarizations']).values())
        if not self.host._confirm_output_replacements(paths, operation='Thickness envelope export'): return
        self.host._run_background_task('Export thickness envelopes', lambda: export_envelope_files(path,result),
            lambda files: self.progress_label.setText('Exported: ' + ', '.join(str(p) for p in files)), 'Envelope export error')
