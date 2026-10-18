"""Robust inverse controls and captured verification reports."""
import copy
import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QDialog, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView)

from .robust_design import ROBUST_DEFAULTS, validate_robust_setup, verify_candidates
from .tolerance_analysis import export_tolerance_report
from .io import layer_config_to_dict
from .ui_controls import messagebox, filedialog


class RobustSettings(QWidget):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self.fields = {}
        for i, (key, label, options) in enumerate([
            ('samples', 'Joint tolerance samples', [str(2**i) for i in range(4, 12)]),
            ('seed', 'Sampling seed', None),
            ('repeats', 'Verification yield repeats', [str(i) for i in range(2, 17)]),
            ('levels', 'Verification refinements', [str(i) for i in range(1, 5)])]):
            edit = QComboBox() if options else QLineEdit()
            if options: edit.addItems(options)
            (edit.setCurrentText if options else edit.setText)(str(ROBUST_DEFAULTS[key]))
            self.fields[key] = edit
            edit.setToolTip({'samples': 'Number of joint manufacturing probes per candidate. These sample material tolerances; nominal design ranges still evaluate every defined combination.',
                'seed': 'Integer seed for repeatable manufacturing samples shared across candidate designs.',
                'repeats': 'Independent yield trials; configured in Verify candidates.',
                'levels': 'Finer operating grids; configured in Verify candidates.'}[key])
            grid.addWidget(QLabel(label), i//2, 2*(i%2))
            grid.addWidget(edit, i//2, 2*(i%2)+1)
            if key in ('repeats', 'levels'):
                grid.itemAtPosition(i//2, 2*(i%2)).widget().hide()
                edit.hide()  # Stored for project compatibility; edited in verification.
            (edit.currentTextChanged if options else edit.textChanged).connect(host._schedule_inverse_work_count)
        note = QLabel('Per-layer objectives use the manufacturing bounds edited below. These are separate from the exhaustive Thickness ranges. Worst-gap ranking samples the rectangular bounds; yield ranking uses the chosen distributions/groups. Shared stack variation is disabled in this mode. Verification uses fresh seeds and finer grids; sampled bounds are not guarantees.')
        note.setWordWrap(True)
        grid.addWidget(note, 2, 0, 1, 4)
        button = QPushButton('Edit per-layer tolerance bounds…')
        button.clicked.connect(lambda: host.tolerance_workspace.edit_manufacturing_bounds())
        grid.addWidget(button, 3, 0, 1, 4)

    def capture(self, *, verification=False):
        from .ui_options import MANUFACTURING_OBJECTIVES
        values = dict(ROBUST_DEFAULTS)
        stored = self.stored_values()
        for key in ('repeats', 'levels'): values[key] = stored[key]
        keys = self.fields if verification else ('samples', 'seed') if self.host.inv_score_mode_var.get() in MANUFACTURING_OBJECTIVES else ()
        for k in keys:
            v = self.fields[k]
            values[k] = v.currentText() if isinstance(v, QComboBox) else v.text()
        return validate_robust_setup(values)

    def stored_values(self):
        # Preserve valid preferences without allowing inactive unfinished edits
        # to prevent saving an unrelated nominal study.
        values = dict(ROBUST_DEFAULTS)
        for k, v in self.fields.items():
            raw = v.currentText() if isinstance(v, QComboBox) else v.text()
            try: values[k] = validate_robust_setup({**values, k: raw})[k]
            except ValueError: pass
        return values

    def restore(self, values):
        for key, value in validate_robust_setup(values).items():
            field = self.fields[key]
            (field.setCurrentText if isinstance(field, QComboBox) else field.setText)(str(value))


class VerificationDialog(QDialog):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.result = None
        self.active = False
        self.stop_event = threading.Event()
        self.progress_value = None
        self.last_error = None
        self.setWindowTitle('Verify inverse candidates')
        self.resize(1000, 560)
        layout = QVBoxLayout(self)
        note = QLabel('Checks the captured reflection target using nested frequency/angle grids and increasing joint tolerance samples. Discrete frequency targets remain discrete. Independent yield repeats use the finest operating grid. This comparison covers retained candidates only; it does not prove continuous acceptance.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.choice = QComboBox()
        self.choice.addItems(['All retained candidates', 'Selected candidate'])
        layout.addWidget(self.choice)
        settings = QGridLayout()
        self.fields = {}
        self.field_labels = {}
        for row, (key, label) in enumerate((('samples', 'Joint tolerance samples'), ('seed', 'Sampling seed'),
                ('repeats', 'Independent yield repeats'), ('levels', 'Frequency/angle refinements'))):
            field = QLineEdit(str(host.robust_settings.stored_values()[key]))
            field.setToolTip({'samples': 'Joint tolerance cases per verification level; must be a supported power of two.',
                'seed': 'Reproducible sampling seed. Verification uses fresh scenarios distinct from ranking.',
                'repeats': 'Independent trials used to estimate variation in modeled manufacturing yield.',
                'levels': 'Number of finer frequency/angle grids. Discrete frequency targets remain discrete.'}[key])
            self.fields[key] = field
            self.field_labels[key] = QLabel(label)
            settings.addWidget(self.field_labels[key], row, 0); settings.addWidget(field, row, 1)
        layout.addLayout(settings)
        self.status = QLabel('Choose verification settings, then run verification. These do not change the search ranking.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['Candidate', 'Worst margin (dB)', 'Last change (dB)', 'Modeled yield (%)', 'Repeat range (%)', 'Sampling SE (points)'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.table.itemSelectionChanged.connect(self.show_details)
        actions = QHBoxLayout()
        self.run_button = QPushButton('Run verification')
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_event.set)
        self.export_button = QPushButton('Export verification JSON…')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        self.select_button = QPushButton('Select candidate in search results')
        self.select_button.clicked.connect(self.select_candidate)
        self.select_button.setEnabled(False)
        for button in (self.run_button, self.stop_button, self.export_button, self.select_button): actions.addWidget(button)
        layout.addLayout(actions)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_progress)
        self.sync_settings()

    def sync_settings(self):
        from .tolerance_config import validate_tolerances
        has_parameters = any(any(v['bound'] > 0 for v in validate_tolerances(c.tolerances, c.is_sheet).values()) for c in self.host.layers)
        try:
            shared = self.host._read_inverse_uncertainty_config()
            sampling = has_parameters or (shared.enabled and any((shared.thickness_pct, shared.eps_pct, shared.mu_pct)))
        except ValueError:
            sampling = True  # Run performs strict validation; keep corrections accessible.
        self.applicable_fields = {'levels'} | ({'samples', 'seed'} if sampling else set()) | ({'repeats'} if has_parameters else set())
        for key in self.fields:
            self.fields[key].setVisible(key in self.applicable_fields)
            self.field_labels[key].setVisible(key in self.applicable_fields)

    def showEvent(self, event):
        self.sync_settings()
        super().showEvent(event)

    def reject(self):
        if self.active:
            self.stop_event.set()
            self.status.setText('Stopping verification before closing; previous results retained.')
            return
        super().reject()

    def update_progress(self):
        if self.progress_value:
            done, total, phase = self.progress_value
            self.status.setText(f'{done}/{total} candidates · {phase}')
        if not self.host.job_is_running() and self.active:
            self.active = False
            self.timer.stop()
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.export_button.setEnabled(self.result is not None)
            self.select_button.setEnabled(self.result is not None)
            if self.stop_event.is_set(): self.status.setText('Stopped; previous captured verification retained.')
            elif self.last_error: self.status.setText('Verification failed; previous captured report retained. ' + self.last_error)

    def run(self):
        if self.host.job_is_running(): return
        try:
            self.host._ensure_inverse_result_current()
            configs, freqs, angles, cfg = self.host._inverse_setup_values()
            all_candidates = copy.deepcopy(self.host.inverse_candidates)
            indices = list(range(len(all_candidates))) if self.choice.currentIndex()==0 else [self.host._selected_inverse_index()]
            if not indices or min(indices)<0: raise ValueError('Select a candidate first.')
            candidates = [all_candidates[i] for i in indices]
            self.sync_settings()
            setup = validate_robust_setup({k: v.text() for k, v in self.fields.items() if k in self.applicable_fields})
            # Save verification preferences without changing ranking controls.
            for key in ('repeats', 'levels'):
                self.host.robust_settings.fields[key].setCurrentText(str(setup[key]))
            target = self.host.inverse_result_metadata.get('requirement_db')
            if target is None: target = self.host.inv_target_db.value()
            pol = self.host.inv_wave_pol_var.get()
            continuous = self.host.inverse_result_metadata.get('band_sweep', False)
            identity = self.host._inverse_result_identity
        except Exception as exc:
            messagebox.showerror('Verification setup', str(exc)); return
        self.stop_event.clear()
        self.active = True
        self.progress_value = None
        self.last_error = None
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.timer.start()
        def progress(*values): self.progress_value = values
        def worker():
            try:
                result = verify_candidates(configs, candidates, freqs, angles, pol, cfg, target, setup,
                                           continuous=continuous, stop_requested=self.stop_event.is_set, progress=progress)
            except Exception as exc:
                self.last_error = str(exc)
                self.progress_value = None
                raise
            result['search_identity'] = identity
            result['search_candidate_indices'] = indices
            return result
        def success(result):
            self.result = result
            self.table.setSortingEnabled(False)
            self.table.setRowCount(len(result['candidates']))
            for row, candidate in enumerate(result['candidates']):
                estimate = candidate['yield_estimate']
                values = [indices[row]+1, candidate['worst_margin_db'], candidate['last_change_db'],
                          estimate['mean_pass_pct'] if estimate else None,
                          f'{estimate["min_pass_pct"]:.2f}–{estimate["max_pass_pct"]:.2f}' if estimate else '—',
                          estimate['standard_error_percentage_points'] if estimate else None]
                for col, value in enumerate(values):
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, '—' if value is None else round(value, 5) if isinstance(value, float) else value)
                    item.setData(Qt.UserRole, row)
                    self.table.setItem(row, col, item)
            self.table.setSortingEnabled(True)
            self.table.sortItems(1, Qt.DescendingOrder)
            self.table.selectRow(0)
            self.progress_value = None
            self.status.setText(f'Verified {len(indices)} retained candidates. Sort by margin or modeled yield. Bounds remain sampled; repeat spread measures sampling variability only.')
        self.host._run_background_task('Verify inverse candidates', worker, success, 'Candidate verification error')

    def current_result(self):
        row = self.table.currentRow()
        return self.table.item(row, 0).data(Qt.UserRole) if row>=0 and self.table.item(row,0) else None

    def show_details(self):
        index = self.current_result()
        if index is None or not self.result: return
        c = self.result['candidates'][index]
        w = c['worst_condition']
        self.details.setText(f'Worst sampled condition: {w["frequency_ghz"]:g} GHz, {w["angle_deg"]:g}°, {w["polarization"].upper()}. '
                             + ('Last two refinement changes ≤0.05 dB. ' if c['sampled_stable'] else 'Refinement stability not established. ')
                             + self.result['support_model'] + '. No continuous guarantee; zero failures or zero repeat spread does not establish zero failure probability.')

    def select_candidate(self):
        index = self.current_result()
        if index is None or self.host.job_is_running(): return
        try:
            self.host._ensure_inverse_result_current()
            if self.host._inverse_result_identity != self.result['search_identity'] or [layer_config_to_dict(c) for c in self.host._snapshot_layers()] != self.result['layers']:
                raise ValueError('Search inputs or tolerance bounds changed. Rerun verification before using this comparison.')
            self.host.inv_results_list.setCurrentRow(self.result['search_candidate_indices'][index])
            self.host.inverse_workspace_tabs.setCurrentIndex(1)
            self.accept()
        except Exception as exc: messagebox.showerror('Verification selection', str(exc))

    def export(self):
        if not self.result or self.active: return
        path = filedialog.asksaveasfilename(title='Export verification', defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            try: export_tolerance_report(path, self.result)
            except Exception as exc: messagebox.showerror('Verification export', str(exc))
