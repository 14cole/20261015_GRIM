"""No-code recording, review, and batch replay of dataset transforms."""
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from GRIM_Backend.datasets.work_control import checkpoint, WorkCancelled
from GRIM_Backend.scripting.action_recipe import ActionCapture, ActionRecipe, step_label
from GRIM_Backend.ui.dataset_actions import DATASET_ID_ROLE


class ActionRecorderDialog(QDialog):
    SETTINGS_KEY = 'automation/action-recipe-v1'

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._replay_pending = False
        self.capture = ActionCapture(self._changed)
        self.setWindowTitle('Record Actions')
        self.setModal(False)
        self.resize(590, 590)
        layout = QVBoxLayout(self)
        hint = QLabel(
            'Select one example dataset, then record your dataset operations. '
            'Each new result is selected automatically so the next operation continues '
            'the workflow. Stop, select one or more target datasets, and run the recipe.\n\n'
            'Records transforms such as Crop, Regrid, Offset, Wrap, Statistics, and '
            'Time Gate. Coordinates keep the recording’s axis units. Plotting, overlays, '
            'file actions, and solver tools are not recorded. Combining multiple input '
            'datasets is not supported.'
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.name = QLineEdit('Recorded actions')
        self.name.setPlaceholderText('Recipe name')
        layout.addWidget(self.name)
        bar = QHBoxLayout()
        self.record = QPushButton('Record new workflow')
        self.stop = QPushButton('Stop recording')
        self.load = QPushButton('Load…')
        self.save = QPushButton('Save As…')
        for button in (self.record, self.stop, self.load, self.save):
            bar.addWidget(button)
        layout.addLayout(bar)
        self.message = QLabel('Ready to record.')
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        layout.addWidget(self.message)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(120)
        self.report.hide()
        layout.addWidget(self.report)
        self.keep = QPushButton('Keep only the recorded steps')
        layout.addWidget(self.keep)
        self.steps = QListWidget()
        layout.addWidget(self.steps, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('Select a step to review its parameters.')
        self.details.setMaximumHeight(120)
        layout.addWidget(self.details)
        footer = QHBoxLayout()
        self.run = QPushButton('Run on selected datasets')
        self.close_button = QPushButton('Close')
        footer.addWidget(self.run, 1)
        footer.addWidget(self.close_button)
        layout.addLayout(footer)
        self.record.clicked.connect(self.start_recording)
        self.stop.clicked.connect(self.stop_recording)
        self.keep.clicked.connect(self.keep_steps)
        self.run.clicked.connect(self.run_selected)
        self.load.clicked.connect(self.load_recipe)
        self.save.clicked.connect(self.save_recipe)
        self.close_button.clicked.connect(self.hide)
        self.name.editingFinished.connect(self._rename)
        self.steps.currentRowChanged.connect(self._show_details)
        window.btn_record_actions.clicked.connect(self.open_recorder)
        window.table.selectionModel().selectionChanged.connect(self._update_buttons)
        window.python_recorder.action_listener = self._capture_action
        saved = window._settings.value(self.SETTINGS_KEY, '')
        if saved:
            try:
                self.capture.recipe = ActionRecipe.loads(str(saved))
                self.capture.error = str(window._settings.value(self.SETTINGS_KEY + '/error', ''))
                self.name.setText(self.capture.recipe.name)
                self.message.setText('Restored the last recipe. Select targets to run it.')
            except (ValueError, TypeError, KeyError, RecursionError):
                self.message.setText('The saved recipe could not be restored. Load a recipe or record a new one.')
        self._refresh()
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._update_buttons)
        self.timer.start()

    def _busy(self):
        return self.window._background_job_active() or bool(getattr(self.window, '_isar_busy', False))

    def _update_buttons(self):
        busy = self._busy()
        if self._replay_pending and not busy:
            self._replay_pending = False
            self.message.setText(self.window.status.currentMessage())
        recording = self.capture.recording
        count = len(self.window.table.selectionModel().selectedRows())
        usable = bool(self.capture.recipe.steps) and not recording and not self.capture.error
        self.record.setEnabled(not busy and not recording and count == 1)
        self.stop.setEnabled(recording and not busy)
        self.load.setEnabled(not busy and not recording)
        self.save.setEnabled(usable and not busy)
        self.run.setEnabled(usable and count > 0 and not busy)
        self.run.setText(f'Run on {count} selected dataset(s)')
        self.keep.setVisible(bool(self.capture.error))
        self.keep.setEnabled(not busy)
        self.name.setReadOnly(recording or busy)

    def _refresh(self):
        selected = self.steps.currentRow()
        self.steps.clear()
        for index, step in enumerate(self.capture.recipe.steps, start=1):
            self.steps.addItem(f'{index}. {step_label(step)}')
        if self.steps.count():
            self.steps.setCurrentRow(min(max(selected, 0), self.steps.count() - 1))
        else:
            self.details.clear()
        count = len(self.capture.recipe.steps)
        self.window.btn_record_actions.setText(
            f'Stop Recording ({count})' if self.capture.recording else 'Record Actions…')
        if self.capture.error:
            self.message.setText(self.capture.error)
        elif self.capture.recording:
            self.message.setText(f'Recording: {count} step(s). Use Dataset Operations to continue.')
        self._update_buttons()

    def _changed(self):
        if self.capture.recipe.steps:
            self.window._settings.setValue(self.SETTINGS_KEY, self.capture.recipe.dumps())
        else:
            self.window._settings.setValue(self.SETTINGS_KEY, '')
        self.window._settings.setValue(self.SETTINGS_KEY + '/error', self.capture.error)
        self._refresh()

    def _rename(self):
        self.capture.recipe.name = self.name.text().strip() or 'Recorded actions'
        self.name.setText(self.capture.recipe.name)
        self._changed()

    def _show_details(self, index):
        if not 0 <= index < len(self.capture.recipe.steps):
            self.details.clear()
            return
        step = self.capture.recipe.steps[index]
        from GRIM_Backend.scripting.action_recipe import FUNCTIONS, METHODS
        import inspect
        registry = FUNCTIONS if step['kind'] == 'function' else METHODS
        bound = inspect.signature(registry[step['name']]).bind(None, *step['args'], **step['kwargs'])
        values = list(bound.arguments.items())[1:]
        self.details.setPlainText('\n'.join(
            f'{key.replace("_", " ").capitalize()}: {value}' for key, value in values
        ) or 'No additional parameters.')

    def open_recorder(self):
        if self.capture.recording:
            self.stop_recording()
        self._refresh()
        self.show()
        self.raise_()
        self.activateWindow()

    def start_recording(self):
        if self._busy():
            self.message.setText('Wait for the current operation to finish before recording.')
            return
        datasets = self.window._selected_datasets_ordered()
        if not datasets or len(datasets) != 1:
            self.message.setText('Select exactly one example dataset to record a workflow.')
            return
        _, dataset = datasets[0]
        reference = self.window._python_reference_for_dataset(dataset)
        self.capture.start(reference, dataset, self.name.text().strip() or 'Recorded actions')
        self.report.hide()
        self.window.status.showMessage('Recording dataset actions. Click Stop Recording when finished.')
        self.hide()

    def stop_recording(self):
        if self._busy():
            self.message.setText('Wait for the current operation to finish so its result is recorded.')
            return
        self.capture.stop()
        self.message.setText(f'Recorded {len(self.capture.recipe.steps)} step(s). Select targets, then Run.')
        self._refresh()

    def _capture_action(self, outputs, sources, steps, label):
        was_recording = self.capture.recording
        self.capture.capture(outputs, sources, steps, label)
        if was_recording and self.capture.error:
            self.show()
            self.raise_()
        elif was_recording and self.capture.recording:
            dataset_id = self.capture.current_id
            # The publishing loop must finish before changing selection.
            QTimer.singleShot(0, lambda: self._select_result(dataset_id))

    def _select_result(self, dataset_id):
        if not self.capture.recording or dataset_id != self.capture.current_id:
            return
        for row in range(self.window.table.rowCount()):
            item = self.window.table.item(row, 0)
            if item.data(DATASET_ID_ROLE) == dataset_id:
                self.window.table.selectRow(row)
                break

    def keep_steps(self):
        self.capture.error = ''
        self.message.setText('Only the listed steps will run. Select targets, then Run.')
        self._changed()

    def save_recipe(self):
        if self.capture.recording or self.capture.error or not self.capture.recipe.steps:
            return
        self._rename()
        path, _ = QFileDialog.getSaveFileName(self, 'Save action recipe', '', 'GRIM action recipe (*.grim-actions.json)')
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.grim-actions.json'
        try:
            from PySide6.QtCore import QSaveFile, QIODevice
            output = QSaveFile(path)
            payload = self.capture.recipe.dumps().encode('utf-8')
            if not output.open(QIODevice.WriteOnly):
                raise OSError(output.errorString())
            if output.write(payload) != len(payload) or not output.commit():
                raise OSError(output.errorString())
            self.message.setText(f'Saved {Path(path).name}.')
        except (OSError, ValueError) as exc:
            self.message.setText(f'Could not save recipe: {exc}')

    def load_recipe(self):
        if self._busy() or self.capture.recording:
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Load action recipe', '', 'GRIM action recipe (*.json)')
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as stream:
                recipe = ActionRecipe.loads(stream.read(10_000_001))
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            self.message.setText(f'Could not load recipe: {exc}')
            return
        self.capture.recipe = recipe
        self.capture.error = ''
        self.report.hide()
        self.name.setText(recipe.name)
        self.message.setText(f'Loaded {len(recipe.steps)} step(s). Select targets, then Run.')
        self._changed()

    def run_selected(self):
        if self._busy() or self.capture.recording or self.capture.error or not self.capture.recipe.steps:
            return
        datasets = self.window._selected_datasets_ordered(use_selection_order=True)
        if not datasets:
            self.message.setText('Select one or more target datasets in the table.')
            return
        self._rename()
        recipe = ActionRecipe.loads(self.capture.recipe.dumps())
        references = [self.window._python_reference_for_dataset(ds) for _, ds in datasets]
        if not self.window._preflight_derived_outputs('Replay actions', [
            (ds, tuple(int(n) for n in ds.power.shape)) for _, ds in datasets
        ]):
            return

        def compute(progress):
            results, failures = [], []
            total = len(datasets) * len(recipe.steps)
            for index, (name, dataset) in enumerate(datasets):
                checkpoint()
                try:
                    result = recipe.run(dataset, lambda done, count, label: progress(
                        index * count + done, total, f'{name}: {label}'))
                except WorkCancelled:
                    raise
                except Exception as exc:
                    failures.append(f'{name}: {exc}')
                else:
                    results.append((index, name, result))
                progress((index + 1) * len(recipe.steps), total, name)
            return results, failures

        def publish(payload):
            self._replay_pending = False
            results, failures = payload
            for index, name, result in results:
                output_name = f'{name} [{recipe.name}]'
                output_id = self.window._add_dataset_row(
                    result, output_name,
                    f'Action recipe {recipe.name}: ' + ' → '.join(map(step_label, recipe.steps)), '')
                source = references[index]
                if source is not None:
                    self.window.python_recorder.record_expression(
                        self.window._python_output_reference(output_id, output_name), [source],
                        lambda variables: recipe.expression(variables[0]),
                        comment=f'Action recipe: {recipe.name}', action_steps=recipe.steps)
            message = f'Created {len(results)} dataset(s); {len(failures)} failed.'
            self.window.status.showMessage(message)
            self.message.setText(message)
            self.report.setPlainText('\n'.join(failures))
            self.report.setVisible(bool(failures))
            self.show()
            self._update_buttons()

        if self.window._start_background_callable('Replay actions', compute, publish, reports_progress=True):
            self._replay_pending = True
            self.report.hide()
            self.message.setText(f'Running {len(recipe.steps)} steps on {len(datasets)} dataset(s). Use Cancel Job to stop.')
            self._update_buttons()
