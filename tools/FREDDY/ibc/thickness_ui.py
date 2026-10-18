"""One exhaustive thickness workflow: individual layers and their joint grid."""
from __future__ import annotations

import copy
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QLineEdit, QTableWidgetItem, QHeaderView,
                              QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                              QDialog, QPushButton)

from .tolerance_ui import ToleranceWidget
from .tolerance_config import ENVELOPE_MODE, SETUP_DEFAULTS
from .thickness_envelopes import thickness_axes, envelope_workload, case_thicknesses
from .tolerance_analysis import requirement_margin
from .analysis_data import SweepResult
from .plot import style_axis
from .ui_controls import messagebox


class ThicknessWidget(ToleranceWidget):
    retain_layer_metrics = True
    study_name = 'Thickness'

    def __init__(self, host):
        self.detail_panel = None
        self._range_refresh = False
        super().__init__(host)
        self.copy_inverse.hide()
        self.field_labels['a_start'].setText('Incidence from normal (°)')
        self.run_button.setText('Analyze every combination')
        self.result_status.setText('Run the layer ranges to view individual and combined results.')
        self.edit_range.setText('Edit selected layer…')
        self.input_note.setText('Select layers to vary and enter minimum, maximum and step in inches. '
            'Every defined value is evaluated individually and in every joint combination. '
            'Unchecked layers stay at their nominal thickness. Material properties and sheet resistances stay nominal.')
        self.thickness_ranges.setMinimumHeight(220)
        self.thickness_ranges.setToolTip('Ranges are shared with Inverse Design. Values start at minimum and advance by step without exceeding maximum. '
            'An off-step maximum is excluded. Nominal is a separate reference.')
        self.summary.setColumnCount(5)
        self.summary.setHorizontalHeaderLabels(['Study', 'Best peak (dB)', 'Worst peak (dB)', 'Worst margin (dB)', 'Passing grid cases'])
        self.summary.setWordWrap(False)
        for col in (1,2): self.summary.horizontalHeaderItem(col).setToolTip('Peak = largest reflection over all selected frequencies, angles and polarizations. Best minimizes that peak; worst maximizes it.')
        self.summary.setMaximumHeight(185)
        self.summary.setSortingEnabled(False)
        self.summary.itemSelectionChanged.disconnect()
        self.summary.itemSelectionChanged.connect(self.select_summary)
        self.stack_details = QLabel()
        self.stack_details.setWordWrap(True)
        self.tabs.widget(1).layout().addWidget(self.stack_details)
        self.point_details = QLabel('Click an envelope point to inspect the thickness combinations producing its bounds.')
        self.point_details.setWordWrap(True)
        self.tabs.widget(1).layout().addWidget(self.point_details)
        self.canvas.mpl_connect('button_press_event', self.inspect_point)
        for choice in (self.view,self.parameter_choice,self.pol_choice,self.angle_choice):
            choice.currentIndexChanged.connect(lambda *_: self.point_details.clear())
        self.fields['polarization'].setToolTip('Evaluate TE, TM, or both. Complete-stack best/worst and pass counts include every selected polarization.')
        self.fields['target'].setToolTip('Maximum allowed PEC reflection in dB. A grid stack passes only if every selected frequency, angle and polarization meets this limit. More negative means less reflection.')
        self.refresh_workload()

    def capture_setup(self):
        state = super().capture_setup()
        return {**{k:v for k,v in state.items() if k not in ('points','samples','seed','repeats','mode')}, 'mode':ENVELOPE_MODE}

    def restore_setup(self, setup):
        # Old sensitivity projects retain their layer tolerance metadata, but
        # hidden sampling controls are never used to configure a thickness run.
        self._range_refresh=True
        try: super().restore_setup({**SETUP_DEFAULTS, **setup, 'mode':ENVELOPE_MODE})
        finally: self._range_refresh=False
        self.refresh_layers()

    def _refresh_setup_visibility(self, *_):
        hidden = {'mode','points','samples','seed','repeats'}
        for key, edit in self.fields.items():
            caption=self.field_labels[key]
            self.setup_grid.removeWidget(caption); self.setup_grid.removeWidget(edit)
            caption.setVisible(key not in hidden); edit.setVisible(key not in hidden)
        row=0
        for entries in self.field_rows:
            keys=[key for key,_ in entries if key not in hidden]
            if not keys: continue
            for col,key in enumerate(keys):
                self.setup_grid.addWidget(self.field_labels[key],row,col*2)
                self.setup_grid.addWidget(self.fields[key],row,col*2+1)
            row+=1

    def refresh_ranges(self):
        self._range_refresh=True
        try:
            for controls in getattr(self, 'range_controls', []): controls.deleteLater()
            table=self.thickness_ranges
            table.setRowCount(0);table.setColumnCount(6)
            table.setHorizontalHeaderLabels(['Layer','Vary','Nominal (in)','Minimum (in)','Maximum (in)','Step (in)'])
            for col in range(1,6): table.setColumnWidth(col,85 if col==1 else 110)
            table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch)
            self.range_controls = []
            for index,layer in enumerate(self.host.layers):
                if layer.is_sheet: continue
                row=table.rowCount();table.insertRow(row)
                from .compute import layer_material_label
                item=QTableWidgetItem(f'{index+1}. {layer_material_label(layer)}')
                item.setData(Qt.UserRole,index);table.setItem(row,0,item)
                from .range_controls import RangeControls
                controls = RangeControls(self, (layer.inv_t_min_in,layer.inv_t_max_in,layer.inv_t_accuracy_in), 'in')
                self.range_controls.append(controls)
                toggle=controls.toggle
                toggle.setToolTip('Evaluate this layer’s entire defined range. Uncheck to hold it at nominal thickness.')
                table.setCellWidget(row,1,toggle)
                table.setItem(row,2,QTableWidgetItem(f'{layer.thickness_in:.12g}'))
                for col,edit in zip((3,4,5),controls.fields):
                    edit.setToolTip('Positive thickness in inches. Minimum = maximum defines one value.' if col!=5 else
                                    'Spacing in inches. All minimum + k × step values up to maximum are evaluated; an off-step maximum is excluded.')
                    table.setCellWidget(row,col,edit)
                    edit.textChanged.connect(self.refresh_workload)
                    edit.editingFinished.connect(self.commit_if_valid)
                toggle.toggled.connect(self.range_toggled)
            if table.rowCount(): table.selectRow(0)
        finally: self._range_refresh=False

    def range_toggled(self, *_):
        self.commit_if_valid()
        self.refresh_workload()

    def range_snapshot(self):
        layers=copy.deepcopy(self.host.layers)
        for row in range(self.thickness_ranges.rowCount()):
            index=self.thickness_ranges.item(row,0).data(Qt.UserRole)
            layer=layers[index]
            enabled=self.thickness_ranges.cellWidget(row,1).isChecked()
            if enabled:
                try:
                    low,high,step=(self.thickness_ranges.cellWidget(row,col).text().strip() for col in (3,4,5))
                    layer.inv_t_min_in=float(low);layer.inv_t_max_in=float(high)
                    layer.inv_t_accuracy_in=float(step) if step else None
                except ValueError:
                    raise ValueError(f'Layer {index+1}: enter numeric minimum, maximum and step.') from None
            else:
                layer.inv_t_min_in=layer.inv_t_max_in=layer.inv_t_accuracy_in=None
            thickness_axes([layer])
        return layers

    def commit_ranges(self):
        if self._range_refresh: return
        snapshot=self.range_snapshot()
        for old,new in zip(self.host.layers,snapshot):
            old.inv_t_min_in,old.inv_t_max_in,old.inv_t_accuracy_in=new.inv_t_min_in,new.inv_t_max_in,new.inv_t_accuracy_in
        if hasattr(self.host,'robust_settings'): self.host._schedule_inverse_work_count()

    def commit_if_valid(self):
        if self._range_refresh or self.host.job_is_running(): return
        try: self.commit_ranges()
        except ValueError: pass  # Show unfinished edits in workload; Run/Save reject them.

    def refresh_workload(self, *_):
        if self._refreshing or self._range_refresh or len(self.fields)!=len(SETUP_DEFAULTS): return
        self.fields['mode'].blockSignals(True)
        self.fields['mode'].setCurrentText(ENVELOPE_MODE)
        self.fields['mode'].blockSignals(False)
        super().refresh_workload()
        self.run_button.setText('Analyze every combination')
        self.input_note.setText('Select layers to vary. Each layer is swept individually with the others nominal, then all defined combinations are evaluated together. '
                               'Ranges are shared with Inverse Design. Material properties and sheet resistances stay nominal.')
        try:
            for row in range(self.thickness_ranges.rowCount()):
                enabled=self.thickness_ranges.cellWidget(row,1).isChecked()
                for col in (3,4,5): self.thickness_ranges.cellWidget(row,col).setEnabled(enabled and not self.host.job_is_running())
            snapshot=self.range_snapshot()
            w=envelope_workload(snapshot,self.capture_setup(),retain_layer_metrics=True)
            self.workload.setText(f'{w["joint_combinations"]:,} joint combinations · {w["evaluations"]:,} total evaluations including individual sweeps and nominal · '
                                  f'{w["response_points"]:,} response points · approximately {w["array_mib"]:.1f} MiB of arrays. No combinations are sampled or skipped.')
        except ValueError as exc: self.workload.setText(str(exc))

    def run(self):
        if self.host.job_is_running(): return
        try: self.commit_ranges()
        except ValueError as exc:
            messagebox.showerror('Thickness ranges',str(exc));return
        super().run()

    def attach_layer_details(self,panel):
        self.detail_panel=panel
        page=QWidget();layout=QVBoxLayout(page);bar=QHBoxLayout()
        self.detail_layer=QComboBox();self.detail_angle=QComboBox()
        bar.addWidget(QLabel('Individual layer'));bar.addWidget(self.detail_layer,1)
        bar.addWidget(QLabel('Incidence angle'));bar.addWidget(self.detail_angle)
        layout.addLayout(bar);layout.addWidget(panel,1)
        panel.show()  # Reparenting a hidden stacked page retains its hidden flag.
        self.tabs.addTab(page,'Layer details')
        self.detail_layer.currentIndexChanged.connect(self.show_layer_details)
        self.detail_angle.currentIndexChanged.connect(self.show_layer_details)
        self.tabs.setTabEnabled(2,False)

    def publish(self):
        super().publish()
        self.view.blockSignals(True)
        self.view.clear();self.view.addItems(['Thickness envelopes','Whole-band best/worst stacks','Layer sensitivity'])
        self.view.blockSignals(False)
        self.summary.setSortingEnabled(False)
        self.summary.setRowCount(len(self.result['panels']))
        for row,p in enumerate(self.result['panels']):
            for col,value in enumerate([p['label'],p['best_stack']['worst_reflection_db'],p['worst_stack']['worst_reflection_db'],
                                        p['worst_stack']['margin_db'],f'{p["passing_cases"]:,} / {p["cases"]:,}']):
                item=QTableWidgetItem(f'{value:.6g}' if isinstance(value,float) else value)
                item.setData(Qt.UserRole,row);self.summary.setItem(row,col,item)
        if self.detail_panel is not None:
            self.detail_panel.target.blockSignals(True)
            self.detail_panel.target.setValue(self.result['target_db'])
            self.detail_panel.target.blockSignals(False)
            self.detail_panel.target.setToolTip('Display threshold for individual-layer bandwidth and contours. Starts at the captured study limit; editing it does not change the exhaustive study ranking or pass counts.')
            self.detail_layer.blockSignals(True);self.detail_angle.blockSignals(True)
            self.detail_layer.clear();self.detail_layer.addItems([p['label'] for p in self.result['panels'][:-1]])
            self.detail_angle.clear();self.detail_angle.addItems([f'{a:g}°' for a in self.result['angles_deg']])
            self.detail_layer.blockSignals(False);self.detail_angle.blockSignals(False)
            self.tabs.setTabEnabled(2,True);self.show_layer_details()
        self.draw()

    def show_layer_details(self,*_):
        if self.result is None or self.detail_panel is None: return
        p=self.result['panels'][max(0,self.detail_layer.currentIndex())]
        ai=max(0,self.detail_angle.currentIndex());angle=self.result['angles_deg'][ai]
        metrics={pol.upper():{k:v[:,pi,ai,:].T for k,v in p['metrics'].items()} for pi,pol in enumerate(self.result['polarizations'])}
        data=SweepResult(list(self.result['frequencies_ghz']),list(p['thickness_values_in']),'Thickness (in)',
            f'{p["label"]} · {angle:g}° incidence · other layers nominal',metrics,
            polarization=self.result['polarizations'][0].upper(),
            summary=self.result['interpretation'])
        self.detail_panel.set_result(data)

    def select_summary(self):
        item=self.summary.item(self.summary.currentRow(),0)
        if item is not None: self.parameter_choice.setCurrentIndex(item.data(Qt.UserRole)+1)

    def clear_result(self):
        super().clear_result()
        self.result_status.setText('Run the layer ranges to view individual and combined results.')
        if hasattr(self,'stack_details'): self.stack_details.clear();self.point_details.clear()
        if self.detail_panel is not None:
            self.tabs.setTabEnabled(2,False);self.detail_panel.result=None;self.detail_panel.draw()

    def draw(self,*_):
        if not hasattr(self,'figure'): return
        if self.result is None or self.view.currentText()=='Thickness envelopes':
            super().draw()
        else:
            r=self.result;self.figure.clear();ax=self.figure.add_subplot(111)
            if self.view.currentText()=='Layer sensitivity':
                panels=sorted(r['panels'][:-1],key=lambda p:p['margin_loss_db'])
                ax.barh([p['label'] for p in panels],[p['margin_loss_db'] for p in panels],color='#529ce8')
                ax.set_xlabel('Largest whole-band margin loss versus nominal (dB)')
                ax.set_title('Individual layer sensitivity · other layers nominal')
            else:
                p=r['panels'][max(0,self.parameter_choice.currentIndex()-1)] if self.parameter_choice.currentIndex() else r['panels'][-1]
                pi=max(0,self.pol_choice.currentIndex());ai=max(0,self.angle_choice.currentIndex()-1)
                for key,color,label in [('best_stack','#59c6a1','Best complete stack'),('worst_stack','#f4ad61','Worst complete stack')]:
                    grid=p[key]['reflection_db'][pi]
                    values=grid.max(axis=0) if self.angle_choice.currentIndex()==0 else grid[ai]
                    ax.plot(r['frequencies_ghz'],values,'.-',color=color,label=label)
                ax.axhline(r['target_db'],linestyle=':',color='#999999',label='Reflection Target')
                ax.set(xlabel='Frequency (GHz)',ylabel='PEC reflection (dB)',
                       title=p['label']+' · '+r['polarizations'][pi].upper()+' · '+self.angle_choice.currentText())
                ax.legend();ax.grid(alpha=.2)
            if self.colors: self.figure.set_facecolor(self.colors['plot_bg']);style_axis(ax,self.colors)
            self.canvas.draw_idle()
        self.summary.setVisible(self.result is not None)
        if hasattr(self,'stack_details') and self.result is not None:
            p=self.result['panels'][max(0,self.parameter_choice.currentIndex()-1)] if self.parameter_choice.currentIndex() else self.result['panels'][-1]
            def describe(key):
                s=p[key];point=s['limiting_point']
                thickness=', '.join(f'L{i+1}: {t:.12g} in' for i,t in enumerate(s['thicknesses_in']) if not self.result['layers'][i].get('is_sheet',False))
                return f'{thickness}; worst reflection {s["worst_reflection_db"]:.6g} dB at {point["frequency_ghz"]:g} GHz, {point["angle_deg"]:g}°, {point["polarization"].upper()}'
            self.stack_details.setText(p['label']+'\nBest complete stack: '+describe('best_stack')+'\nWorst complete stack: '+describe('worst_stack'))
        self.caveat.setText('Every defined grid combination is evaluated. Bounds may come from different stacks at different frequencies. '
            'Whole-band best/worst ranks each complete stack by its highest reflection across every selected frequency, angle and polarization. '
            'Passing grid cases are counts, not manufacturing yield. Refine steps to resolve extrema between grid points.')
        ranking=self.view.currentText()=='Layer sensitivity'
        self.parameter_choice.setVisible(not ranking);self.pol_choice.setVisible(not ranking);self.angle_choice.setVisible(not ranking)

    def inspect_point(self,event):
        if self.result is None or self.view.currentText()!='Thickness envelopes' or event.inaxes not in self.figure.axes or event.xdata is None: return
        r=self.result;selected=self.parameter_choice.currentIndex()
        p=r['panels'][self.figure.axes.index(event.inaxes)] if selected==0 else r['panels'][selected-1]
        pi=max(0,self.pol_choice.currentIndex());fi=int(np.argmin(np.abs(np.asarray(r['frequencies_ghz'])-event.xdata)))
        parts=[]
        for key,label in [('min','Minimum'),('max','Maximum')]:
            ai=max(0,self.angle_choice.currentIndex()-1)
            if self.angle_choice.currentIndex()==0:
                values=p[key+'_db'][pi,:,fi];ai=int(values.argmin() if key=='min' else values.argmax())
            case=int(p[key+'_case_index'][pi,ai,fi]);values=case_thicknesses(r,p,case)
            stack=', '.join(f'L{i+1}: {t:.12g} in' for i,t in enumerate(values) if not r['layers'][i].get('is_sheet',False))
            parts.append(f'{label} {p[key+"_db"][pi,ai,fi]:.6g} dB at {r["angles_deg"][ai]:g}°: {stack}')
        self.point_details.setText(f'{p["label"]} · {r["frequencies_ghz"][fi]:g} GHz · {r["polarizations"][pi].upper()}\n'+'\n'.join(parts))

    def edit_manufacturing_bounds(self):
        """Preserve the separate inverse-design manufacturing model editor."""
        if self.host.job_is_running(): return
        dialog=QDialog(self);dialog.setWindowTitle('Inverse design · manufacturing bounds');dialog.resize(1050,550)
        layout=QVBoxLayout(dialog)
        note=QLabel('These distribution-based bounds apply only to the per-layer inverse objectives and candidate verification. '
                    'The exhaustive Thickness study uses the layer min/max/step ranges instead.')
        note.setWordWrap(True);layout.addWidget(note)
        # Reuse the existing validated per-layer editor, without exposing the
        # retired sensitivity/stats analysis mode controls.
        editor=ToleranceWidget(self.host)
        editor.fields['mode'].setCurrentText('Sensitivity + statistical trials')
        layout.addWidget(editor.parameters);layout.addWidget(editor.group_note)
        close=QPushButton('Done');close.clicked.connect(dialog.accept);layout.addWidget(close)
        dialog.exec()
        editor.deleteLater();dialog.deleteLater()
        self.host._schedule_inverse_work_count()
