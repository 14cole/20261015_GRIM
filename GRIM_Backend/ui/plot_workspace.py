"""One plotting workspace with shared controls and retained plot/ISAR views."""
from PySide6.QtWidgets import QFrame, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout

from GRIM_Backend.ui.responsive_controls import ResponsiveButtonPanel, ResponsiveControlTabs


class PlotWorkspaceMixin:
    def _build_shared_plot_workspace(self, panel, groups_spec):
        layout = panel.layout()
        self._plot_header_stack = QStackedWidget()
        self._plot_header_stack.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._plot_view_stack = QStackedWidget()
        for key in ('plotting', 'isar'):
            self._plot_header_stack.addWidget(self._plot_headers[key])
            self._plot_view_stack.addWidget(self._plot_views[key])
            popup = self._plot_contexts[key].settings_frame
            popup.setParent(panel, popup.windowFlags())

        toolbar = QFrame()
        toolbar.setObjectName('plotToolbar')
        toolbar_layout = QVBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(4)
        self.plot_control_tabs = groups = ResponsiveControlTabs()
        groups.setHeaderActions(self._plot_header_stack)
        groups.setObjectName('plottingControlTabs')
        self._shared_manipulation_roles = []
        self._unavailable_plot_controls = {}
        for title, rows in groups_spec:
            page = ResponsiveButtonPanel()
            page.setObjectName({'Plots': 'plottingButtons', 'ISAR': 'isarButtons',
                                'Plot Manipulation': 'plotManipulationButtons'}[title])
            if title == 'Plot Manipulation':
                self._plot_manipulation_page = page
                self._shared_manipulation_roles = [role for row in rows for role in row]
            else:
                controls = self._plot_controls_by_tab['isar' if title == 'ISAR' else 'plotting']
                for row in rows:
                    for role in row:
                        button = controls[role]
                        page.addWidget(button)
                        button.show()
            groups.addTab(page, title)
        toolbar_layout.addWidget(groups)
        layout.addWidget(toolbar)
        layout.addWidget(self._plot_view_stack, 1)
        groups.currentChanged.connect(self._on_plot_control_page_changed)
        self._refresh_shared_plot_controls()

    def _on_plot_control_page_changed(self, index):
        # Manipulation preserves the current view, including its settings.
        if index in (0, 1):
            self._activate_plot_tab('plotting' if index == 0 else 'isar')
            self._update_plot_color_buttons()
            self.plot_canvas.draw_idle()

    def _refresh_shared_plot_controls(self):
        groups = self.plot_control_tabs
        key = self._active_plot_tab
        index = 0 if key == 'plotting' else 1
        self._plot_header_stack.setCurrentIndex(index)
        self._plot_view_stack.setCurrentIndex(index)
        if groups.currentIndex() != 2:
            blocked = groups.blockSignals(True)
            groups.setCurrentIndex(index)
            groups.blockSignals(blocked)

        # Reuse each context's wired buttons. Moving them does not toggle them,
        # rerun a plot, or mix the two contexts' saved state.
        for page, first in ((groups.header, 1), (self._plot_manipulation_page, 0)):
            while page.flow.count() > first:
                item = page.flow.takeAt(first)
                button = item.widget()
                button.hide()
                button.setParent(self._plot_control_storage)
                del item
        controls = self._plot_controls_by_tab[key]

        def button_for(role):
            if role in controls:
                return controls[role]
            if role not in self._unavailable_plot_controls:
                source = self._plot_controls_by_tab['plotting'][role]
                button = QToolButton(self._plot_control_storage)
                button.setText(source.text())
                button.setEnabled(False)
                button.setToolTip(f'{source.text()} applies to the Plots view.')
                self._unavailable_plot_controls[role] = button
            return self._unavailable_plot_controls[role]

        legend = self._plot_contexts[key].chk_plot_legend
        groups.addCommonWidget(legend)
        legend.show()
        for role in ('auto_plot', 'auto_scale', 'hold', 'clear'):
            button = button_for(role)
            groups.addCommonWidget(button)
            button.show()
        for role in self._shared_manipulation_roles:
            button = button_for(role)
            self._plot_manipulation_page.addWidget(button)
            button.show()
        groups.layout().invalidate()
        groups.updateGeometry()

    def _show_plot_workspace(self, key='plotting'):
        self.main_tabs.setCurrentWidget(self.tab_simple_plots)
        self.plot_control_tabs.setCurrentIndex(0 if key == 'plotting' else 1)

    def _show_isar_workspace(self):
        self._show_plot_workspace('isar')
