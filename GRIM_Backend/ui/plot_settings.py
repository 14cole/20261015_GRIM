"""The single settings layout for each canvas's processing and display controls."""
from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QTabWidget,
)

from GRIM_Backend.plotting.registry import PLOT_MODES


IMAGES = {"waterfall", "isar_image", "isar_composite", "az_vs_range", "range_freq", "delta_map"}
RANGE_IMAGES = {"isar_image", "isar_composite", "az_vs_range"}


class _SettingsTabs(QTabWidget):
    # A hidden advanced page should not force every other page to its height.
    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() + QSize(12, self.tabBar().sizeHint().height() + 12) if page else super().sizeHint()

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint() + QSize(12, self.tabBar().sizeHint().height() + 12) if page else super().minimumSizeHint()


class ContextPlotSettings(QWidget):
    def __init__(self, context):
        super().__init__()
        self.context = c = context
        self.mode = None
        self._artists = None
        self.groups = []
        self.conditions = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel()
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.tabs = _SettingsTabs()
        self.tabs.currentChanged.connect(lambda _: self.tabs.updateGeometry())
        layout.addWidget(self.tabs)
        self.pages = {}
        for title in ("Axes & color", "Plot", "Advanced", "Appearance"):
            page = QWidget()
            box = QVBoxLayout(page)
            box.setContentsMargins(8, 8, 8, 8)
            box.addStretch(1)
            self.tabs.addTab(page, title.replace("&", "&&"))
            self.pages[title] = page

        for axis in ("x", "y"):
            form = self.group("Axes & color", f"{axis.upper()} axis")
            self.conditional_row(form, "Minimum", getattr(c, f"spin_plot_{axis}min"),
                                 when=lambda axis=axis: axis != "x" or self.mode != "azimuth_polar")
            self.conditional_row(form, "Maximum", getattr(c, f"spin_plot_{axis}max"),
                                 when=lambda axis=axis: axis != "x" or self.mode != "azimuth_polar")
            spin = getattr(c, f"spin_plot_{axis}step")
            spin.setToolTip("Tick spacing. Set 0 for automatic ticks.")
            self.row(form, "Tick spacing (0 = auto)", spin)
        form = self.group("Axes & color", "Color scale", IMAGES)
        self.conditional_row(form, "Minimum", c.spin_plot_zmin, when=lambda: self.mode != "delta_map")
        self.conditional_row(form, "Maximum", c.spin_plot_zmax, when=lambda: self.mode != "delta_map")
        self.conditional_row(form, "Tick spacing (0 = auto)", c.spin_plot_zstep, when=lambda: self.mode != "delta_map")
        for spin in (c.spin_plot_zmin, c.spin_plot_zmax):
            spin.setToolTip("Set minimum below maximum for fixed color limits; otherwise limits are automatic.")
        self.row(form, "Colorbar", c.chk_colorbar, c.chk_colorbar_shared)
        self.row(form, "Colormap", c.combo_colormap, c.chk_colormap_invert)
        form = self.group("Axes & color", "Value scale", when=lambda: self.mode != "delta_map")
        self.row(form, "Display values", c.combo_plot_scale)
        form = self.group("Plot", "Polar orientation", {"azimuth_polar"})
        self.row(form, "Zero direction", c.combo_polar_zero)

        a = c.analysis_controls
        if a is not None:
            form = self.group("Plot", "Azimuth reduction", {"frequency", "elevation_sweep"})
            self.row(form, "Statistic", a.combo_sweep_aggregation)
            self.conditional_row(form, "Percentile", a.spin_sweep_percentile,
                                 when=lambda: a.sweep_aggregation()["mode"] == "percentile")
            self.conditional_row(form, "", a.chk_sweep_interpolate,
                                 when=lambda: a.sweep_aggregation()["mode"] == "percentile")
            a.combo_sweep_aggregation.currentIndexChanged.connect(self.refresh_conditions)
            form = self.group("Plot", "CDF population", {"cdf"})
            self.row(form, "Probability", a.combo_cdf)
            self.row(form, "Population & thresholds", a.btn_cdf_settings)
            form = self.group("Plot", "Range–frequency processing", {"range_freq"})
            self.row(form, "Sub-band width", a.spin_range_subband)
            self.row(form, "Window", a.combo_range_window)
            self.row(form, "Range unit", a.combo_range_unit)
        image = c.image_analysis_controls
        if image is not None:
            form = self.group("Plot", "Waterfall axes", {"waterfall"})
            self.row(form, "Horizontal axis", image.combo_x)
            self.row(form, "Vertical axis", image.combo_y)
            self.row(form, "Remaining axis", image.combo_aggregation)

        form = self.group("Plot", "Range image processing", RANGE_IMAGES)
        self.row(form, "Window", c.combo_isar_window)
        self.row(form, "Range unit", c.combo_isar_units)
        self.conditional_row(form, "Reconstruction", c.combo_isar_recon, when=lambda: self.mode in ("isar_image", "isar_composite"))
        self.row(form, "", c.chk_isar_freq_band)
        self.conditional_row(form, "Frequency minimum", c.spin_isar_freq_min, when=c.chk_isar_freq_band.isChecked)
        self.conditional_row(form, "Frequency maximum", c.spin_isar_freq_max, when=c.chk_isar_freq_band.isChecked)
        self.row(form, "Mirror downrange", c.chk_isar_flip_y)
        self.conditional_row(form, "Mirror cross-range", c.chk_isar_flip_x, when=lambda: self.mode in ("isar_image", "isar_composite"))
        c.chk_isar_freq_band.toggled.connect(self.refresh_conditions)
        form = self.group("Plot", "Aperture", RANGE_IMAGES)
        self.row(form, "", c.chk_isar_aperture)
        self.conditional_row(form, "Center (deg)", c.spin_isar_ap_center, when=c.chk_isar_aperture.isChecked)
        self.conditional_row(form, "Width (deg)", c.spin_isar_ap_width, when=c.chk_isar_aperture.isChecked)
        c.chk_isar_aperture.toggled.connect(self.refresh_conditions)
        form = self.group("Axes & color", "Image display", RANGE_IMAGES)
        self.row(form, "Dynamic range", c.spin_isar_peak_drop, c.btn_isar_peak_scale)
        self.conditional_row(form, "", c.chk_isar_square, when=lambda: self.mode in ("isar_image", "isar_composite"))
        c.btn_isar_apply.setText("Apply plot settings")
        c.btn_isar_apply.setToolTip("Recompute the current image using these processing settings. Color and axis limits update directly.")

        form = self.group("Advanced", "Azimuth interpolation", {"isar_image"})
        self.row(form, "", c.chk_isar_az_interp)
        for label, spin in (("Minimum (deg)", c.spin_isar_az_min), ("Maximum (deg)", c.spin_isar_az_max),
                            ("Step (deg)", c.spin_isar_az_step)):
            self.conditional_row(form, label, spin, when=c.chk_isar_az_interp.isChecked)
        c.chk_isar_az_interp.toggled.connect(self.refresh_conditions)
        form = self.group("Advanced", "Scene, composition & memory", {"isar_image", "isar_composite"})
        c.isar_advanced.layout().setRowWrapPolicy(QFormLayout.WrapLongRows)
        c.isar_advanced.scene.setText("Crop to scene bounds")
        c.isar_advanced.memory_allow.setText("Allow budget overrun")
        self.row(form, "", c.isar_advanced)
        c.isar_advanced.changed.connect(self.refresh_conditions)
        form = self.group("Appearance", "Plot appearance")
        self.row(form, "", c.chk_plot_grid_visible)
        self.row(form, "Background", c.btn_plot_bg)
        self.row(form, "Grid", c.btn_plot_grid)
        self.row(form, "Text", c.btn_plot_text)
        self.set_mode(None)
        c.plot_canvas.mpl_connect("draw_event", self.refresh_from_canvas)

    def group(self, page, title, modes=None, when=lambda: True):
        group = QGroupBox(title.replace("&", "&&"))
        self.pages[page].layout().insertWidget(self.pages[page].layout().count()-1, group)
        self.groups.append((group, modes, when, page))
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        return form

    def row(self, form, label, *widgets):
        field = QWidget()
        box = QHBoxLayout(field)
        box.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            box.addWidget(widget)
            widget.show()  # Reparenting an explicitly hidden backing control keeps it hidden.
        text = QLabel(label)
        form.addRow(text, field)
        return text, field

    def conditional_row(self, form, label, *widgets, when):
        self.conditions.append((*self.row(form, label, *widgets), when))

    def refresh_conditions(self, *_):
        advanced = self.context.isar_advanced
        for widget, visible in ((advanced.x_half, advanced.scene.isChecked()),
                                (advanced.y_half, advanced.scene.isChecked()),
                                (advanced.side, advanced.mode.currentData() != "coherent"),
                                (advanced.memory_limit, not advanced.memory_auto.isChecked())):
            advanced.layout().setRowVisible(widget, visible)
        for label, field, when in self.conditions:
            visible = bool(when())
            label.setVisible(visible)
            field.setVisible(visible)
        visible_pages = set()
        for group, modes, when, page in self.groups:
            visible = self.mode is not None and (modes is None or self.mode in modes) and when()
            group.setVisible(bool(visible))
            if visible:
                visible_pages.add(page)
        for index, page in enumerate(self.pages):
            self.tabs.setTabVisible(index, page in visible_pages)
        self.tabs.setVisible(bool(visible_pages))
        if hasattr(self, "apply_footer"):
            self.apply_footer.setVisible(self.mode in RANGE_IMAGES)
        self.tabs.updateGeometry()

    def set_mode(self, mode):
        self.mode = mode
        self.context.isar_advanced.set_plot_mode(mode)
        title = PLOT_MODES[mode].label if mode in PLOT_MODES else (mode or "").replace("_", " ").title()
        self.heading.setText(f"Settings for {title}" if mode else "Create a plot to see its settings.")
        self.refresh_conditions()

    def refresh_from_canvas(self, *_):
        c = self.context
        artists = tuple(artist for ax in (c.plot_axes or [c.plot_ax])
                        for artist in (*ax.lines, *ax.images, *ax.collections)
                        if not getattr(artist, "_grim_spatial_overlay", False))
        if artists != self._artists:
            self._artists = artists
            self.set_mode(c.last_plot_mode if artists else None)


def configure_plot_settings(context):
    popup = context.settings_frame
    # Replace the empty popup placeholder with the only settings layout.
    old = popup.scroll_area.takeWidget()
    old.deleteLater()
    popup.plot_controls = ContextPlotSettings(context)
    popup.content_widget = popup.plot_controls
    popup.scroll_area.setWidget(popup.content_widget)
    footer = popup.plot_controls.apply_footer = QWidget(popup)
    footer_layout = QHBoxLayout(footer)
    footer_layout.setContentsMargins(0, 0, 0, 0)
    note = QLabel("Processing changes require Apply.\nAxis and color limits update directly.")
    note.setWordWrap(True)
    footer_layout.addWidget(note, 1)
    footer_layout.addWidget(context.btn_isar_apply)
    context.btn_isar_apply.show()
    popup.layout().addWidget(footer)
    footer.hide()
    popup.plot_controls.tabs.currentChanged.connect(
        lambda _: popup.scroll_area.verticalScrollBar().setValue(0))
    popup.resize(620, 620)
    popup.protect_controls()
