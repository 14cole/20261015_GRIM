"""Display-only image edits without dataset reads or repeated transforms."""
from __future__ import annotations

import numpy as np

from .modes import common


def _style_colorbar(owner, colorbar, label):
    owner._apply_colorbar_ticks(colorbar)
    colorbar.set_label(label, color=owner._current_plot_text())
    colorbar.ax.tick_params(colors=owner._current_plot_text())


def _set_limits(owner, meshes, *, shared, mode):
    if mode == "delta_map":
        return  # Delta Map keeps its dedicated symmetric difference scale.
    low, high = owner.spin_plot_zmin.value(), owner.spin_plot_zmax.value()
    limits = common.finite_data_limits(mesh.get_array() for mesh in meshes) if shared and low >= high else None
    for mesh in meshes:
        pair = (low, high) if low < high else (limits or common.finite_data_limits([mesh.get_array()]))
        if pair is None:
            continue
        if mode == "range_freq" and low >= high and not owner._plot_scale_is_linear():
            pair = (max(pair[0], pair[1] - 60.), pair[1])
        mesh.set_clim(*pair)


def update_image_style(owner):
    """Return true when the current image accepted a pure display update."""
    mode = getattr(owner, "last_plot_mode", None)
    if mode == "waterfall":
        return update_waterfall_display(owner)
    if mode not in ("isar_image", "isar_composite", "az_vs_range", "range_freq", "delta_map"):
        return False
    axes = list(getattr(owner, "plot_axes", None) or [owner.plot_ax])
    meshes = [artist for ax in axes for artist in [*ax.images, *ax.collections]
              if hasattr(artist, "get_array") and artist.get_array() is not None]
    if not meshes:
        return False
    figure = owner.plot_figure
    state = getattr(figure, "_grim_image_display_state", None)
    if state is None or state["mode"] != mode or state["meshes"] != meshes:
        linear = owner._plot_scale_is_linear()
        label = {"isar_composite": "Image Intensity (linear)" if linear else "Image Intensity (dB)",
                 "isar_image": "Image Intensity (linear)" if linear else "Image Intensity (dB)",
                 "az_vs_range": "Range image intensity (linear, a.u.)" if linear else "Range image intensity (dB re 1 a.u.)",
                 "delta_map": "A - B (dB)"}.get(mode, "")
        if mode == "range_freq":
            label = getattr(figure, "_grim_range_colorbar_label", "Range profile")
        extends = "neither"
        if mode == "delta_map":
            limits = meshes[-1].get_clim()
            values = np.ma.asarray(meshes[-1].get_array())
            extends = "both" if np.any((values < limits[0]) | (values > limits[1])) else "neither"
        state = dict(mode=mode, meshes=meshes, axes=axes, label=label, extend=extends,
                     # A single-axis colorbar wraps its original subplot in
                     # a smaller nested grid. Retain the original data slot,
                     # otherwise hiding the initial bar keeps its empty gutter.
                     specs=[ax.get_subplotspec().get_topmost_subplotspec()
                            if ax.get_subplotspec() is not None else None for ax in axes], bars=None)
        figure._grim_image_display_state = state
    shared = bool(owner.chk_colorbar_shared.isChecked()) if mode != "range_freq" else True
    shown = bool(owner.chk_colorbar.isChecked())
    _set_limits(owner, meshes, shared=shared, mode=mode)
    target = (shown, shared)
    current = (bool(owner.plot_colorbars), len(owner.plot_colorbars) <= 1)
    if (state["bars"] is None and current != target) or (state["bars"] is not None and state["bars"] != target):
        owner._remove_colorbar()
        for ax, spec in zip(axes, state["specs"]):
            if spec is not None:
                ax.set_subplotspec(spec)
        if shown:
            owner.plot_colorbars = ([figure.colorbar(meshes[-1], ax=axes, extend=state["extend"])] if shared else
                [figure.colorbar(mesh, ax=mesh.axes, extend=state["extend"]) for mesh in meshes])
    state["bars"] = target
    for colorbar in owner.plot_colorbars:
        _style_colorbar(owner, colorbar, state["label"])
    owner._update_current_python_plot_style()
    owner.plot_canvas.draw_idle()
    return True


def update_waterfall_display(owner):
    """Relayout retained image artists and exact native cut records in place."""
    figure = owner.plot_figure
    state = getattr(figure, "_grim_waterfall_display", None)
    if state is None or any(ax not in figure.axes for ax in state["axes"]):
        return False
    controls = getattr(owner, "image_analysis_controls", None)
    if controls is None:
        return False
    if (tuple(controls.axes()) != state["physical_axes"]
            or controls.aggregation() != state["aggregation"]):
        return False
    horizontal = bool(getattr(controls, "horizontal_cuts", controls.linked_cuts)())
    vertical = bool(getattr(controls, "vertical_cuts", controls.linked_cuts)())
    shown, shared = bool(owner.chk_colorbar.isChecked()), bool(owner.chk_colorbar_shared.isChecked())
    target = (horizontal, vertical, shown, shared)
    _set_limits(owner, state["meshes"], shared=shared, mode="waterfall")
    if target != state["layout"]:
        controller = getattr(figure, "_grim_linked_cuts", None)
        position = getattr(controller, "position", None)
        old_cut_axes = [cut for _ax, cuts, _record in controller.panels for cut in cuts if cut is not None] if controller else []
        if controller is not None:
            for artists in controller.artists:
                for artist in artists[2:]:
                    if artist is not None:
                        artist.remove()
            controller.disconnect()
        owner._remove_colorbar()
        for ax in old_cut_axes:
            if ax in figure.axes:
                figure.delaxes(ax)
        axes, meshes, records = state["axes"], state["meshes"], state["records"]
        columns = min(3, int(np.ceil(np.sqrt(len(axes)))))
        rows = int(np.ceil(len(axes) / columns))
        outer = figure.add_gridspec(rows, columns + int(shown and shared),
            width_ratios=[1.] * columns + ([.04] if shown and shared else []))
        cut_axes, colorbar_axes = [], []
        for index, (ax, record) in enumerate(zip(axes, records)):
            widths = [4.] + ([1.4] if vertical else []) + ([.16] if shown and not shared else [])
            grid = outer[index // columns, index % columns].subgridspec(
                1 + int(horizontal), len(widths), width_ratios=widths,
                height_ratios=[3., 1.] if horizontal else [1.])
            ax.set_subplotspec(grid[0, 0])
            h = figure.add_subplot(grid[1, 0], sharex=ax) if horizontal else None
            v = figure.add_subplot(grid[0, 1], sharey=ax) if vertical else None
            cut_axes.append((h, v))
            for cut in (h, v):
                if cut is not None:
                    owner._style_axes(cut)
                    cut.tick_params(labelsize=7)
            compact = horizontal or vertical or len(axes) > 1
            ax.set_title((f"{index + 1} · " + state["short_titles"][index]) if compact else state["titles"][index],
                color=owner._current_plot_text(), fontsize=9 if compact else 11)
            ax.set_xlabel("" if horizontal else state["x_label"], fontsize=8 if compact else 10, labelpad=1)
            ax.set_ylabel(state["y_label"], fontsize=8 if compact else 10, labelpad=1)
            ax.tick_params(labelsize=7 if compact else 10, labelbottom=not horizontal)
            if h is not None:
                h.set_xlabel(state["x_label"], fontsize=8, labelpad=1)
            if v is not None:
                v.tick_params(labelleft=False)
            if shown and not shared:
                colorbar_axes.append(figure.add_subplot(grid[0, -1]))
        if shown:
            if shared:
                colorbar_axes.append(figure.add_subplot(outer[:, -1]))
                owner.plot_colorbars = [figure.colorbar(meshes[-1], cax=colorbar_axes[0])]
            else:
                owner.plot_colorbars = [figure.colorbar(mesh, cax=cax) for mesh, cax in zip(meshes, colorbar_axes)]
        from GRIM_Backend.ui.image_analysis import LinkedImageCuts
        controller = LinkedImageCuts(owner.plot_canvas, list(zip(axes, cut_axes, records)), can_select=state["can_select"])
        figure._grim_linked_cuts = controller
        context = getattr(owner, "_plot_contexts", {}).get(getattr(owner, "_active_plot_tab", None))
        if context is not None:
            context.session.own(controller)
        if callable(getattr(controls, "bind", None)):
            controls.bind(controller)
        if position is not None:
            controller.update(*position, notify=False)
        state["layout"] = target
    for index, colorbar in enumerate(owner.plot_colorbars):
        label = state["shared_label"] if shared else state["labels"][index]
        _style_colorbar(owner, colorbar, label)
    owner._update_current_python_plot_style()
    owner.plot_canvas.draw_idle()
    return True
