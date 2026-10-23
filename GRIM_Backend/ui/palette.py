"""Shared GRIM application colors, independent of the main GUI and solvers."""

from __future__ import annotations

APPLICATION_PALETTES: dict[str, dict[str, object]] = {
    "Colorful": {
        "is_dark": True,
        "win_bg": "#111827",
        "panel_bg": "#19162f",
        "text": "#f8fafc",
        "head_bg": "#312e81",
        "border": "#6d28d9",
        "hover": "#0e7490",
        "checked_bg": "#7c3aed",
        "checked_border": "#22d3ee",
        "grid": "#475569",
        "muted": "#c4b5fd",
        "fg": "#f8fafc",
        "plot_line_freq": "#22d3ee",
        "plot_line_angle": "#f472b6",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#7c3aed", "#0891b2", "#db2777", "#d97706",
            "#4f46e5", "#0e7490", "#9333ea", "#0284c7",
        ),
    },
    "Dark": {
        "is_dark": True,
        "win_bg": "#0f172a",
        "panel_bg": "#0b1222",
        "text": "#dbeafe",
        "head_bg": "#172554",
        "border": "#1e3a8a",
        "hover": "#1d4ed8",
        "checked_bg": "#2563eb",
        "checked_border": "#3b82f6",
        "grid": "#475569",
        "muted": "#94a3b8",
        "fg": "#dbeafe",
        "plot_line_freq": "#38bdf8",
        "plot_line_angle": "#a78bfa",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#1e3a8a", "#1d4ed8", "#172554", "#2563eb",
            "#1e40af", "#3b82f6", "#334155", "#0284c7",
        ),
    },
    "Neutral Dark": {
        "is_dark": True,
        "win_bg": "#111827",
        "panel_bg": "#1f2937",
        "text": "#f3f4f6",
        "head_bg": "#273244",
        "border": "#374151",
        "hover": "#475569",
        "checked_bg": "#2563eb",
        "checked_border": "#60a5fa",
        "grid": "#374151",
        "muted": "#9ca3af",
        "fg": "#f3f4f6",
        "success": "#34d399",
        "warning": "#fbbf24",
        "danger": "#f87171",
        "plot_line_freq": "#38bdf8",
        "plot_line_angle": "#c4b5fd",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#2563eb", "#0e7490", "#7c3aed", "#b45309",
            "#047857", "#be185d", "#4338ca", "#475569",
        ),
    },
    "Windows Classic": {
        # Early-2000s Windows chrome. "ui_style": "classic" switches
        # GRIM_Backend/ui/classic_theme.py on; see that module to remove.
        "ui_style": "classic",
        "is_dark": False,
        "win_bg": "#d4d0c8",
        "panel_bg": "#d4d0c8",
        "field_bg": "#ffffff",
        "bevel_light": "#ffffff",
        "bevel_dark": "#404040",
        "checked_face": "#e8e6e0",
        "text": "#000000",
        "head_bg": "#d4d0c8",
        "border": "#808080",
        "hover": "#000080",
        "checked_bg": "#000080",
        "checked_border": "#000080",
        "grid": "#c0c0c0",
        "muted": "#808080",
        "fg": "#000000",
        "success": "#008000",
        "warning": "#808000",
        "danger": "#800000",
        "plot_line_freq": "#000080",
        "plot_line_angle": "#800000",
        "plot_worst": "#008080",
        "layer_colors": (
            "#000080", "#008080", "#800000", "#808000",
            "#800080", "#008000", "#000000", "#808080",
        ),
    },
}
DEFAULT_APPLICATION_PALETTE = "Dark"
APPLICATION_PALETTE_DESCRIPTIONS = {
    "Colorful": "Purple, cyan, and magenta dark application chrome",
    "Dark": "GRIM blue/slate dark application chrome",
    "Neutral Dark": "Neutral slate surfaces, subtle borders, and blue actions",
    "Windows Classic": "Early-2000s Windows gray chrome with bevels and navy selection",
}
APPLICATION_PALETTE_SETTINGS_KEY = "appearance/application_palette"
LEGACY_APPLICATION_PALETTE_NAMES = {
    "Light": "Dark",
    "Raytheon": "Dark",
    "Raytheon-inspired": "Dark",
}
# Compatibility export for extensions/tests that used GRIM's former one fixed
# palette. It remains the exact default Dark palette.
BLUE_PALETTE = APPLICATION_PALETTES[DEFAULT_APPLICATION_PALETTE]


def normalize_application_palette_name(value: object) -> str:
    """Return a current palette name, including legacy-setting migration."""

    normalized = str(value).strip()
    normalized = LEGACY_APPLICATION_PALETTE_NAMES.get(
        normalized,
        normalized,
    )
    if normalized not in APPLICATION_PALETTES:
        return DEFAULT_APPLICATION_PALETTE
    return normalized
