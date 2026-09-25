"""Colors and the shared plotly layout.

Figures are built in the light palette. Each light color has a dark
counterpart in DARK_MAP, and app.js swaps them when the page is in dark
mode, so one figure definition serves both themes. The series colors are
the same four slots the matplotlib figures in figures/ use.
"""

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLIN = "#a8a69f"  # clinical features, next to blue for PK/PD
NEUTRAL = "#cfcdc6"  # midpoint of the blue-orange diverging scale
SHADE = "rgba(137,135,129,0.14)"  # outcome window and similar bands

# sequential blue ramp for heatmaps, lightest first
RAMP = ["#f4f3f0", "#cde2fb", "#86b6ef", "#256abf", "#104281"]

ALGOS = ["enet", "rf", "xgb", "cat"]
ALGO_LABELS = {"enet": "Elastic net", "rf": "Random forest", "xgb": "XGBoost", "cat": "CatBoost"}
ALGO_COLORS = {"enet": BLUE, "rf": ORANGE, "xgb": AQUA, "cat": YELLOW}
ALGO_SYMBOLS = {"enet": "circle", "rf": "square", "xgb": "diamond", "cat": "triangle-up"}

DARK_MAP = {
    SURFACE: "#1a1a19",
    PAGE: "#141413",
    INK: "#ffffff",
    INK2: "#c3c2b7",
    MUTED: "#95948b",
    GRID: "#2f2f2d",
    AXIS: "#4a4a46",
    BLUE: "#3987e5",
    ORANGE: "#d95926",
    AQUA: "#199e70",
    YELLOW: "#c98500",
    CLIN: "#6e6d67",
    NEUTRAL: "#5c5b55",
    SHADE: "rgba(149,148,139,0.16)",
    RAMP[0]: "#232322",
    RAMP[1]: "#1c5cab",
    RAMP[2]: "#3987e5",
    RAMP[3]: "#86b6ef",
    RAMP[4]: "#cde2fb",
}

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


def axis(title="", **kw):
    a = {
        "title": {"text": title, "font": {"size": 11.5, "color": INK2}, "standoff": 8},
        "gridcolor": GRID,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS,
        "ticks": "",
        "tickfont": {"size": 11, "color": INK2},
        "automargin": True,
    }
    a.update(kw)
    return a


def layout(xtitle="", ytitle="", legend=True, **kw):
    lay = {
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT, "size": 11, "color": INK},
        "xaxis": axis(xtitle),
        "yaxis": axis(ytitle),
        "margin": {"l": 8, "r": 8, "t": 30 if legend else 8, "b": 8},
        "hoverlabel": {"bgcolor": SURFACE, "bordercolor": GRID, "align": "left", "font": {"family": FONT, "size": 12, "color": INK}},
        "hovermode": "closest",
        "showlegend": legend,
        "legend": {
            "orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11.5, "color": INK2},
            "itemclick": "toggle", "itemdoubleclick": "toggleothers",
        },
        "dragmode": False,
    }
    lay.update(kw)
    return lay
