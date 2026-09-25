/* Draws the plotly figures embedded in the page, swaps variants when a
   dropdown changes, recolors everything for dark mode and makes the
   tables sortable. If plotly.js did not load, each chart falls back to the
   static PNG drawn by matplotlib. */
(function () {
  "use strict";

  const FIGS = JSON.parse(document.getElementById("figures").textContent);
  const DARK = {};
  const darkSrc = JSON.parse(document.getElementById("dark-map").textContent);
  Object.keys(darkSrc).forEach((k) => { DARK[k.toLowerCase()] = darkSrc[k]; });

  const root = document.documentElement;
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : media.matches);

  // One pass over the figure, replacing every light color with its dark twin.
  function recolor(value) {
    if (typeof value === "string") {
      const d = DARK[value.toLowerCase()];
      return d === undefined ? value : d;
    }
    if (Array.isArray(value)) return value.map(recolor);
    if (value && typeof value === "object") {
      const out = {};
      for (const k in value) out[k] = recolor(value[k]);
      return out;
    }
    return value;
  }

  const config = (name) => ({
    displaylogo: false,
    responsive: true,
    scrollZoom: false,
    displayModeBar: "hover",
    modeBarButtonsToRemove: ["lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "toggleSpikelines"],
    toImageButtonOptions: { format: "png", filename: name, scale: 2 }
  });

  function variantKey(id) {
    const sels = document.querySelectorAll(`select[data-for="${id}"]`);
    if (!sels.length) return Object.keys(FIGS[id])[0];
    return Array.from(sels, (s) => s.value).join("|");
  }

  function draw(node) {
    const id = node.dataset.fig;
    const key = variantKey(id);
    let fig = FIGS[id][key];
    if (!fig) return;
    if (isDark()) fig = recolor(fig);
    const layout = Object.assign({}, fig.layout, { autosize: true });
    const fn = node.dataset.plotted ? window.Plotly.react : window.Plotly.newPlot;
    node.dataset.plotted = "1";
    fn(node, fig.data, layout, config(`${id}_${key.replace(/\W+/g, "_")}`));
  }

  const plots = Array.from(document.querySelectorAll("[data-fig]"));

  function fallback() {
    plots.forEach((node) => {
      const png = node.dataset.png;
      node.classList.remove("plot");
      node.innerHTML = png
        ? `<img src="figures/${png}" alt="${node.getAttribute("aria-label") || ""}">`
        : '<p class="fallback">This chart needs plotly.js, which did not load. The table next to it has the numbers.</p>';
    });
  }

  function drawAll() { plots.forEach(draw); }

  if (typeof window.Plotly === "undefined") {
    fallback();
  } else {
    drawAll();
    document.querySelectorAll("select[data-for]").forEach((sel) => {
      sel.addEventListener("change", () => draw(document.getElementById(sel.dataset.for)));
    });
    media.addEventListener("change", () => { if (!root.dataset.theme) drawAll(); });
  }

  // theme toggle; the page follows the system setting until it is used
  const toggle = document.getElementById("theme");
  function paintToggle() {
    const dark = isDark();
    toggle.textContent = dark ? "Light mode" : "Dark mode";
    toggle.setAttribute("aria-pressed", String(dark));
  }
  toggle.addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    paintToggle();
    if (typeof window.Plotly !== "undefined") drawAll();
  });
  media.addEventListener("change", paintToggle);
  paintToggle();

  // sortable tables: click a header to sort, again to reverse
  document.querySelectorAll("table.sortable").forEach((table) => {
    const heads = Array.from(table.querySelectorAll("th"));
    heads.forEach((th, col) => {
      th.querySelector("button").addEventListener("click", () => {
        const asc = th.getAttribute("aria-sort") !== "ascending";
        heads.forEach((h) => h.removeAttribute("aria-sort"));
        th.setAttribute("aria-sort", asc ? "ascending" : "descending");
        const body = table.tBodies[0];
        const rows = Array.from(body.rows);
        const val = (row) => {
          const cell = row.cells[col];
          const raw = cell.dataset.sort !== undefined ? cell.dataset.sort : cell.textContent.trim();
          const num = parseFloat(raw);
          return isNaN(num) || !/^[-+]?[\d.]/.test(raw) ? raw.toLowerCase() : num;
        };
        rows.sort((a, b) => {
          const x = val(a), y = val(b);
          const cmp = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
          return asc ? cmp : -cmp;
        });
        rows.forEach((r) => body.appendChild(r));
      });
    });
  });
})();
