"""Searchable, navigation-only FREDDY help, shared by standalone and GRIM."""
from __future__ import annotations

from html import escape
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QListWidget,
                              QListWidgetItem, QSplitter, QTextBrowser, QVBoxLayout, QWidget)


MODE_TOPICS = {'Impedance': 'impedance', 'IBC Batch': 'batch', 'Off Angle': 'angle',
               'Thickness': 'thickness', 'Inverse Design': 'inverse',
               'Material Mix': 'mix', 'Material Explorer': 'explorer'}


def workflow(mode, steps, expected):
    destination = mode + (' Setup' if mode not in ('Material Mix', 'Material Explorer') else '')
    return (f'<p><a href="mode:{escape(mode)}">Open {escape(destination)} →</a></p>'
            '<h3>How to use it</h3><ol>' + ''.join(f'<li>{step}</li>' for step in steps)
            + '</ol><h3>Expected results and decisions</h3>' + expected)


def catalog(rows):
    return ('<table cellspacing="0" cellpadding="7" width="100%">'
            '<tr><th align="left">Plot / view</th><th align="left">What it shows and how to read it</th></tr>'
            + ''.join(f'<tr><td valign="top"><b>{name}</b></td><td>{description}</td></tr>'
                      for name, description in rows) + '</table>')


ANALYSIS_PLOTS = [
    ('Resistance', 'Real front-face input impedance versus frequency in ohms. IBC Batch overlays selected thicknesses. At broadside, a PEC absorber approaching 377 Ω with near-zero reactance is approaching a free-space match; resistance alone is insufficient.'),
    ('Reactance', 'Imaginary input impedance versus frequency in ohms. Zero crossings can locate a resonance or match, but verify the resistance and reflection at the same frequency.'),
    ('Frequency curves', 'Selected metric versus frequency for chosen thicknesses or angles. The selected curve is emphasized; check Plot in the comparison table to overlay others. Reflection curves include the target and shade passing bands for the selected choice.'),
    ('Heatmap', 'Frequency on the vertical axis; thickness or incidence angle on the horizontal axis; color is the chosen metric. A labeled contour marks the reflection target when it crosses the grid. Hover to read the sampled value (z) under the cursor. Click the map to select a column and nearest frequency; dashed lines mark them and the toolbar reports that Selected (x, y, z) sample. Read the colorbar before interpreting colors.'),
    ('TE/TM comparison', 'Off Angle: two reflection maps use one shared color scale. Choose Polarization: Both in Setup before running. Both responses export as separate _TE.csv and _TM.csv files. Isotropic stacks agree at 0° and generally separate at oblique incidence. This view always compares reflection; use individual plots for other metrics.'),
    ('Bandwidth', 'Widest contiguous frequency interval meeting the reflection target, versus thickness or angle. Nominal and worst analyzed reflection are shown together when bounds exist. This favors broad bands rather than deep, narrow nulls. Crossings are linearly interpolated in dB within the chosen band.'),
    ('Band coverage', 'Percentage of the selected frequency interval meeting the target, including separate passing intervals. It can be high even when no single wide band passes. A single-frequency comparison reports passing-point coverage and has no bandwidth.'),
    ('Tolerance envelope', 'Nominal reflection and pointwise lower/upper reflection bounds versus frequency for the selected thickness or angle. The upper reflection bound is the least-negative, worst analyzed case. The case giving the bound may change with frequency. This is not a probability distribution or manufacturing yield.'),
    ('Frequency of minimum', 'Frequency of the lowest sampled reflection versus thickness, including IBC Batch. The minimum is searched across the full completed frequency sweep, not just the comparison band. Use it to track tuning; it does not measure bandwidth or tolerance robustness.'),
    ('At selected frequency', 'The selected metric versus angle or thickness at the nearest computed frequency to Slice GHz. The plot title states the actual sample used; no extra solver run or frequency interpolation is performed.'),
    ('Power balance', 'Nominal reflected, absorbed, and transmitted fractions of incident power in percent. For a passive stack their sum should be about 100%. PEC backing has zero transmission. Air backing can transmit strongly even when reflection is small.'),
    ('Coating error vs angle', 'After Check GHOST coating approximation: maximum over sampled frequencies of |Γ scalar IBC − Γ planar stack| versus angle, separately for TE/TM. This dimensionless complex-amplitude difference includes phase; it is neither reflection dB nor dBsm error.'),
    ('Coating error map', 'The same absolute complex-reflection difference resolved by frequency and angle, with separate TE/TM panels. Locate where a broadside scalar IBC departs from the oblique planar stack. It does not certify finite-body GHOST RCS accuracy.'),
]

INVERSE_PLOTS = [
    ('Reflection curves', 'Retained candidates versus frequency. Worst analyzed case is the highest reflection across analyzed angles and tolerance cases at each frequency. Point percentile describes those analyzed points only; a lower percentile is more optimistic, not a yield claim.'),
    ('Depth vs bandwidth', 'Compare the depth of candidate reflection minima with their widest passing band. Use this tradeoff to spot narrow, deep nulls versus broader useful response. Select a point to inspect its candidate; the display target does not rerank the search objective.'),
    ('Analysis history', 'Scores in grid evaluation order and the running best score. This shows progress through the finite combination grid, not a physical response curve or convergence guarantee. A stopped run has only evaluated part of the grid.'),
    ('Selected candidate angle map', 'Reflection versus frequency and analyzed angle for the selected candidate, using its captured case data. Read the bound in the title. This isolates where an otherwise good average candidate fails at an angle. Clear Auto under Color scale to compare candidates on fixed limits.'),
    ('Selected candidate tolerance', 'Nominal response and analyzed tolerance envelope at the chosen captured angle. Compare the upper reflection bound with the target before accepting a candidate. Unanalyzed angles and tolerances are not inferred.'),
]

MIX_PLOTS = [
    ('Synthesized ε/μ', 'Effective relative permittivity and permeability versus frequency for the recipe or selected candidate. Solid/dashed curves distinguish real and imaginary parts; dotted target curves appear for property matching. Passive imaginary parts are negative under FREDDY’s convention.'),
    ('Effective loss tangent', '−Im(ε)/Re(ε) and −Im(μ)/Re(μ) versus frequency. Near-zero real parts can make a ratio unstable or undefined; examine the original ε/μ curves. A larger loss tangent does not by itself mean better absorber performance.'),
    ('Property match error', 'Relative property mismatch (%) at target frequencies. Lower is a closer property match under the chosen weights; it does not establish a stack reflection requirement. Zero means an exact match under that error definition.'),
    ('Model sensitivity', 'Effective-property differences among available mixing rules at the indicated comparison frequency. Large separation means the morphology/model choice matters; it is not a statistically calibrated uncertainty interval.'),
    ('Performance heatmap', 'For a recipe plus thickness performance target: the chosen metric versus frequency and angle, with a requirement contour. Reflection/transmission upper limits pass below the target; absorption lower limits pass above it. Read the metric’s dB or percent unit. Clear Auto color scale and enter Min/Max (press Enter to apply) to compare blends on one scale.'),
    ('Worst angle at each frequency', 'Performance at the least favorable analyzed angle for each frequency. Upper-limit objectives take the maximum; lower-limit absorption objectives take the minimum. Inspect this curve for band failures.'),
    ('Worst frequency at each angle', 'Performance at the least favorable analyzed frequency for each angle. Use it to see angular coverage of the requested requirement. The worst tolerance corner controls pass/fail even when average corners are used for ranking.'),
    ('Material Explorer curves', 'Four panels show ε real, ε imaginary, μ real, and μ imaginary on each file’s native frequency grid. Select sources to compare them without changing the layer stack. The comparison table interpolates at one common frequency and marks out-of-range sources.'),
]


def guide_topics(physics_html):
    """Ordered content, kept independent of editable project/run state."""
    return [
        ('overview', 'Start here · choose a workflow', '''
<p><b>FREDDY — Frequency-Dependent Reflection and EM Dielectric Dimensional Yield</b></p>
<p>Use FREDDY to design and examine an <b>infinite planar material stack</b>:
reflection, transmission, absorption, and input impedance. It does not predict finite-object RCS or dBsm.</p>
<p>Search for a task, plot name, or term, then choose a topic. Press <b>F1</b> from a mode for its workflow,
or use <b>Explain view</b> in analysis results for the plot catalog. Workflow links open Setup and preserve your inputs.</p>
<h3>Choose your starting point</h3>
<ul><li>Check measured ε/μ files and overlap: <a href="topic:explorer">Material Explorer</a>.</li>
<li>Evaluate a known coating and produce one nominal IBC: <a href="topic:impedance">Impedance</a>.</li>
<li>Tune one layer: <a href="topic:thickness">Thickness</a>; export several candidates: <a href="topic:batch">IBC Batch</a>.</li>
<li>Check angular and polarization robustness: <a href="topic:angle">Off Angle</a>.</li>
<li>Search layer thicknesses and sheet resistance for the chosen materials: <a href="topic:inverse">Inverse Design</a>.</li>
<li>Predict a blend or search ingredient fractions: <a href="topic:mix">Material Mix</a>.</li></ul>
<h3>A useful design loop</h3>
<p>Validate material coverage → build incident-side-to-backing stack → compute a nominal response →
tune thickness or run inverse design → check angles and tolerance bounds → export the chosen nominal material/IBC →
validate its use in GHOST. Save a project to preserve the configuration and material references.</p>
<h3>Understand the target first</h3>
<p>For example, reflection ≤ −10 dB means at most 10% reflected power. With PEC backing this implies at least 90%
absorption. With air backing it can instead mean transmission. Absorbed power in dB approaches <b>0 dB</b> as absorption
approaches 100%; more-negative absorption dB is less absorption.</p>
<p>Results retain the last successful run and its original context. Setup edits require another run to affect them.
Opening a project clears results. Save plot exports the displayed figure; solver CSVs retain full computed samples.</p>'''),
        ('impedance', 'Workflow · Impedance and GHOST', workflow('Impedance', [
            'Choose Add Layer and select Measured material CSV or Constant εr / μr (all frequencies). For a constant layer enter relative εr real/imaginary and μr real/imaginary, plus thickness; no CSV is needed. Add Sheet creates a zero-thickness resistive sheet. Arrange layers from the illuminated face toward the backing.',
            'Choose a frequency sweep inside every material’s coverage, PEC or air backing, an output CSV, and optional systematic tolerances. Compute the broadside response.',
            'In Results inspect Resistance, Reactance, Frequency curves, and Power balance. Set a comparison target and band. Use Run details to verify the captured setup.',
            'For a GHOST coating, choose PEC backing and run Check GHOST coating approximation over relevant angles. Review both coating-error plots.',
            'Export the nominal IBC. In integrated GRIM, attach the verified nominal export to a saved/loaded GHOST geometry, then Save Geometry to persist it. Use the physics topic’s polarization mapping when comparing.'
        ], '<p>A useful PEC absorber has low reflection over the required band, adequate tolerance margin, and impedance close to the incident wave impedance. Broadside air-backed impedance is not generally a one-sided closed-body IBC. Off-angle and tolerance reports are analysis CSVs; select the nominal three-column file for GHOST.</p>')),
        ('thickness', 'Workflow · individual and combined thickness sweeps', workflow('Thickness', [
            'Build the stack. In Thickness Setup, check Vary for each bulk layer to sweep and enter its minimum, maximum and step in inches. These ranges are shared with Inverse Design. Unchecked layers stay nominal.',
            'Choose frequency and incidence-angle start/stop/step, TE/TM or Both, and a reflection target. Every minimum + k × step value up to maximum is included; an off-step maximum is excluded.',
            'Analyze every combination. Results show each layer varied individually with the others nominal, followed by all layer ranges varied independently. Three layers with 3, 4 and 5 values produce 60 joint stacks. There is no sampling count or random seed.',
            'Thickness envelopes show pointwise minimum and maximum PEC reflection. Select one angle to isolate thickness effects; All selected angles pools their spread. Click an envelope point to see the actual stacks producing its bounds.',
            'Whole-band best/worst stacks shows realizable stacks ranked by their highest reflection over every selected frequency, angle and polarization. Layer sensitivity ranks the largest individual-layer margin loss relative to nominal. Passing grid cases are counts, not manufacturing yield.',
            'Layer details retains the individual-layer heatmap, frequency curves, phase, power balance, bandwidth and slice tools at the chosen captured angle. Other layers remain nominal in this view.',
            'Export envelope CSV writes separate _TE.csv and _TM.csv files when both polarizations were analyzed, including every panel, angle, frequency and the thickness combinations attaining each bound. Export study JSON includes captured settings, layers, detailed metrics, bounds, witnesses and best/worst stacks. Stop keeps the previous completed result.'
        ], '<p>Material properties and sheet resistances remain nominal. Nominal is a separate reference and can lie outside the grid bounds. Different envelope points may come from different stacks; the envelope need not be realizable by one stack. Whole-band best/worst uses a fixed stack across the full operating grid. Exhaustive means every defined discrete combination, not certified continuous extrema. Reduce thickness/frequency/angle steps to check resolution. Preflight rejects excessive work rather than silently skipping combinations. Projects store inputs and clear results when loaded.</p>')),
        ('angle', 'Workflow · Off Angle (TE/TM)', workflow('Off Angle', [
            'Start with a promising stack. Set frequency range and incidence angles from the surface normal (0° broadside, less than 90°).',
            'Choose TE, TM or Both, then Compute. Both exports separate _TE.csv and _TM.csv files. Unsupported oblique TM for directional materials is rejected; select TE for that model.',
            'Compare TE/TM maps on their common scale. Click an angle to inspect its curve. Use the upper analyzed reflection bound when screening tolerance cases.',
            'Set the required band and target. Read the passing sampled angular ranges in the note or export comparison CSV, then refine angle/frequency sampling near a pass/fail boundary.'
        ], '<p>TE/TM should coincide at broadside for an isotropic stack. Oblique response can split, shift, or degrade. A grouped passing angular range only describes the sampled angles; it does not prove every intermediate angle passes.</p>')),
        ('batch', 'Workflow · export several IBCs', workflow('IBC Batch', [
            'Use Impedance to set the shared frequency sweep and build the stack. Select one bulk layer in IBC Batch.',
            'Set its thickness start/stop/step and unit (in or mm), an output folder, and filename settings. Review the preview count and endpoints.',
            'Choose Export IBC batch. Existing outputs are listed for replacement confirmation; successful publication produces a nominal broadside PEC-backed IBC per thickness.',
            'Compare reflection, bandwidth, resistance, or reactance in Results. Select a row or thickness, then Use selected IBC for GHOST for the file you intend to attach.'
        ], '<p>The files differ by one layer’s nominal thickness. They are not angle-specific IBCs or tolerance envelopes. Selecting a batch for export does not choose one output automatically for GHOST. A file changed after export must be re-exported before the verified handoff.</p>')),
        ('inverse', 'Workflow · Inverse Design', workflow('Inverse Design', [
            'Add/edit layers from measured CSVs or constant εr/μr. Define fixed values or minimum/maximum/step for variable bulk thickness and sheet resistance. Material values stay fixed during the search; keep other parameters fixed to control the combination count.',
            'Choose a band sweep or discrete frequency targets, incidence angles, and TE, TM or Both. Both ranks against both polarizations together; comparison curves show the worse polarization at each case/angle. Review the exact Cartesian-grid combination count. A maximum off the configured step is not included.',
            'For a peak requirement choose Whole-band PEC reflection requirement (worst point), then enter Reflection Target (dB), for example −10. The score is the worst reflection across all analyzed frequencies, angles, and tolerance cases minus that limit; lower is better and a gap ≤ 0 passes. The existing worst-corner and average-corner mean objectives remain available.',
            'Choose how many best candidates to retain. Optional .fsearch checkpoints save scores and run identity; also save the project and source materials. Constant material values live in the project. Changing the search objective, requirement, or ε/μ invalidates reuse of prior scores.',
            'Analyze all combinations. Stop and keep best retains finished scores; Resume remaining continues matching inputs. Reducing Keep best saves result storage but does not reduce evaluations.',
            'In Results compare Reflection curves and Depth vs bandwidth at the display target. Inspect Selected candidate angle map and Selected candidate tolerance, then Apply Selected or save the candidate stack.',
            'For per-layer robustness, use Edit per-layer tolerance bounds in Inverse Design and disable shared uncertainty. Choose Per-layer tolerances: worst sampled reflection gap or modeled failure fraction (%). Worst-gap probes the rectangular support independently of distributions/groups; yield uses the entered distribution/group model. Every design uses the same trial deviations. Zero sampled failures is not a guarantee.',
            'Use Verify candidates in Results and set verification samples, seed, yield repeats and refinements in that dialog. Only applicable controls are shown. Each refinement bisects frequency/angle intervals and doubles joint interior tolerance samples. Discrete frequency targets remain discrete. Verification uses fresh seeds, reports worst conditions and refinement changes, and estimates yield with independent repeats on the finest grid. Sort retained candidates by verified margin or yield and export the captured JSON report.',
            'Recompute the chosen stack in Impedance/Off Angle as needed and export its nominal IBC when satisfied.'
        ], '<p>For example, a design with reflection −40 and −1 dB has a better mean than a flat −11 dB design, but misses a −10 dB whole-band requirement by 9 dB. The flat design passes with 1 dB margin and wins under the whole-band objective. Results shows the captured requirement, Gap, and Req. margin; margin = −gap and nonnegative passes. Changing the Results target or percentile only changes plotted bandwidth/coverage, not rank, captured pass/fail, or margin. Discrete targets require every requested point to pass; no bandwidth between them is implied. A completed grid exhausts configured combinations, not a continuous design space. Verify between samples with a finer sweep. Point percentiles are not manufacturing yield.</p>')),
        ('tolerance', 'Workflow · tolerance analysis', workflow('Thickness', [
            'Use Thickness for exhaustive individual and combined thickness ranges. Sensitivity is a result of this study, not a separate sampling mode.',
            'For distribution-based inverse objectives, use Inverse Design → Edit per-layer tolerance bounds. Those manufacturing assumptions remain separate from the exhaustive thickness grid.',
            'Inverse manufacturing bounds support thickness, sheet resistance and signed material components, using percent of nominal magnitude or absolute units. Distribution and shared-group settings affect modeled yield, not exhaustive thickness results.'
        ], '<p>Manufacturing yield is conditional on assumed distributions and correlations; counting equally weighted grid cases is not measured yield. The former standalone Sensitivity only and Sensitivity + statistical trials modes have been consolidated into Thickness. Old projects retain their layer manufacturing metadata for inverse verification; dormant sample counts and seeds do not control thickness runs.</p>')),
        ('mix', 'Workflow · Material Mix', workflow('Material Mix', [
            'Add measured ingredient files and check their common frequency coverage. Choose a mixing rule consistent with the expected morphology; inspect its advisory. Enter volume parts/fraction bounds and optional explicit densities.',
            'For a known formulation use Calculate recipe. For property matching select a constant ε/μ or material-file target, property weights, and Find matching recipes.',
            'For a performance target choose the metric/direction, band, angles, backing-related objective, and candidate thickness range. Select the sampling budget and optional refinement, then run the search.',
            'Inspect synthesized ε/μ, property errors or the performance map, and model sensitivity. For performance objectives a requirement gap ≤ 0 passes; worst analyzed tolerance corners determine pass/fail.',
            'Select the recipe to Export selected CSV or Add selected as layer, then validate the complete multilayer stack with Impedance and Off Angle.'
        ], '<p>A property match is not automatically a good absorber. A passing performance candidate applies to its captured recipe, thickness, band, angles, and assumptions. Fractions are volumetric; densities must be supplied to derive weight fractions. Mixing laws are effective-medium predictions, so compare them with measured blends. Stop search discards an incomplete mix search; it does not preserve partial candidates like Inverse Design.</p>')),
        ('explorer', 'Workflow · Material Explorer', workflow('Material Explorer', [
            'Add or drop material CSVs, or add current stack/mix sources. Select sources to show their native-frequency curves. Add Air as a reference when useful.',
            'Read the coverage table to find a common band. Inspect ε/μ real and imaginary curves for discontinuities, unexpected units, or sign conventions.',
            'Choose a common comparison frequency and read the interpolated values and loss tangents. Out-of-range values are unavailable; FREDDY does not extrapolate.',
            'Open the raw-values table only when needed. If a file changes on disk, reload it explicitly; failed reload keeps the last valid data visible with a status message.'
        ], '<p>Expect passive loss to have negative imaginary ε/μ in the e<sup>+jωt</sup> convention. This view inspects files without editing the stack or exporting a GHOST coating. Explorer sources are session-only and are not saved in the project.</p>')),
        ('plots-analysis', 'Plot catalog · sweeps and IBC', '<p>Available views depend on the mode and captured run. Metric chooses PEC/air reflection, phase, absorption, or transmission where supported.</p>' + catalog(ANALYSIS_PLOTS) + '''
<h3>Comparison controls and exports</h3><p>Comparison limit and Band GHz apply to <b>reflection</b>. The chosen reflection basis is PEC or air; transmission uses the air basis.
Nominal, upper analyzed bound, and lower analyzed bound select pointwise data. Tolerance span is upper minus lower, and is a difference, not absolute reflection;
its comparison table uses nominal reflection. Bandwidth charts show nominal and worst analyzed reflection together.</p>
<p><b>Export comparison CSV</b> writes every sampled choice in original sweep order with its basis, target, band, widest passing bandwidth,
coverage, worst reflection, positive passing margin, full-sweep sampled null frequency, original run context, and output file.
The margin is target minus worst reflection: nonnegative passes the entire selected band. This analysis report is not an IBC input.</p>
<p>Band calculations interpolate linearly in dB between computed frequencies. Use smaller steps around narrow features and pass/fail boundaries.
Display reduction for long curves preserves extrema; it does not change full-resolution metrics or solver exports.</p>
<p><b>Color scale</b> appears above maps and is automatic by default. Clear Auto to fix Min and Max: later runs keep that scale,
so maps of different stacks match in a report. Each metric, and its tolerance span, keeps its own limits. Fixed limits are saved with
the project, and colorbar arrows mark values clipped to them.</p>'''),
        ('plots-inverse', 'Plot catalog · inverse candidates', catalog(INVERSE_PLOTS)),
        ('plots-mix', 'Plot catalog · blends and materials', catalog(MIX_PLOTS)),
        ('physics', 'Physics · variables, units and files', physics_html),
        ('troubleshooting', 'Checks · expected results and runtime', '''
<h3>Sanity checks</h3><ul>
<li><b>Lossless PEC-backed stack:</b> reflection is near 0 dB and absorbed power near zero percent. Phase and reactance can still change strongly with frequency.</li>
<li><b>Air-like, air-backed stack:</b> very small reflection, transmission near 0 dB, negligible absorption; transmission phase includes propagation.</li>
<li><b>Lossy PEC coating:</b> a reflection dip corresponds to increased absorption. A lossy material can still reflect strongly if poorly matched.</li>
<li><b>Phase discontinuity near ±180°:</b> usually wrapping, not a sudden material change. Tolerance phases are aligned to nominal before bounds are formed.</li>
<li><b>Heatmap unavailable:</b> it needs at least two frequency samples and two angles/thicknesses. Use Frequency curves or At selected frequency for a single column/row.</li>
<li><b>No sampled choice passes:</b> inspect the worst point, band, backing, polarization, and bound. A deep null or a good mean score does not guarantee a whole-band pass.</li>
<li><b>Source coverage error:</b> narrow the sweep to every input’s measured overlap or supply a valid wider-band measurement. Changing plotting limits cannot extend material coverage.</li></ul>
<h3>Keep work manageable</h3><p>Start with a coarse survey, then refine a useful region and check convergence. Add tolerances and the second polarization when needed.
Work grows with frequency samples × angles/thicknesses × tolerance cases × candidate combinations. Inverse Design reuses prepared wave terms and retains compact scores,
but finer steps in multiple variable layers multiply the search count. Mix refinement adds evaluations to each retained seed.</p>
<p>Inverse Design supports stop/keep/resume and checkpoints. Material Mix supports cooperative stopping with incomplete work discarded.
Thickness supports stopping while retaining the previous completed study. Ordinary Impedance, Off Angle, and IBC Batch jobs complete their submitted work; closing is blocked while a job is active.</p>
<h3>Know what a tolerance means</h3><p>Impedance and Off Angle evaluate configured systematic scale cases for thickness and material properties. Those controls do not independently perturb every layer,
fit measured covariance, or sample a manufacturing distribution. Their envelopes describe the analyzed cases only.
Use <a href="topic:thickness">Thickness</a> for every independent layer-thickness combination. Inverse Design retains its separate manufacturing distributions and groups. Validate with finer sampling and measurements before making engineering acceptance claims.</p>'''),
    ]


class GuideWidget(QWidget):
    mode_requested = Signal(str)

    def __init__(self, physics_html, parent=None):
        super().__init__(parent)
        self.topics = guide_topics(physics_html)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        title = QLabel('About & Guide')
        title.setStyleSheet('font-size: 20px; font-weight: 600;')
        layout.addWidget(title)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search workflows, plots, units…')
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName('Search FREDDY guide')
        row.addWidget(self.search, 1)
        self.count = QLabel()
        row.addWidget(self.count)
        layout.addLayout(row)
        split = QSplitter(Qt.Horizontal)
        self.topic_list = QListWidget()
        self.topic_list.setAccessibleName('Guide topics')
        self.topic_list.setMinimumWidth(170)
        self.topic_list.setTextElideMode(Qt.ElideRight)
        self.topic_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.topic_list.setStyleSheet('QListWidget::item { padding: 6px 8px; }')
        self.browser = QTextBrowser()
        self.browser.setAccessibleName('Guide topic content')
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.setMinimumWidth(280)
        split.addWidget(self.topic_list)
        split.addWidget(self.browser)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([245, 760])
        layout.addWidget(split, 1)
        self.topic_list.currentItemChanged.connect(self._show_topic)
        self.browser.anchorClicked.connect(self._follow_link)
        self.search.textChanged.connect(self._filter)
        self._filter('')

    def _filter(self, query):
        current = self.topic_list.currentItem()
        previous = current.data(Qt.UserRole) if current else 'overview'
        self.topic_list.clear()
        words = query.casefold().split()
        for key, title, html in self.topics:
            plain = re.sub('<[^>]+>', ' ', title + ' ' + html).casefold()
            if all(word in plain for word in words):
                item = QListWidgetItem(title)
                item.setToolTip(title)
                item.setData(Qt.UserRole, key)
                self.topic_list.addItem(item)
                if key == previous:
                    self.topic_list.setCurrentItem(item)
        self.count.setText(f'{self.topic_list.count()} topics')
        if self.topic_list.currentRow() < 0 and self.topic_list.count():
            self.topic_list.setCurrentRow(0)
        if not self.topic_list.count():
            self.browser.setHtml('<h2>No matching topics</h2><p>Try a shorter term, such as reflection, tolerance, thickness, or CSV.</p>')

    def apply_theme(self, colors):
        palette = self.browser.palette()
        for role in (QPalette.Link, QPalette.LinkVisited):
            palette.setColor(role, QColor(colors['accent']))
        self.browser.setPalette(palette)
        self.browser.document().setDefaultStyleSheet(
            'a { color: ' + colors['accent'] + '; } th { background-color: '
            + colors['head_bg'] + '; } td { border-bottom: 1px solid '
            + colors['panel_bg'] + '; }')
        self._show_topic(self.topic_list.currentItem())

    def _show_topic(self, item, _previous=None):
        if item is None:
            return
        key = item.data(Qt.UserRole)
        _key, title, html = next(topic for topic in self.topics if topic[0] == key)
        self.browser.setHtml(f'<h2>{escape(title)}</h2>{html}')
        query = self.search.text().strip()
        if query:
            self.browser.find(query)

    def open_topic(self, key):
        self.search.clear()
        for index in range(self.topic_list.count()):
            if self.topic_list.item(index).data(Qt.UserRole) == key:
                self.topic_list.setCurrentRow(index)
                return
        self.topic_list.setCurrentRow(0)

    def _follow_link(self, url):
        if url.scheme() == 'topic':
            self.open_topic(url.path())
        elif url.scheme() == 'mode' and url.path() in MODE_TOPICS:
            self.mode_requested.emit(url.path())
