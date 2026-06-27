# Neo-Swiss

**UI specification for the cvLayer workbench.**
A scientist's instrument, not a dashboard.


| Field     | Value                                         |
| --------- | --------------------------------------------- |
| Document  | UI-001                                        |
| Revision  | 0.1.0                                         |
| ViewPort     | cvLayer / Techtronica                         |
| Companion | `neo_swiss_ui_guide.html` (rendered specimen) |
| note      | use marathon color scheme                     |


---

## /00 Philosophy

Neo-Swiss is the engineering translation of a layered design philosophy. It is not a single style. It is the deliberate composition of four philosophical traditions, each assigned to a specific layer of the system, each doing a specific job that the others cannot do.

The first layer is the **Swiss substrate** — the rigorous twelve-column grid, the neo-grotesque typography, the desaturated graphite palette, the hairline border grammar. This layer comes from the Müller-Brockmann / Emil Ruder / Hans Hofmann tradition of mid-twentieth-century Zürich and Basel design. Its job is to provide *navigability under density*. Without it, the workbench would be unreadable; with it, the operator's eye can saccade across an information-dense surface confidently because similar things appear in similar positions and visual hierarchy is provided by mathematical relationships rather than by decorative emphasis.

The second layer is the **Japanese maximalist accent layer** — the willingness to cluster information densely, the use of saturated accent color for categorical differentiation, the technical-callout vocabulary borrowed from 1990s and early-2000s Japanese magazine design. This layer comes from publications like *Gun Professionals*, *Option*, *Famitsu*, and the work of studios like The Designers Republic. Its job is to provide *categorical legibility under density*. When a workbench surface contains forty different kinds of state simultaneously, the operator's eye needs visual channels beyond typography and layout to distinguish categories. Surgical color accents do that work faster than any text label could.

The third layer is the **CAD annotation grammar** — the dimensional callouts, the bounding boxes, the wireframe overlays, the monospace technical labels, the willingness to render hex codes and timestamps and identifiers as visible design elements. This layer borrows from engineering-drawing conventions, Figma's developer-handoff vocabulary, the visual language of brands like ACRONYM and Teenage Engineering. Its job is to provide *epistemic credibility*. When the workbench renders annotations in this register, it communicates that the system is being measured, that relationships between elements are quantified, that the operator is looking at instrumentation rather than illustration. For senior practitioners, this trust signal is enormous.

The fourth layer is the **eco-brutalist warmth** — warm material tones (oak, clay, botanical green) reserved exclusively for surfaces where humans speak: notes, hypotheses, comments, investigation narratives, README-equivalent content. This layer borrows from the Brazilian eco-brutalist tradition of Lina Bo Bardi, Paulo Mendes da Rocha, and the integrated landscapes of Roberto Burle Marx. Its job is to provide *humane warmth in a long-dwell environment*. Workbenches are inhabited for hours and days at a time; without warmth somewhere in the system, sterile cognitive fatigue sets in. By restricting the warmth to human-authored content, the warmth becomes a signal — the visual register tells the operator immediately whether they are reading machine-reported state or human-authored thought.

The system is built for senior practitioners performing investigative work over heterogeneous ML state. It rejects progressive disclosure: experts find click-through gating insulting and slow. It rejects aggregation as default: scientists need disaggregated views and distributions, not averages and summary judgments. It rejects judgments where evidence is possible: instead of green checkmarks meaning "healthy," show the actual metrics and let the operator's trained eye render the judgment. The user already understands the system; the surface lays itself out for inspection.

Every decision in this guide derives from one of those four layers. When a design choice is unclear, return to the layer it belongs to and ask what that layer's commitment requires.

---

## /01 Color tokens

The base palette is a desaturated graphite ramp running from pure void to paper. Saturated color is reserved for state semantics and applied surgically. The eco-brutalist layer adds warm-material tones for human-authored content only. There are three distinct color registers in the system, and they must never be conflated.

### /01.A Substrate ramp — desaturated graphite

The substrate is twelve stops from deepest void to highest paper. Each stop has a specific role; they are not interchangeable.


| Token             | Hex       | Role                                                                 |
| ----------------- | --------- | -------------------------------------------------------------------- |
| `--ns-pure-void`  | `#050608` | Page background. The app void. The deepest surface in the system.    |
| `--ns-void`       | `#0A0B0D` | Cell streams, terminal panels, the inside of code surfaces.          |
| `--ns-graphite-1` | `#131518` | Default panel surface. The standard container background.            |
| `--ns-graphite-2` | `#1A1D21` | Raised panel. Hover surface. Slight elevation cue.                   |
| `--ns-graphite-3` | `#23272D` | Row separators. Inactive borders. The faintest visible line.         |
| `--ns-graphite-4` | `#2E333A` | Default border. The hairline rule that defines panel edges.          |
| `--ns-graphite-5` | `#3F4651` | Strong border. Focus state. Selection emphasis.                      |
| `--ns-mist-1`     | `#5A6370` | Annotation text. Idle entity state. The quietest legible foreground. |
| `--ns-mist-2`     | `#7C8593` | Secondary text. Meta values. Timestamps and IDs in normal state.     |
| `--ns-mist-3`     | `#A4ACB8` | Body text default. Most prose copy in the workbench.                 |
| `--ns-bone`       | `#D8DCE2` | Emphasized body. Numeric values. Entity names.                       |
| `--ns-paper`      | `#ECEEF2` | Display headings. Primary text where maximum hierarchy is needed.    |


The ramp is intentionally cool. It carries no warm cast, because warm tones are reserved for the eco-brutalist layer and any leakage of warmth into the substrate would erode that semantic boundary. If a designer is tempted to warm up the substrate to make it feel friendlier, the answer is that friendliness in this system comes from the eco-brutalist layer applied to human-content surfaces, not from globally warming the substrate.

### /01.B Surgical accents — state semantics

Each accent has exactly one semantic role. Reusing accents for any other purpose breaks the system's signaling contract. There are four accents and they cover the four states that matter operationally.


| Token             | Hex       | Semantic role                                                                   |
| ----------------- | --------- | ------------------------------------------------------------------------------- |
| `--ns-vermillion` | `#E63E2B` | **ALERT.** Human attention required. Failures, blocking errors, critical drift. |
| `--ns-amber`      | `#E8A317` | **ATTENTION.** Degraded but operational. Drift warnings. Soft-fail conditions.  |
| `--ns-cyan`       | `#2BC4D9` | **PROCESS.** Active system events. Running jobs. In-flight operations.          |
| `--ns-radiation`  | `#5BE872` | **ALIVE.** Healthy. Actively used. Successfully completed. Electric vitality.   |


Each accent also has a dimmed companion (`--ns-vermillion-dim`, `--ns-amber-dim`, `--ns-cyan-dim`, `--ns-radiation-dim`) used when the state is present but should not pull foreground attention — for example, the alive indicator on a registry row that is not the operator's current focus. The dim variants exist precisely so the bright variants can be reserved for active foreground emphasis without blowing out the system every time something is healthy.

The naming of *radiation* for the alive accent is deliberate. The earlier candidate name *botanical* described an organic mossy green that sits in the eco-brutalist warmth layer; it does not carry the right semantic weight for state signaling. *Radiation* signals active emission — the green of a Geiger counter trace, of CRT phosphor, of an oscilloViewPort reading a strong signal. It communicates that the entity is not merely present but actively healthy, which is what the alive role requires.

### /01.C Eco-brutalist warmth — human-content layer

This layer is reserved for surfaces where humans speak: notes, hypotheses, comments, investigation narratives, README content, any annotation authored by an operator rather than reported by the system. It must never be used for system-reported state. The warm cast is the visual signal that says "a person wrote this."


| Token            | Hex       | Role                                                                                          |
| ---------------- | --------- | --------------------------------------------------------------------------------------------- |
| `--ns-oak`       | `#3A2E22` | Recessed surface for human notes. The dark-wood ground for written content.                   |
| `--ns-oak-warm`  | `#5C4830` | Annotation block accents. Mid-warmth tone.                                                    |
| `--ns-oak-grain` | `#7D6240` | Borders on human-authored content. The visible-grain edge.                                    |
| `--ns-clay`      | `#8C5A3C` | Optional cultural-warmth accent. Used sparingly.                                              |
| `--ns-botanical` | `#4A8C3F` | Mossy organic green. Plant-life imagery in human-content surfaces only. Never used for state. |


Note specifically the relationship between *botanical* and *radiation*. They are both green. They are not interchangeable. *Botanical* is the deep mossy green of established plant life — it sits in the warmth layer, alongside oak and clay, and is used only inside human-content surfaces if green is needed for organic imagery. *Radiation* is the electric vital signal-green at the surgical accent layer. Confusing them collapses two different semantic registers into one and breaks the system.

---

## /02 Typography

The system uses two typefaces. **Inter** (with Söhne or ABC Diatype as commercial alternates) carries human-readable content. **JetBrains Mono** (with Berkeley Mono or IBM Plex Mono as alternates) carries machine-readable values: identifiers, hashes, timestamps, metrics, hex codes. The bilingual encoding is itself an information channel — operators learn within minutes that monospace is system truth and proportional sans is human framing, and that bilingual encoding makes the dense page parseable.

### /02.A Type scale

The scale is restricted on purpose. Fluid type and decorative size variation produce drift; a small fixed scale produces coherence.


| Role       | Size | Weight                                | Use                                                                                |
| ---------- | ---- | ------------------------------------- | ---------------------------------------------------------------------------------- |
| Display    | 32px | 500                                   | Mastheads only. The largest text in the system.                                    |
| Large      | 24px | 500                                   | Page-level titles.                                                                 |
| Section    | 18px | 500                                   | Section heads within a page.                                                       |
| Emphasis   | 15px | 500                                   | Panel titles, entity names.                                                        |
| Base       | 13px | 400                                   | Workbench body. The default reading size. The workbench is dense; 13px is correct. |
| Small      | 12px | 400                                   | Subordinate prose. Helper copy. Inline descriptions of secondary fields.           |
| Mono base  | 13px | 400                                   | Identifiers, paths, structured machine values.                                     |
| Mono mini  | 11px | 400                                   | Timestamps, metrics, ETAs.                                                         |
| Mono micro | 10px | 400, 0.05em letter-spacing, uppercase | Technical annotations, badge labels, callout dimensions.                           |


### /02.B Numeral discipline

Numerals are tabular. Use `font-feature-settings: "tnum"` wherever numbers may change at runtime — metrics, counters, timestamps, percentages. Proportional figures cause horizontal jitter that destroys density. This is non-negotiable because the workbench depends on the operator's eye scanning numeric columns; if the digits don't align, the scan fails.

### /02.C Letter spacing and case

Body text uses default letter spacing and sentence case. Mono micro labels use 0.05em letter spacing and uppercase; this is the only place uppercase appears in the system. Mono mini timestamps and metrics use 0.02em letter spacing and remain in their natural case. Display headings use slightly tightened letter spacing (-0.02em) for optical balance at large size.

---

## /03 Grid and spacing

### /03.A Page grid

Twelve-column page grid. No inter-column gutter. Panels share hairline borders rather than negative space. This is the most distinctive structural choice in the system and the one that most strongly differentiates Neo-Swiss from contemporary enterprise design. Modern enterprise UIs use generous whitespace as a luxury signal; Neo-Swiss uses adjacency and shared hairlines because the workbench is a dense investigative instrument and every pixel of negative space is a pixel where information could have been.

### /03.B Vertical rhythm

All component dimensions, padding, and gaps must be a multiple of 4 pixels. The base unit is 4px. The standard scale is 0, 2, 4, 8, 12, 16, 24, 32, 48, 64. Drift in spacing is the first sign of system rot — when a designer reaches for 7px or 13px or 27px because something "looks right," they are usually fixing a different problem (typography off, hierarchy unclear) by adjusting spacing.

### /03.C Border radius

Border radius is 0 by default, 1px on small accents, 2px maximum. The system is rectilinear. Rounded corners signal consumer SaaS, lifestyle apps, marketing surfaces. Neo-Swiss is instrumentation. Instruments have square corners.

### /03.D Composition rules

- Adjacent panels share a 0.5px graphite-4 border. The shared edge is the seam.
- Panel internal padding is typically 16px (`--ns-s-5`) but may be 12px in dense contexts and 24px in display contexts.
- Vertical gaps between elements within a panel are 8px (`--ns-s-3`) by default.
- Section-to-section vertical gap on a page is 64px (`--ns-s-9`).

---

## /04 Technical annotation grammar

The annotation grammar is what makes the system *Neo-Swiss* rather than just Swiss. Every entity carries small inline annotations — identifier, version, timestamp, type, lineage depth — rendered in monospace with hairline boxes, in the visual register of CAD drawings and engineering diagrams. This is the layer that says *this is instrumentation, not illustration*.

There are two distinct callout forms. They serve different purposes and must never be mixed.

### /04.01 Tagged callout — semantic state

A tagged callout is a small inline element consisting of a 6×6px square indicator followed by uppercase mono micro text. The indicator may be empty (border only, signaling neutral metadata), filled cyan (signaling active process), or filled vermillion (signaling alert state). Examples of correct use:

- `[ ] SCN.FALL.V3` — neutral entity tag
- `[■] JOB.RUNNING` — active process state, cyan-filled indicator
- `[■] DRIFT.0.84` — alert state, vermillion-filled indicator

Tagged callouts are used for entity-level tags, state badges, and provenance stamps. The text is uppercase mono micro (10px, 0.05em letter spacing). The indicator is a 6×6px square with 1px border in `--ns-mist-1` when empty, or filled with the surgical accent of the state being communicated.

### /04.02 Dimensional callout — quantitative value

A dimensional callout is a small inline element with a 0.5px graphite-5 border and 1px-by-4px padding, containing uppercase mono micro text. There is no leading indicator. Examples:

- `[ 1X ]` — dimensional ratio
- `[ 640PX ]` — pixel dimension
- `[ SHA:7F4A92 ]` — hash identifier
- `[ 0.847 mAP ]` — quantitative metric

Dimensional callouts are used for dimensions, ratios, hashes, instrument readings, and any quantitative or identifier value that needs to be visibly tagged as a measurement rather than as prose.

### /04.03 The distinction is rigid

Tagged callouts (with leading indicator) carry semantic state. Dimensional callouts (bordered, no indicator) carry quantitative or identifier values. Mixing the forms — using a leading indicator on a dimensional value, or omitting the border on a state tag — collapses the visual distinction and the operator loses a reliable parsing channel. Treat the two forms as separate vocabulary.

---

## /05 Component grammar

The workbench is composed from a small set of primitives. Every panel in cvLayer should resolve into combinations of these. New components require justification — the cost of a new primitive is system-wide, the cost of a recombination is local.

### /05.01 Panel

The panel is the fundamental container. Specification:

- Background: `--ns-graphite-1` (`#131518`).
- Border: 0.5px solid `--ns-graphite-4`.
- Border radius: 0.
- Internal padding: 16px standard.
- The panel has a header rail at top, separated from the panel body by a 0.5px graphite-3 horizontal hairline.

The header rail itself has a defined structure. On the left side it carries: a panel ID in mono micro (e.g., `SCN.REGISTRY`) in `--ns-mist-1`; the panel title in 13px sans 500 weight in `--ns-paper`; optionally a tagged callout summarizing the panel's state (e.g., `[ ] 7 ACTIVE`). On the right side it carries metadata in mono micro: sort order, filter state, and any other panel-level controls expressed as text rather than as buttons. The right side is informational and dense; the operator can see at a glance what configuration the panel is in without clicking anything.

### /05.02 Entity row

The entity row is the primitive for rendering one entity in a list. It is a 5-column grid:

- Column 1 (16px): a state dot, 8×8px, colored by entity state (alive/alert/attention/process/idle).
- Column 2 (1fr): the entity name in 13px sans 500 weight, color `--ns-paper`.
- Column 3 (auto): the entity ID and type in mono mini, color `--ns-mist-1`. Format: `v3 :: yolo_detection`.
- Column 4 (auto): the entity's primary metric or current value in mono mini, color `--ns-bone`. Format: `mAP 0.847` or `TRAINING 38%`.
- Column 5 (auto): the entity's last-change timestamp in mono micro, color `--ns-mist-1`. Format: `04 MAY 14:22`.

Rows are separated by 0.5px graphite-3 horizontal lines. Vertical padding per row is 4px (so each row is roughly 24px tall). This produces a density of about 40 rows per 1000px of vertical space — dense enough to scan a model registry at a glance, sparse enough that individual rows remain legible.

### /05.03 Cell stream

The cell stream renders the Jupyter-analogous output of a backbone in real time, streamed from the WebSocket `/events` channel. Specification:

- Background: `--ns-pure-void` (`#050608`). The inside of the cell stream is the deepest surface in the system, a deliberate contrast with the panel surface around it.
- Border: 0.5px solid `--ns-graphite-4`.
- Padding: 12px.
- Font: mono mini (11px), line height 1.7.
- Each line is a 2-column grid: an 80px-wide timestamp column in `--ns-mist-1`, and a 1fr message column whose color depends on severity.

Severity colors:

- `info` — `--ns-mist-3` (default informational message)
- `ok` — `--ns-radiation` (success, best-so-far, gate passed)
- `warn` — `--ns-amber` (guard exceeded, soft warning)
- `err` — `--ns-vermillion` (failure, blocking error)
- `cyan` — `--ns-cyan` (system event, configuration applied)

Cell streams are never hidden behind a tab. While a backbone is active, its cell stream must be visible somewhere in the operator's current workspace. This is one of the load-bearing trust commitments of the system: the operator can always see what the system is doing in real time, without having to navigate to a "logs" view.

### /05.04 Metric block

The metric block is the primitive for displaying a single quantitative value with optional delta and provenance. Structure:

- A label in mono micro (10px uppercase, 0.05em letter spacing), color `--ns-mist-1`. The label includes the unit and provenance, not just the metric name. Example: `mAP @ 50 / V3` rather than just `mAP`.
- A value in mono large (18px), color `--ns-paper`. Tabular numerals are mandatory. Letter spacing -0.01em for optical correctness at this size.
- An optional delta in mono mini (11px), color `--ns-mist-2` for neutral deltas, `--ns-amber` for concerning deltas, `--ns-radiation` for positive deltas. Format: `+0.034 vs v2`.

Metric blocks may be assembled into horizontal strips of 2-4, separated by 0.5px graphite-4 vertical lines, contained inside a larger panel. The strip pattern is preferred over standalone metric blocks because it lets the operator's eye compare across related metrics in one saccade.

### /05.05 State badge

The state badge is the primitive for inline state indication where a tagged callout would be too small. Specification:

- Padding: 2px vertical, 6px horizontal.
- Border: 0.5px solid currentColor (the badge's text color).
- Background: transparent. The badge reads as line-art instrumentation, not as a colored chart.
- Font: mono micro (10px uppercase, 0.06em letter spacing).
- Color: the surgical accent corresponding to the state (vermillion, amber, cyan, radiation, or mist-1 for idle).

Examples: `[ ALIVE ]`, `[ RUNNING ]`, `[ DEGRADED ]`, `[ FAILED ]`, `[ IDLE ]`, `[ VERIFIED ]`, `[ QUEUED ]`, `[ DRIFT ]`.

The transparent-background-with-colored-border-and-text approach is deliberate. Filled badges (color background with white or contrasting text) belong to consumer SaaS vocabulary; line-art badges belong to instrumentation vocabulary. The system reads as drafted rather than printed.

---

## /06 Workbench specimen — full assembly

The canonical workbench layout assembles the primitives into a five-zone composition. This is the layout that should be the first thing an operator sees when opening cvLayer, and it should be replicable across the platform's other surfaces (the tactical HUD, future Techtronica MLOps surfaces) so the visual grammar transfers.

### /06.A Top rail

A horizontal strip across the top of the page, full width, with a 0.5px graphite-4 bottom border. The rail is divided into a left half and a right half.

The left half carries: a system identifier in mono micro paper-color (`CVLAYER / WORKBENCH`); a service health indicator in mono micro mist-1 (`PORT 8787 / OK`); and a series of tagged callouts summarizing global system state (`[ ] 7 SCENARIOS`, `[■] 2 RUNNING` cyan-filled, `[■] 1 ALERT` vermillion-filled).

The right half carries: a current timestamp in mono micro mist-2 in ISO format (`2026-05-04T14:22:08Z`); and a session duration indicator in mono micro mist-1 (`SESSION 4H 12M`).

The top rail is the operator's at-a-glance system pulse. Without looking at any panel below, the operator can see whether the system is healthy, what's running, what needs attention, and how long their current session has been.

### /06.B Left column — scenario registry (280px wide)

A panel containing:

- Header rail with title `SCENARIO REGISTRY` and panel ID `/REG.01`.
- A vertical list of scenario entity rows in compact form (mini variant: 6×6px state dot, name, version ID; no metric or timestamp columns at this density).
- A 0.5px graphite-4 horizontal divider.
- Below the divider, a section labeled `DATASETS` in mono micro, followed by a list of dataset entity rows in the same compact form.

The left column is the operator's primary navigation surface. Clicking any entity row promotes that entity to the focal panel in the center column.

### /06.C Center column — focal investigation (1fr, fills available space)

A panel containing the currently focused entity's full state. For a focused scenario like `fall_detection`:

- Header rail with title `FOCUS / FALL_DETECTION V3` and panel ID `/INV.01`.
- A horizontal strip of tagged callouts summarizing focal state: `[ ] SCN.FALL.V3`, `[■] VERIFIED` cyan-filled, `[ SHA:7F4A92 ]` dimensional callout.
- A 2×2 grid of metric blocks: `mAP @ 50 / V3` showing 0.847; `FRAMES` showing 4,217; `P99 INFER MS` showing 42.1; `DRIFT 24H` showing 0.31 in amber (because 0.31 indicates attention-needed).
- A 0.5px graphite-4 horizontal divider.
- A cell stream labeled `CELL STREAM / TRAIN` showing the most recent training output, scrolling upward as new lines arrive.

The center column is where the operator's investigation lives. It changes as the operator promotes different entities; everything in the column is contextual to the focal entity.

### /06.D Right column — small multiples (320px wide)

A panel containing a 2×3 grid of small sparkline tiles. Each tile is a small-multiple chart of one operational metric over time. Specification per tile:

- Background: `--ns-pure-void`.
- Border: 0.5px solid `--ns-graphite-4`.
- Padding: 8px.
- A label in mono micro mist-1.
- A 24px-tall sparkline in SVG. Stroke color matches the metric's semantic register: mist-3 for neutral, radiation for positive metrics trending up (mAP), amber for concerning trends (latency, drift), cyan for activity-level metrics (queue depth).
- A current value in mono base bone-color.

The six tiles in the canonical layout: LOSS, mAP, P99 LATENCY, DRIFT, QUEUE, COST/HR. Tufte's small-multiples principle is doing its full work here — anomalies pop because they break the visual rhythm of the grid, and the operator can scan all six metrics in a single saccade.

### /06.E Bottom — event pulse strip

A horizontal strip across the bottom of the workbench, full width, with a 0.5px graphite-4 top border. Background `--ns-graphite-1`. The strip renders a single line of mono micro text in mist-2 color, scrolling horizontally (or static, depending on operator preference), showing the most recent system events from the WebSocket pulse. Format: `[timestamp] / [ViewPort] :: [event]` separated by `/`.

The pulse strip is the workbench's continuous heartbeat. The operator never has to look at it directly, but its presence in peripheral vision communicates that the system is alive and breathing. When something concerning enters the pulse — a vermillion or amber-colored event — the operator's peripheral vision catches it immediately even if they're focused elsewhere.

---

## /07 Rules of composition

The system holds together because every contributor follows a small set of structural laws. These rules are the load-bearing constraints of Neo-Swiss. Breaking them does not produce variation — it produces incoherence.

### /07.01 — Saturated color is reserved for state

**Do:** Reserve saturated color for state semantics. Vermillion = alert. Amber = degraded. Cyan = process. Radiation = alive.

**Do not:** Use accent colors for visual interest, branding, or category coloring outside their semantic role. The moment radiation appears on something that is not actively-alive, the operator can no longer trust the alive-signal anywhere in the system.

### /07.02 — Show evidence at the level of judgment

**Do:** When a model is "healthy," show the metric, not just the green dot. When a job is "running," show the progress percentage and ETA, not just the cyan indicator. The state indicator and the underlying evidence appear together.

**Do not:** Replace data with verdicts. Scientists distrust judgments because they know how much complexity they hide. A green checkmark with no accompanying number is a marketing surface, not a scientific instrument.

### /07.03 — Bilingual encoding is a system commitment

**Do:** Use mono for system truth (IDs, timestamps, metrics, hashes, dimensions). Use sans for human framing (labels, descriptions, narratives, panel titles).

**Do not:** Mix mono and sans within a single semantic role. A timestamp rendered in sans loses its bilingual signal; a panel title rendered in mono looks like data. The operator's typographic parsing is fast and reliable only if the encoding is consistent.

### /07.04 — Identity is always visible

**Do:** Every entity displays its identity (name, version, ID) and its temporal context (last change, age) in its rail. The operator can always tell what they are looking at and when it last changed.

**Do not:** Strip identity to "look cleaner." Anonymous rows lose lineage and forfeit the system's epistemic credibility. The operator should never have to hover or click to find out what version of a thing they are seeing.

### /07.05 — Relationships are ambient

**Do:** Render relationships as ambient connections. Lineage and dependency are continuous visual elements, not modal views.

**Do not:** Hide relationships behind a "lineage" tab. Relationships are first-class data and must be persistently visible. The detective-corkboard metaphor only works if the strings between artifacts are always there, not retrieved on request.

### /07.06 — Density is the default

**Do:** Maintain density. The default workbench page should fill its ViewPort with information; whitespace serves prose, not state.

**Do not:** Add padding to "let the design breathe." Breathing space is for marketing surfaces, not investigative instruments. The operator's appetite for density is one of the things the system is respecting; reducing density is a failure of respect.

### /07.07 — Warmth signals human voice

**Do:** Reserve the eco-brutalist warm layer (oak, clay, botanical) for human-authored content: notes, hypotheses, comments, narratives. Botanical is the organic green; radiation is the electric alive-state green — they are not interchangeable.

**Do not:** Apply warm tones to system-reported state. The warmth signals human voice; misuse erodes the signal. If the operator cannot distinguish at a glance between machine-reported and human-authored content, the system has lost a critical channel.

### /07.08 — Motion is informational

**Do:** Keep animations damped, mechanical, and short — under 180ms. Movement is informational, not expressive.

**Do not:** Use bouncy, elastic, or overshoot easings. Decorative motion is consumer-app vocabulary. Workbench motion exists only to confirm causality (this changed because of that) and to draw attention (this needs your eyes), never to delight.

---

## /08 Implementation notes

Two implementation targets matter: the web stack (FastAPI dashboard at port 8787, future React/Vue surfaces) and the PyQt6 stylesheet for the cvLayer desktop UI. The token names below are stable; the values are starting positions and may be tuned through use.

### /08.A CSS variable export

```css
:root {
  /* substrate */
  --ns-pure-void:    #050608;
  --ns-graphite-1:   #131518;
  --ns-graphite-3:   #23272D;
  --ns-graphite-4:   #2E333A;
  --ns-mist-1:       #5A6370;
  --ns-bone:         #D8DCE2;
  --ns-paper:        #ECEEF2;

  /* surgical accents */
  --ns-vermillion:   #E63E2B;  /* alert */
  --ns-amber:        #E8A317;  /* attention */
  --ns-cyan:         #2BC4D9;  /* process */
  --ns-radiation:    #5BE872;  /* alive */

  /* eco-brutalist warmth */
  --ns-oak:          #3A2E22;
  --ns-clay:         #8C5A3C;
  --ns-botanical:    #4A8C3F;  /* organic green for human content */

  /* typography */
  --ns-font-sans:    "Inter", -apple-system, system-ui, sans-serif;
  --ns-font-mono:    "JetBrains Mono", "Berkeley Mono", monospace;
}
```

### /08.B PyQt6 stylesheet excerpt

```css
QMainWindow, QWidget {
    background-color: #050608;
    color: #D8DCE2;
    font-family: "Inter", sans-serif;
    font-size: 13px;
}

QFrame[role="panel"] {
    background-color: #131518;
    border: 1px solid #2E333A;
    border-radius: 0;
}

QLabel[role="mono"] {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    color: #7C8593;
    letter-spacing: 0.5px;
}

QLabel[state="alert"]   { color: #E63E2B; }
QLabel[state="warn"]    { color: #E8A317; }
QLabel[state="process"] { color: #2BC4D9; }
QLabel[state="alive"]   { color: #5BE872; }
```

Use Qt's dynamic property mechanism to toggle states at runtime:

```python
label.setProperty("state", "alert")
label.style().polish(label)
```

This keeps semantic state declarative rather than scattered through palette manipulation. When the underlying ontological state changes (a job transitions from running to failed), the corresponding Qt property is updated and the stylesheet re-polishes the widget — the visual change is a consequence of the state change, not a side effect of code that happened to run elsewhere.

### /08.C Implementation order

When implementing the system in cvLayer, the recommended order is:

1. Establish the design tokens as a Qt stylesheet at the application level. Without the tokens defined, every subsequent step has to re-decide what colors and sizes mean.
2. Migrate one heavily-used panel (the Catalog Panel is the recommended starting point, since it's the operator's primary navigation surface) to the new system in full. This pressure-tests the tokens and surfaces gaps before they propagate.
3. Migrate the Dataset Panel next. The audio-mode Dataset Panel is the densest information surface in cvLayer and exercises every primitive in the system.
4. Migrate the Training Console Panel. The cell-stream primitive lives here and is critical to the workbench's trust commitments.
5. Add the top rail and event pulse strip to the main window. These are global elements, not panel-internal, and their addition transforms the felt experience of the system from static-tool to live-instrument.
6. Migrate the remaining panels in order of operator-facing importance.

Visual-language migration can proceed in parallel with new feature work; the panels currently working in default Qt styling will continue to work during the migration, just with visual inconsistency. The inconsistency is the cost of incremental migration and is preferable to the cost of a flag-day rewrite.

---

## /09 Reference index — lineage of the system

Neo-Swiss is a synthesis. It is not invented from scratch. Every layer has antecedents that are worth knowing because they tell future contributors what the system is trying to be.

**Swiss substrate.** Josef Müller-Brockmann's grid systems. Emil Ruder's typography. Hans Hofmann's compositional rigor. The lineage from Akzidenz-Grotesk through Helvetica to Inter. The substrate provides navigability under density.

**Japanese maximalist accent.** 1990s and 2000s Japanese magazine layouts — *Gun Professionals*, *Option*, *Famitsu*, *Combat Magazine*. The Designers Republic's information-dense game-industry work. Information density treated as respect for the reader. Saturation as categorical channel.

**CAD annotation.** Engineering drawing conventions. Figma's developer-handoff vocabulary. ACRONYM technical apparel branding. Teenage Engineering's product packaging. Annotations as instrumentation signal.

**Eco-brutalist warmth.** Lina Bo Bardi's MASP. Paulo Mendes da Rocha's concrete-and-light architecture. Roberto Burle Marx's integrated landscapes. Material honesty plus organic warmth. Reserved for human-authored layers.

**Information design tradition.** Edward Tufte on data density and small multiples. Ben Shneiderman's overview-first principle. The Bloomberg terminal as the canonical investigative-density precedent. NASA mission control telemetry boards. The hospital monitoring station as a counterexample of monitoring-density mistaken for investigative-density.

**Domain precedent.** VS Code's panel system as the IDE inheritance source. Observable's reactive notebook model. Linear's ambient state model. Cyberpunk 2077's netrunner UI as the visual precedent for the synthesis. Hex's analyst workbench layout. Mathematica and Bloomberg as workbench archetypes. The detective corkboard as the operational metaphor.

---

## End matter

This document is the prose twin of `neo_swiss_ui_guide.html`. The HTML version demonstrates the system by being an instance of it; this version describes the system in words. Anyone implementing Neo-Swiss in cvLayer, Techtronica, or any future surface should read both. Disagreements between the two documents should be resolved in favor of the HTML version, because the HTML is self-validating — if a token value or component specification appears different in the two documents, the rendered HTML is the source of truth.

Revisions to this document should match revisions to the HTML guide. The two are versioned together.

---

NEO-SWISS / UI-001 / REV 0.1.0  
CVLAYER WORKBENCH  
2026-05-04