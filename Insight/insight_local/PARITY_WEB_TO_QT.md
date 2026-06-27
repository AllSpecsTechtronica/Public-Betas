# Insight Web To Qt Parity

Source of truth for interaction/style parity:
- Web HUD: `Insight/main.py`
- Local Qt HUD: `Insight/insight_local`

## Interaction Dynamics

| Area | Web | Qt Local | Status |
| --- | --- | --- | --- |
| Bootstrap loading gate | Progressive `signal/data/video/live`, fallback exit, success/failure blink | Present | Parity |
| Source swap flow | Prepare, confirm, commit, ready/fail | Present | Parity |
| Preview selection | Click card -> focus track | Present | Parity |
| Focus clear / lock | Clear button, history lock, ROI lock | Present | Parity |
| ROI direct manipulation | Toggle, move, resize, shape swap, center, presets, double-click capture | Present | Parity |
| ROI monocle animation | Circle mode: 3D push on double-click capture | Flash pulse animation | Parity |
| Timeline interactions | Toggle, close, clear, delete single, inspect archived focus, TTL | Present | Parity |
| Keyboard shortcuts | `Esc`, `V`, `R`, `T` | Present | Parity |
| ROI AI flow | Provider, prompt, async status/result | Present | Parity |
| Transport reconnect | Browser-only WebRTC reconnect loop | Not applicable locally | Divergent by design |
| Interim speech captions | Browser speech interim + final | Final-only today | Gap |

## Style Dynamics

| Area | Web | Qt Local | Status |
| --- | --- | --- | --- |
| Flight strip | Dense chip row (80px min-width, 6px 12px padding), buttonish source chip | Present | Parity |
| Flight strip title | ATLAS / Tactical vision system | ATLAS / Tactical vision system | Parity |
| Preview rail | Card with thumbnail, event pill, 2x2 stat grid | Present | Parity |
| Focus stats | Four tactical stat chips | Present | Parity |
| Focus overlay heatmap | 3-layer radial gradient (outer, middle, core) | 3-layer radial gradient | Parity |
| Focus overlay boxes | Dashed bbox, label, corner ticks | Present | Parity |
| Scan rows | Confidence bar + percent + area | Present | Parity |
| Event feed | Alternating rows, high-contrast tags | Present | Parity |
| Timeline evidence cards | 132px wide, TTL ring, thumb, delete affordance | Present | Parity |
| Circle ROI | Inner ring (40%), bottom-half textured fill, horizontal dashed crosshair | Present | Parity |
| ROI label | "ROI" | "ROI" | Parity |
| Subtitle bar | Spectrum + interim/final speech styling | Spectrum + final only | Gap |
| Loading gate | Atlas Link, 4-stage progress, blink success/failure | Present | Parity |
| Video veil | Subtle dark gradient overlay | Present | Parity |
| Tab styling | Bottom red signal underline, text-shadow on active | Bottom red underline, active highlight | Parity |
| Sidebar perspective | CSS `perspective(900px) rotateY(-4deg)` | No 3D transform (Qt limitation) | Minor delta |

## Remaining Gaps

1. Interim speech captions: Web Speech API provides real-time interim text; Qt uses google SpeechRecognition which delivers final-only. This is a platform-level gap.
2. Sidebar 3D perspective: CSS 3D transforms (`rotateY`) have no direct Qt equivalent. The sidebar functions identically but lacks the subtle visual skew.
