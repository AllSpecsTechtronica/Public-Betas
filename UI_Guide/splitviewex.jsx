import React, { useState, useCallback, useRef, useEffect, useMemo } from "react";

// ============================================================
// CONSTANTS
// ============================================================

const ACTIVITY_RAIL_W = 48;
const SIDEBAR_MIN = 180;
const SIDEBAR_DEFAULT = 240;
const INSPECTOR_MIN = 200;
const INSPECTOR_DEFAULT = 260;
const BOTTOM_MIN = 80;
const BOTTOM_DEFAULT = 160;
const STATUS_BAR_H = 24;

const ACCENT = "#c57a2e";
const ACCENT_DIM = "#8a5520";
const ACCENT_GLOW = "rgba(197, 122, 46, 0.15)";
const BG_0 = "#0e0e0e";
const BG_1 = "#151515";
const BG_2 = "#1a1a1a";
const BG_3 = "#222222";
const BORDER = "#2a2a2a";
const BORDER_LIGHT = "#333333";
const TEXT_PRIMARY = "#cccccc";
const TEXT_DIM = "#666666";
const TEXT_MUTED = "#444444";
const GREEN = "#5a9a6a";
const GREEN_DIM = "#3a6a4a";
const RED = "#aa5555";
const BLUE = "#5588bb";
const CYAN = "#5aaaaa";

const FONT_MONO = "'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace";

const MODES_ECO = [
  { id: "ecosystem", icon: "ECO", label: "Ecosystem" },
];

const MODES_WORK = [
  { id: "explore", icon: "EXP", label: "Explorer" },
  { id: "test", icon: "TST", label: "Test" },
  { id: "data", icon: "DAT", label: "Data" },
  { id: "notes", icon: "NTE", label: "Notes" },
  { id: "settings", icon: "CFG", label: "Settings" },
];

const PRESETS = {
  train: { label: "TRAIN", splits: [40, 35, 25], bottomH: 180 },
  evaluate: { label: "EVAL", splits: [20, 50, 30], bottomH: 120 },
  lineage: { label: "LINEAGE", splits: [15, 20, 65], bottomH: 100 },
};

// ============================================================
// GRAPH DATA (the estate)
// ============================================================

const GRAPH_NODES = [
  { id: "s1", type: "scenario", name: "eeg_entropy_v4.5", status: "active", x: 0.35, y: 0.3 },
  { id: "s2", type: "scenario", name: "subvocal_vl6180_alpha", status: "training", x: 0.65, y: 0.25 },
  { id: "s3", type: "scenario", name: "motor_intention_dual", status: "idle", x: 0.5, y: 0.55 },
  { id: "s4", type: "scenario", name: "phonemic_decoder_v2", status: "queued", x: 0.25, y: 0.65 },
  { id: "s5", type: "scenario", name: "fusion_vl53l1x_roi", status: "idle", x: 0.75, y: 0.6 },
  { id: "cp38", type: "checkpoint", name: "cp-038 (best)", status: "saved", x: 0.2, y: 0.2 },
  { id: "cp48", type: "checkpoint", name: "cp-048 (latest)", status: "saved", x: 0.42, y: 0.15 },
  { id: "r1", type: "range", name: "rest_baseline", status: "pass", x: 0.15, y: 0.45 },
  { id: "r2", type: "range", name: "subvocal_yes_no", status: "pass", x: 0.55, y: 0.42 },
  { id: "r3", type: "range", name: "motor_reach", status: "warn", x: 0.4, y: 0.72 },
  { id: "j1", type: "job", name: "train_entropy_048", status: "running", x: 0.48, y: 0.38 },
  { id: "j2", type: "job", name: "eval_subvocal_batch", status: "queued", x: 0.78, y: 0.38 },
  { id: "a1", type: "asset", name: "nrf52840_firmware_v3", status: "stable", x: 0.85, y: 0.75 },
  { id: "a2", type: "asset", name: "vl6180_driver.ino", status: "dev", x: 0.72, y: 0.78 },
];

const GRAPH_EDGES = [
  { from: "s1", to: "cp38" }, { from: "s1", to: "cp48" }, { from: "s1", to: "r1" },
  { from: "s1", to: "j1" }, { from: "s2", to: "r2" }, { from: "s2", to: "j2" },
  { from: "s3", to: "r3" }, { from: "s3", to: "s1", label: "parent" },
  { from: "s4", to: "s3", label: "fork" }, { from: "s5", to: "a1" },
  { from: "s5", to: "a2" }, { from: "cp48", to: "j1" }, { from: "r2", to: "j2" },
  { from: "s2", to: "a2" },
];

const NODE_COLORS = {
  scenario: ACCENT,
  checkpoint: CYAN,
  range: GREEN,
  job: BLUE,
  asset: TEXT_DIM,
};

const NODE_SHAPES = {
  scenario: "circle",
  checkpoint: "diamond",
  range: "square",
  job: "triangle",
  asset: "hex",
};

const STATUS_COLORS = {
  active: GREEN, training: ACCENT, idle: TEXT_MUTED, queued: BLUE,
  saved: CYAN, pass: GREEN, warn: ACCENT, running: ACCENT,
  stable: GREEN_DIM, dev: BLUE,
};

// ============================================================
// SHARED COMPONENTS
// ============================================================

function DragHandle({ orientation, onDrag }) {
  const dragging = useRef(false);

  const onMouseDown = useCallback((e) => {
    e.preventDefault();
    dragging.current = true;
    document.body.style.cursor = orientation === "vertical" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
    const onMouseMove = (ev) => {
      if (dragging.current) onDrag(orientation === "vertical" ? ev.clientX : ev.clientY);
    };
    const onMouseUp = () => {
      dragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, [onDrag, orientation]);

  const isV = orientation === "vertical";
  return (
    <div
      onMouseDown={onMouseDown}
      style={{
        [isV ? "width" : "height"]: 5,
        cursor: isV ? "col-resize" : "row-resize",
        position: "relative", flexShrink: 0, zIndex: 2,
      }}
    >
      <div style={{
        position: "absolute",
        [isV ? "left" : "top"]: 2,
        [isV ? "top" : "left"]: 0,
        [isV ? "bottom" : "right"]: 0,
        [isV ? "width" : "height"]: 1,
        background: BORDER,
      }} />
    </div>
  );
}

function PaneHeader({ title, subtitle, actions }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "6px 10px", borderBottom: `1px solid ${BORDER}`,
      background: BG_2, flexShrink: 0, minHeight: 28,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 11, fontFamily: FONT_MONO, color: ACCENT, fontWeight: 600, letterSpacing: "0.05em" }}>{title}</span>
        {subtitle && <span style={{ fontSize: 10, fontFamily: FONT_MONO, color: TEXT_DIM }}>{subtitle}</span>}
      </div>
      {actions && <div style={{ display: "flex", gap: 4 }}>{actions}</div>}
    </div>
  );
}

function SmallBtn({ children, onClick, active }) {
  return (
    <button onClick={onClick} style={{
      background: active ? `${ACCENT}22` : "none",
      border: `1px solid ${active ? ACCENT : BORDER}`,
      color: active ? ACCENT : TEXT_DIM,
      fontFamily: FONT_MONO, fontSize: 9, padding: "1px 6px", cursor: "pointer",
      letterSpacing: "0.03em", transition: "all 0.15s",
    }}>
      {children}
    </button>
  );
}

// ============================================================
// ECOSYSTEM VIEW
// ============================================================

function EcosystemGraph({ nodes, edges, selected, onSelect, onDescend, width, height }) {
  const margin = 40;

  const nodePositions = useMemo(() => {
    const pos = {};
    nodes.forEach((n) => {
      pos[n.id] = {
        x: margin + n.x * (width - margin * 2),
        y: margin + n.y * (height - margin * 2),
      };
    });
    return pos;
  }, [nodes, width, height, margin]);

  const renderNode = (node) => {
    const p = nodePositions[node.id];
    if (!p) return null;
    const isSel = selected === node.id;
    const color = NODE_COLORS[node.type] || TEXT_DIM;
    const r = node.type === "scenario" ? 14 : 8;

    let shape;
    if (node.type === "scenario") {
      shape = <circle cx={p.x} cy={p.y} r={r} />;
    } else if (node.type === "checkpoint") {
      const s = r;
      shape = <polygon points={`${p.x},${p.y - s} ${p.x + s},${p.y} ${p.x},${p.y + s} ${p.x - s},${p.y}`} />;
    } else if (node.type === "range") {
      const s = r * 0.8;
      shape = <rect x={p.x - s} y={p.y - s} width={s * 2} height={s * 2} />;
    } else if (node.type === "job") {
      const s = r;
      shape = <polygon points={`${p.x},${p.y - s} ${p.x + s},${p.y + s * 0.6} ${p.x - s},${p.y + s * 0.6}`} />;
    } else {
      shape = <circle cx={p.x} cy={p.y} r={r * 0.7} />;
    }

    return (
      <g
        key={node.id}
        style={{ cursor: "pointer" }}
        onClick={(e) => { e.stopPropagation(); onSelect(node.id); }}
        onDoubleClick={(e) => {
          e.stopPropagation();
          if (node.type === "scenario") onDescend(node);
        }}
      >
        {isSel && (
          <circle cx={p.x} cy={p.y} r={r + 8} fill="none" stroke={color} strokeWidth={1} opacity={0.3}>
            <animate attributeName="r" values={`${r + 6};${r + 10};${r + 6}`} dur="2s" repeatCount="indefinite" />
          </circle>
        )}
        <g
          fill={isSel ? color : "none"}
          stroke={color}
          strokeWidth={isSel ? 2 : 1.2}
          opacity={isSel ? 1 : 0.7}
        >
          {shape}
        </g>
        {/* status pip */}
        <circle
          cx={p.x + r * 0.7} cy={p.y - r * 0.7} r={3}
          fill={STATUS_COLORS[node.status] || TEXT_MUTED}
          stroke={BG_0} strokeWidth={1}
        />
        {/* label */}
        <text
          x={p.x} y={p.y + r + 14}
          textAnchor="middle" fill={isSel ? TEXT_PRIMARY : TEXT_DIM}
          fontSize={isSel ? 10 : 9} fontFamily={FONT_MONO}
        >
          {node.name.length > 20 ? node.name.slice(0, 18) + ".." : node.name}
        </text>
      </g>
    );
  };

  return (
    <svg width={width} height={height} style={{ display: "block" }} onClick={() => onSelect(null)}>
      <rect width={width} height={height} fill={BG_0} />
      {/* grid */}
      <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke={BORDER} strokeWidth="0.5" opacity="0.4" />
        </pattern>
      </defs>
      <rect width={width} height={height} fill="url(#grid)" />

      {/* edges */}
      {edges.map((e, i) => {
        const from = nodePositions[e.from];
        const to = nodePositions[e.to];
        if (!from || !to) return null;
        const isHighlight = selected === e.from || selected === e.to;
        return (
          <g key={i}>
            <line
              x1={from.x} y1={from.y} x2={to.x} y2={to.y}
              stroke={isHighlight ? ACCENT : BORDER_LIGHT}
              strokeWidth={isHighlight ? 1.2 : 0.6}
              opacity={isHighlight ? 0.8 : 0.3}
              strokeDasharray={e.label ? "4 3" : "none"}
            />
            {e.label && (
              <text
                x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 5}
                textAnchor="middle" fill={TEXT_MUTED} fontSize={8} fontFamily={FONT_MONO}
              >
                {e.label}
              </text>
            )}
          </g>
        );
      })}

      {/* nodes */}
      {nodes.map(renderNode)}

      {/* legend */}
      <g transform={`translate(${width - 150}, 16)`}>
        {Object.entries(NODE_COLORS).map(([type, color], i) => (
          <g key={type} transform={`translate(0, ${i * 16})`}>
            <circle cx={6} cy={6} r={4} fill="none" stroke={color} strokeWidth={1} />
            <text x={16} y={10} fill={TEXT_MUTED} fontSize={9} fontFamily={FONT_MONO}>{type}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function EcosystemInspector({ node, allNodes, edges, onDescend }) {
  if (!node) {
    return (
      <div style={{ padding: 16, color: TEXT_MUTED, fontSize: 11, fontFamily: FONT_MONO }}>
        <div style={{ marginBottom: 12 }}>// select a node in the graph</div>
        <div style={{ color: TEXT_DIM, lineHeight: 1.6 }}>
          click to inspect<br />
          double-click scenario to descend into workbench
        </div>
      </div>
    );
  }

  const neighbors = edges
    .filter((e) => e.from === node.id || e.to === node.id)
    .map((e) => {
      const otherId = e.from === node.id ? e.to : e.from;
      const other = allNodes.find((n) => n.id === otherId);
      return { edge: e, node: other };
    })
    .filter((n) => n.node);

  const quickActions = [];
  if (node.type === "scenario") {
    quickActions.push({ label: "DESCEND TO WORKBENCH", action: () => onDescend(node), primary: true });
    quickActions.push({ label: "FORK LINEAGE", action: () => {} });
    quickActions.push({ label: "ARCHIVE", action: () => {} });
    quickActions.push({ label: "JUMP TRAINING", action: () => {} });
  } else if (node.type === "checkpoint") {
    quickActions.push({ label: "RESTORE", action: () => {} });
    quickActions.push({ label: "COMPARE WITH BEST", action: () => {} });
    quickActions.push({ label: "EXPORT", action: () => {} });
  } else if (node.type === "range") {
    quickActions.push({ label: "RUN EVAL", action: () => {} });
    quickActions.push({ label: "EDIT RANGE", action: () => {} });
  } else if (node.type === "job") {
    quickActions.push({ label: "VIEW LOG", action: () => {} });
    quickActions.push({ label: node.status === "running" ? "PAUSE" : "START", action: () => {} });
  }

  return (
    <div style={{ padding: 10, fontFamily: FONT_MONO, fontSize: 11, overflow: "auto", height: "100%" }}>
      {/* Identity */}
      <div style={{ marginBottom: 14 }}>
        <div style={{
          fontSize: 10, color: NODE_COLORS[node.type] || TEXT_DIM,
          letterSpacing: "0.08em", marginBottom: 4, textTransform: "uppercase",
        }}>
          {node.type}
        </div>
        <div style={{ color: TEXT_PRIMARY, fontSize: 12, fontWeight: 600 }}>{node.name}</div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
          <span style={{
            width: 6, height: 6, borderRadius: "50%",
            background: STATUS_COLORS[node.status] || TEXT_MUTED,
            display: "inline-block",
          }} />
          <span style={{ color: TEXT_DIM, fontSize: 10 }}>{node.status}</span>
        </div>
      </div>

      {/* Neighbors (blast radius) */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ color: ACCENT, fontSize: 10, letterSpacing: "0.06em", marginBottom: 6 }}>
          CONNECTIONS ({neighbors.length})
        </div>
        {neighbors.map((n, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 6,
            padding: "3px 6px", marginBottom: 2,
            background: BG_3, cursor: "pointer",
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: NODE_COLORS[n.node.type] || TEXT_MUTED,
              display: "inline-block", flexShrink: 0,
            }} />
            <span style={{ color: TEXT_PRIMARY, fontSize: 10, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {n.node.name}
            </span>
            <span style={{ color: TEXT_MUTED, fontSize: 9 }}>{n.node.type}</span>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ color: ACCENT, fontSize: 10, letterSpacing: "0.06em", marginBottom: 6 }}>
          ACTIONS
        </div>
        {quickActions.map((a, i) => (
          <button key={i} onClick={a.action} style={{
            display: "block", width: "100%", padding: "6px 8px", marginBottom: 3,
            background: a.primary ? `${ACCENT}22` : BG_3,
            border: `1px solid ${a.primary ? ACCENT : BORDER}`,
            color: a.primary ? ACCENT : TEXT_DIM,
            fontFamily: FONT_MONO, fontSize: 10, cursor: "pointer",
            letterSpacing: "0.04em", textAlign: "left",
            transition: "all 0.15s",
          }}>
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function EcosystemSidebar({ nodes, selected, onSelect, filterType, setFilterType }) {
  const types = ["all", ...new Set(nodes.map((n) => n.type))];
  const filtered = filterType === "all" ? nodes : nodes.filter((n) => n.type === filterType);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <PaneHeader title="ESTATE" subtitle={`${nodes.length} nodes`} />
      {/* Type filter */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: 3, padding: "6px 8px",
        borderBottom: `1px solid ${BORDER}`,
      }}>
        {types.map((t) => (
          <button key={t} onClick={() => setFilterType(t)} style={{
            background: filterType === t ? `${ACCENT}22` : "transparent",
            border: `1px solid ${filterType === t ? ACCENT_DIM : BORDER}`,
            color: filterType === t ? ACCENT : TEXT_MUTED,
            fontFamily: FONT_MONO, fontSize: 8, padding: "2px 5px",
            cursor: "pointer", letterSpacing: "0.04em", textTransform: "uppercase",
          }}>
            {t}
          </button>
        ))}
      </div>
      {/* Node list */}
      <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
        {filtered.map((n) => (
          <div key={n.id} onClick={() => onSelect(n.id)} style={{
            padding: "5px 10px", cursor: "pointer",
            background: selected === n.id ? `${ACCENT}18` : "transparent",
            borderLeft: selected === n.id ? `2px solid ${ACCENT}` : `2px solid transparent`,
            transition: "all 0.1s",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{
                width: 6, height: 6, borderRadius: n.type === "range" ? 0 : "50%",
                background: NODE_COLORS[n.type] || TEXT_MUTED,
                display: "inline-block", flexShrink: 0,
                transform: n.type === "checkpoint" ? "rotate(45deg)" : "none",
              }} />
              <span style={{
                color: selected === n.id ? TEXT_PRIMARY : TEXT_DIM,
                fontSize: 10, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
              }}>
                {n.name}
              </span>
            </div>
            <div style={{ marginLeft: 12, marginTop: 1 }}>
              <span style={{
                fontSize: 8, padding: "0px 4px",
                color: STATUS_COLORS[n.status] || TEXT_MUTED,
              }}>
                {n.status}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================
// WORKBENCH PANE CONTENT
// ============================================================

function ConfigPane({ scenario }) {
  const params = [
    { key: "model", val: "EntropyThoughtDetector v4.5" },
    { key: "sampling_rate", val: "512 Hz" },
    { key: "electrode_type", val: "dry, behind-ear" },
    { key: "entropy_window", val: "256 samples" },
    { key: "baseline_mode", val: "input_only (no feedback)" },
    { key: "batch_size", val: "32" },
    { key: "learning_rate", val: "1.2e-4 (cosine decay)" },
    { key: "dropout", val: "0.15" },
    { key: "comprehension_comp", val: "enabled" },
    { key: "noise_filter", val: "dual_calibration" },
  ];
  return (
    <div style={{ padding: 10, fontFamily: FONT_MONO, fontSize: 11, color: TEXT_PRIMARY, overflow: "auto", height: "100%" }}>
      <div style={{ color: TEXT_DIM, marginBottom: 8 }}>// {scenario?.name || "no scenario"}.yaml</div>
      {params.map((p, i) => (
        <div key={i} style={{ display: "flex", marginBottom: 3 }}>
          <span style={{ color: TEXT_DIM, minWidth: 180 }}>{p.key}:</span>
          <span style={{ color: p.key === "baseline_mode" ? ACCENT : TEXT_PRIMARY }}>{p.val}</span>
        </div>
      ))}
    </div>
  );
}

function MetricsPane() {
  const points = Array.from({ length: 60 }, (_, i) => {
    const base = 90 - (i * 0.3) + Math.sin(i * 0.3) * 4 + (Math.random() - 0.5) * 2;
    return base;
  });
  const maxV = Math.max(...points);
  const minV = Math.min(...points);
  const h = 80, w = 280;
  const pathD = points.map((v, i) => {
    const x = (i / (points.length - 1)) * w;
    const y = h - ((v - minV) / (maxV - minV)) * h;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  const metrics = [
    { label: "accuracy", value: "91.7%", delta: "+0.3%" },
    { label: "loss", value: "0.0338", delta: "-0.0004" },
    { label: "bio_score", value: "0.73", delta: "+0.02" },
    { label: "spectral_slope", value: "-1.42", delta: "" },
  ];

  return (
    <div style={{ padding: 10, fontFamily: FONT_MONO, fontSize: 11, color: TEXT_PRIMARY, overflow: "auto", height: "100%" }}>
      <svg width={w} height={h + 10} style={{ display: "block", marginBottom: 12 }}>
        <path d={pathD} fill="none" stroke={ACCENT} strokeWidth={1.5} />
        <text x={0} y={h + 9} fill={TEXT_DIM} fontSize={9} fontFamily={FONT_MONO}>epoch 0</text>
        <text x={w} y={h + 9} fill={TEXT_DIM} fontSize={9} fontFamily={FONT_MONO} textAnchor="end">epoch 48</text>
      </svg>
      {metrics.map((m, i) => (
        <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, padding: "2px 0" }}>
          <span style={{ color: TEXT_DIM }}>{m.label}</span>
          <span>
            <span style={{ color: TEXT_PRIMARY }}>{m.value}</span>
            {m.delta && <span style={{ color: m.delta.startsWith("+") ? GREEN : RED, marginLeft: 6, fontSize: 10 }}>{m.delta}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

function LineagePane() {
  const snapshots = [
    { id: "cp-048", ts: "14:23:04", score: "0.0338", tag: "latest" },
    { id: "cp-047", ts: "14:23:02", score: "0.0342", tag: "" },
    { id: "cp-038", ts: "14:18:41", score: "0.0289", tag: "best" },
    { id: "cp-001", ts: "13:55:02", score: "0.1247", tag: "init" },
  ];
  return (
    <div style={{ padding: 10, fontFamily: FONT_MONO, fontSize: 11, color: TEXT_PRIMARY, overflow: "auto", height: "100%" }}>
      <div style={{ color: TEXT_DIM, marginBottom: 8 }}>// checkpoint lineage</div>
      {snapshots.map((s, i) => (
        <div key={i} style={{
          display: "flex", alignItems: "center", gap: 8, marginBottom: 6,
          padding: "4px 6px",
          background: s.tag === "best" ? `${ACCENT}11` : "transparent",
          border: s.tag === "best" ? `1px solid ${ACCENT}33` : "1px solid transparent",
        }}>
          <span style={{ color: ACCENT, minWidth: 54 }}>{s.id}</span>
          <span style={{ color: TEXT_DIM, minWidth: 70 }}>{s.ts}</span>
          <span style={{ color: TEXT_PRIMARY, minWidth: 54 }}>{s.score}</span>
          {s.tag && (
            <span style={{
              fontSize: 9, padding: "1px 5px",
              background: s.tag === "best" ? ACCENT_DIM : BG_3,
              color: s.tag === "best" ? "#fff" : TEXT_DIM,
            }}>{s.tag}</span>
          )}
        </div>
      ))}
    </div>
  );
}

function TestPane() {
  const ranges = [
    { name: "rest_baseline", samples: 2048, status: "pass", acc: "94.2%" },
    { name: "subvocal_yes_no", samples: 512, status: "pass", acc: "88.1%" },
    { name: "motor_reach", samples: 1024, status: "warn", acc: "71.3%" },
    { name: "noise_rejection", samples: 4096, status: "pass", acc: "96.8%" },
  ];
  return (
    <div style={{ padding: 10, fontFamily: FONT_MONO, fontSize: 11, color: TEXT_PRIMARY, overflow: "auto", height: "100%" }}>
      <div style={{ color: TEXT_DIM, marginBottom: 8 }}>// eval ranges</div>
      {ranges.map((r, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
          <span style={{
            width: 6, height: 6,
            background: r.status === "pass" ? GREEN : r.status === "warn" ? ACCENT : RED,
            display: "inline-block", flexShrink: 0,
          }} />
          <span style={{ minWidth: 140 }}>{r.name}</span>
          <span style={{ color: TEXT_DIM, minWidth: 50 }}>{r.samples}s</span>
          <span style={{ color: r.status === "warn" ? ACCENT : TEXT_PRIMARY }}>{r.acc}</span>
        </div>
      ))}
    </div>
  );
}

const LOG_LINES = [
  { ts: "14:23:01.442", level: "INFO", msg: "Training epoch 47/200 -- loss: 0.0342 -- lr: 1.2e-4" },
  { ts: "14:23:01.891", level: "INFO", msg: "Validation accuracy: 91.7% (best: 93.1% @ epoch 38)" },
  { ts: "14:23:02.103", level: "WARN", msg: "GPU memory pressure: 7.2 / 8.0 GB -- consider reducing batch size" },
  { ts: "14:23:02.558", level: "INFO", msg: "Checkpoint saved: /runs/entropy_v4.5/cp-047.pt (14.2 MB)" },
  { ts: "14:23:03.011", level: "DATA", msg: "BLE throughput: 3.1 kB/s @ 7.5ms conn interval -- 412 Hz effective" },
  { ts: "14:23:03.442", level: "INFO", msg: "Spectral slope: -1.42 | Kurtosis: 3.81 | Bio score: 0.73" },
  { ts: "14:23:04.103", level: "WARN", msg: "Baseline drift detected: +12 uV over last 60s -- recalibrating" },
  { ts: "14:23:04.558", level: "INFO", msg: "Training epoch 48/200 -- loss: 0.0338 -- lr: 1.18e-4" },
];

// ============================================================
// MAIN APP
// ============================================================

export default function DualPlaneWorkbench() {
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ w: 1200, h: 700 });

  // Plane state: "ecosystem" or "workbench"
  const [plane, setPlane] = useState("ecosystem");
  const [transitionDir, setTransitionDir] = useState(null); // "descend" | "ascend" | null

  // Ecosystem state
  const [ecoSelected, setEcoSelected] = useState(null);
  const [ecoFilterType, setEcoFilterType] = useState("all");
  const [ecoSidebarW, setEcoSidebarW] = useState(200);
  const [ecoInspectorW, setEcoInspectorW] = useState(260);

  // Workbench state
  const [activeMode, setActiveMode] = useState("explore");
  const [activePreset, setActivePreset] = useState("train");
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [sidebarW, setSidebarW] = useState(SIDEBAR_DEFAULT);
  const [inspectorW, setInspectorW] = useState(INSPECTOR_DEFAULT);
  const [bottomH, setBottomH] = useState(BOTTOM_DEFAULT);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [bottomOpen, setBottomOpen] = useState(true);
  const [splits, setSplits] = useState(PRESETS.train.splits);

  // Container measurement
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setContainerSize({ w: r.width, h: r.height });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Transition animation
  useEffect(() => {
    if (transitionDir) {
      const timer = setTimeout(() => setTransitionDir(null), 350);
      return () => clearTimeout(timer);
    }
  }, [transitionDir]);

  // Descend from ecosystem into workbench
  const handleDescend = useCallback((node) => {
    const scenario = {
      id: node.id, name: node.name, status: node.status, jobs: 0,
    };
    setSelectedScenario(scenario);
    setTransitionDir("descend");
    setTimeout(() => setPlane("workbench"), 50);
  }, []);

  // Ascend back to ecosystem
  const handleAscend = useCallback(() => {
    setTransitionDir("ascend");
    setTimeout(() => setPlane("ecosystem"), 50);
  }, []);

  const applyPreset = useCallback((key) => {
    setActivePreset(key);
    setSplits([...PRESETS[key].splits]);
    setBottomH(PRESETS[key].bottomH);
  }, []);

  // Workbench layout math
  const wbMainLeft = ACTIVITY_RAIL_W + sidebarW + 5;
  const wbMainRight = inspectorOpen ? inspectorW + 5 : 0;
  const wbMainW = containerSize.w - wbMainLeft - wbMainRight;
  const wbMainTopH = containerSize.h - STATUS_BAR_H - (bottomOpen ? bottomH + 5 : 0);
  const paneW = (idx) => Math.max(0, (splits[idx] / 100) * wbMainW - (idx < 2 ? 5 : 0));

  // Ecosystem layout math
  const ecoMainW = containerSize.w - ACTIVITY_RAIL_W - ecoSidebarW - 5 - ecoInspectorW - 5;
  const ecoMainH = containerSize.h - STATUS_BAR_H;

  // Workbench split drags
  const onSplit1Drag = useCallback((x) => {
    const rel = x - wbMainLeft;
    const pct = Math.max(10, Math.min(60, (rel / wbMainW) * 100));
    const remaining = 100 - pct;
    const ratio2 = splits[1] / (splits[1] + splits[2]);
    setSplits([pct, remaining * ratio2, remaining * (1 - ratio2)]);
  }, [wbMainLeft, wbMainW, splits]);

  const onSplit2Drag = useCallback((x) => {
    const rel = x - wbMainLeft;
    const pct = Math.max(splits[0] + 10, Math.min(90, (rel / wbMainW) * 100));
    setSplits([splits[0], pct - splits[0], 100 - pct]);
  }, [wbMainLeft, wbMainW, splits]);

  const selectedEcoNode = GRAPH_NODES.find((n) => n.id === ecoSelected);

  const SCENARIOS_WB = GRAPH_NODES.filter((n) => n.type === "scenario").map((n) => ({
    id: n.id, name: n.name, status: n.status, jobs: GRAPH_EDGES.filter((e) => {
      const other = e.from === n.id ? e.to : e.from;
      return GRAPH_NODES.find((nd) => nd.id === other && nd.type === "job");
    }).length,
  }));

  // Transition style
  const getTransitionStyle = () => {
    if (!transitionDir) return { opacity: 1, transform: "scale(1)", transition: "all 0.3s ease" };
    if (transitionDir === "descend") return { opacity: 0.9, transform: "scale(1.02)", transition: "all 0.3s ease" };
    if (transitionDir === "ascend") return { opacity: 0.9, transform: "scale(0.98)", transition: "all 0.3s ease" };
    return {};
  };

  return (
    <div ref={containerRef} style={{
      width: "100%", height: "100vh", background: BG_0, color: TEXT_PRIMARY,
      fontFamily: FONT_MONO, fontSize: 11, display: "flex", flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* ---- MAIN CONTENT ---- */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", ...getTransitionStyle() }}>

        {/* ==== ACTIVITY RAIL (shared shell) ==== */}
        <div style={{
          width: ACTIVITY_RAIL_W, background: BG_1, borderRight: `1px solid ${BORDER}`,
          display: "flex", flexDirection: "column", alignItems: "center",
          paddingTop: 6, gap: 1, flexShrink: 0,
        }}>
          {/* Plane toggle at top */}
          <button
            onClick={() => plane === "ecosystem" ? null : handleAscend()}
            style={{
              width: 38, height: 38, display: "flex", alignItems: "center", justifyContent: "center",
              background: plane === "ecosystem" ? `${ACCENT}22` : "transparent",
              border: "none", cursor: "pointer",
              borderLeft: plane === "ecosystem" ? `2px solid ${ACCENT}` : "2px solid transparent",
              color: plane === "ecosystem" ? ACCENT : TEXT_DIM,
              fontFamily: FONT_MONO, fontSize: 9, fontWeight: 700,
              letterSpacing: "0.04em", transition: "all 0.15s",
            }}
            title="Ecosystem -- radar + mission control"
          >
            ECO
          </button>

          <div style={{ width: 28, height: 1, background: BORDER, margin: "4px 0" }} />

          {/* Workbench modes */}
          {MODES_WORK.map((m) => (
            <button key={m.id} onClick={() => {
              if (plane !== "workbench") {
                if (!selectedScenario && SCENARIOS_WB.length > 0) {
                  setSelectedScenario(SCENARIOS_WB[0]);
                }
                setTransitionDir("descend");
                setTimeout(() => setPlane("workbench"), 50);
              }
              setActiveMode(m.id);
            }} title={m.label} style={{
              width: 38, height: 34, display: "flex", alignItems: "center", justifyContent: "center",
              background: plane === "workbench" && activeMode === m.id ? `${ACCENT}22` : "transparent",
              border: "none", cursor: "pointer",
              borderLeft: plane === "workbench" && activeMode === m.id ? `2px solid ${ACCENT}` : "2px solid transparent",
              color: plane === "workbench" && activeMode === m.id ? ACCENT : TEXT_MUTED,
              fontFamily: FONT_MONO, fontSize: 9, fontWeight: 600,
              letterSpacing: "0.04em", transition: "all 0.15s",
              opacity: plane === "workbench" ? 1 : 0.5,
            }}>
              {m.icon}
            </button>
          ))}

          <div style={{ flex: 1 }} />

          {/* Presets (workbench only) */}
          {plane === "workbench" && (
            <div style={{ borderTop: `1px solid ${BORDER}`, paddingTop: 4, marginBottom: 4, width: "100%" }}>
              {Object.entries(PRESETS).map(([key, p]) => (
                <button key={key} onClick={() => applyPreset(key)} style={{
                  display: "block", width: "100%", padding: "4px 0",
                  background: activePreset === key ? `${ACCENT}15` : "transparent",
                  border: "none", cursor: "pointer",
                  color: activePreset === key ? ACCENT : TEXT_MUTED,
                  fontFamily: FONT_MONO, fontSize: 8, fontWeight: 700,
                  letterSpacing: "0.08em",
                }}>
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ==== ECOSYSTEM PLANE ==== */}
        {plane === "ecosystem" && (
          <>
            {/* Eco Sidebar */}
            <div style={{
              width: ecoSidebarW, background: BG_1, overflow: "hidden",
              display: "flex", flexDirection: "column", flexShrink: 0,
            }}>
              <EcosystemSidebar
                nodes={GRAPH_NODES} selected={ecoSelected}
                onSelect={setEcoSelected}
                filterType={ecoFilterType} setFilterType={setEcoFilterType}
              />
            </div>

            <DragHandle orientation="vertical" onDrag={(x) => {
              setEcoSidebarW(Math.max(160, Math.min(350, x - ACTIVITY_RAIL_W)));
            }} />

            {/* Eco Graph */}
            <div style={{ flex: 1, overflow: "hidden", background: BG_0 }}>
              <EcosystemGraph
                nodes={GRAPH_NODES} edges={GRAPH_EDGES}
                selected={ecoSelected} onSelect={setEcoSelected}
                onDescend={handleDescend}
                width={Math.max(100, ecoMainW)} height={Math.max(100, ecoMainH)}
              />
            </div>

            <DragHandle orientation="vertical" onDrag={(x) => {
              setEcoInspectorW(Math.max(200, Math.min(400, containerSize.w - x)));
            }} />

            {/* Eco Inspector */}
            <div style={{
              width: ecoInspectorW, background: BG_1, overflow: "hidden",
              display: "flex", flexDirection: "column", flexShrink: 0,
            }}>
              <PaneHeader title="INSPECTOR" subtitle={selectedEcoNode ? selectedEcoNode.type : "none"} />
              <div style={{ flex: 1, overflow: "auto" }}>
                <EcosystemInspector
                  node={selectedEcoNode} allNodes={GRAPH_NODES}
                  edges={GRAPH_EDGES} onDescend={handleDescend}
                />
              </div>
            </div>
          </>
        )}

        {/* ==== WORKBENCH PLANE ==== */}
        {plane === "workbench" && (
          <>
            {/* WB Sidebar */}
            <div style={{
              width: sidebarW, background: BG_1, overflow: "hidden",
              display: "flex", flexDirection: "column", flexShrink: 0,
            }}>
              <PaneHeader
                title="SCENARIOS"
                subtitle={`${SCENARIOS_WB.length}`}
                actions={<SmallBtn onClick={handleAscend}>ASCEND</SmallBtn>}
              />
              <div style={{ flex: 1, overflow: "auto", padding: "4px 0" }}>
                {SCENARIOS_WB.map((s) => (
                  <div key={s.id} onClick={() => setSelectedScenario(s)} style={{
                    padding: "6px 12px", cursor: "pointer",
                    background: selectedScenario?.id === s.id ? `${ACCENT}18` : "transparent",
                    borderLeft: selectedScenario?.id === s.id ? `2px solid ${ACCENT}` : "2px solid transparent",
                  }}>
                    <div style={{ color: selectedScenario?.id === s.id ? TEXT_PRIMARY : TEXT_DIM, fontSize: 11 }}>
                      {s.name}
                    </div>
                    <div style={{ display: "flex", gap: 8, marginTop: 2 }}>
                      <span style={{
                        fontSize: 9, padding: "1px 4px",
                        background: s.status === "training" ? `${ACCENT}33` : s.status === "active" ? `${GREEN}33` : BG_3,
                        color: s.status === "training" ? ACCENT : s.status === "active" ? GREEN : TEXT_MUTED,
                      }}>{s.status}</span>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ borderTop: `1px solid ${BORDER}`, padding: "8px 12px", background: BG_2, fontSize: 10 }}>
                <div style={{ color: TEXT_DIM, marginBottom: 4 }}>JOB QUEUE</div>
                <div style={{ color: TEXT_PRIMARY }}>4 total | 1 running | 1 queued</div>
              </div>
            </div>

            <DragHandle orientation="vertical" onDrag={(x) => {
              setSidebarW(Math.max(SIDEBAR_MIN, Math.min(400, x - ACTIVITY_RAIL_W)));
            }} />

            {/* WB Main + Bottom */}
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Main splits */}
              <div style={{ height: wbMainTopH, display: "flex", overflow: "hidden", flexShrink: 0 }}>
                {/* Pane 1 */}
                <div style={{ width: paneW(0), overflow: "hidden", display: "flex", flexDirection: "column", background: BG_1, flexShrink: 0 }}>
                  <PaneHeader title="CONFIG" subtitle={selectedScenario?.name} />
                  <div style={{ flex: 1, overflow: "auto" }}>
                    <ConfigPane scenario={selectedScenario} />
                  </div>
                </div>
                <DragHandle orientation="vertical" onDrag={onSplit1Drag} />
                {/* Pane 2 */}
                <div style={{ width: paneW(1), overflow: "hidden", display: "flex", flexDirection: "column", background: BG_1, flexShrink: 0 }}>
                  <PaneHeader
                    title="METRICS" subtitle="epoch 48"
                    actions={<SmallBtn onClick={() => setInspectorOpen(!inspectorOpen)}>{inspectorOpen ? "HIDE" : "SHOW"} INSPECT</SmallBtn>}
                  />
                  <div style={{ flex: 1, overflow: "auto" }}><MetricsPane /></div>
                </div>
                <DragHandle orientation="vertical" onDrag={onSplit2Drag} />
                {/* Pane 3 */}
                <div style={{ width: paneW(2), overflow: "hidden", display: "flex", flexDirection: "column", background: BG_1, flexShrink: 0 }}>
                  <PaneHeader title={activePreset === "evaluate" ? "EVAL RANGES" : "LINEAGE"} />
                  <div style={{ flex: 1, overflow: "auto" }}>
                    {activePreset === "evaluate" ? <TestPane /> : <LineagePane />}
                  </div>
                </div>
              </div>

              {bottomOpen && <DragHandle orientation="horizontal" onDrag={(y) => {
                setBottomH(Math.max(BOTTOM_MIN, Math.min(400, containerSize.h - STATUS_BAR_H - y)));
              }} />}

              {bottomOpen && (
                <div style={{ height: bottomH, background: BG_1, overflow: "hidden", display: "flex", flexDirection: "column", flexShrink: 0 }}>
                  <PaneHeader title="OUTPUT" subtitle="training log" actions={<SmallBtn onClick={() => setBottomOpen(false)}>COLLAPSE</SmallBtn>} />
                  <div style={{ flex: 1, overflow: "auto", padding: "4px 10px" }}>
                    {LOG_LINES.map((l, i) => (
                      <div key={i} style={{ display: "flex", gap: 8, marginBottom: 1, lineHeight: "18px" }}>
                        <span style={{ color: TEXT_MUTED, minWidth: 90, flexShrink: 0 }}>{l.ts}</span>
                        <span style={{ minWidth: 36, flexShrink: 0, color: l.level === "WARN" ? ACCENT : l.level === "DATA" ? BLUE : TEXT_DIM }}>{l.level}</span>
                        <span style={{ color: TEXT_PRIMARY }}>{l.msg}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {inspectorOpen && <DragHandle orientation="vertical" onDrag={(x) => {
              setInspectorW(Math.max(INSPECTOR_MIN, Math.min(450, containerSize.w - x)));
            }} />}

            {inspectorOpen && (
              <div style={{
                width: inspectorW, background: BG_1, overflow: "hidden",
                display: "flex", flexDirection: "column", flexShrink: 0,
              }}>
                <PaneHeader title="INSPECTOR" subtitle={selectedScenario?.name} />
                <div style={{ flex: 1, overflow: "auto", padding: 10, fontSize: 11 }}>
                  <div style={{ color: TEXT_DIM, marginBottom: 10 }}>// entity detail</div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: ACCENT, fontSize: 10, marginBottom: 4, letterSpacing: "0.06em" }}>SCENARIO</div>
                    <div style={{ color: TEXT_PRIMARY }}>{selectedScenario?.name}</div>
                    <div style={{ color: TEXT_DIM, fontSize: 10, marginTop: 2 }}>status: {selectedScenario?.status}</div>
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: ACCENT, fontSize: 10, marginBottom: 4, letterSpacing: "0.06em" }}>HARDWARE</div>
                    <div style={{ color: TEXT_DIM }}>MCU: nRF52840 (XIAO)</div>
                    <div style={{ color: TEXT_DIM }}>Sensor: VL6180 ToF</div>
                    <div style={{ color: TEXT_DIM }}>Electrodes: dry, BTE</div>
                    <div style={{ color: TEXT_DIM }}>BLE interval: 7.5ms</div>
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: ACCENT, fontSize: 10, marginBottom: 4, letterSpacing: "0.06em" }}>LAST RUN</div>
                    <div style={{ color: TEXT_DIM }}>duration: 12m 38s</div>
                    <div style={{ color: TEXT_DIM }}>epochs: 48/200</div>
                    <div style={{ color: TEXT_DIM }}>best_loss: 0.0289</div>
                    <div style={{ color: TEXT_DIM }}>gpu_peak: 7.2 GB</div>
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: ACCENT, fontSize: 10, marginBottom: 6, letterSpacing: "0.06em" }}>COMPARE</div>
                    <button style={{
                      width: "100%", padding: "6px 0", background: BG_3,
                      border: `1px solid ${BORDER}`, color: TEXT_DIM,
                      fontFamily: FONT_MONO, fontSize: 10, cursor: "pointer",
                    }}>+ OPEN RUN B</button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* ==== STATUS BAR (shared, persistent) ==== */}
      <div style={{
        height: STATUS_BAR_H, background: BG_2, borderTop: `1px solid ${BORDER}`,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 10px", fontSize: 10, flexShrink: 0,
      }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <span style={{ color: GREEN }}>{">"} CONNECTED</span>
          <span style={{
            color: plane === "ecosystem" ? CYAN : ACCENT,
            fontWeight: 600, letterSpacing: "0.06em",
          }}>
            {plane === "ecosystem" ? "ECOSYSTEM" : "WORKBENCH"}
          </span>
          {plane === "workbench" && selectedScenario && (
            <span style={{ color: TEXT_DIM }}>focus: {selectedScenario.name}</span>
          )}
          {plane === "workbench" && (
            <span style={{ color: TEXT_DIM }}>preset: {PRESETS[activePreset].label}</span>
          )}
          {plane === "workbench" && !bottomOpen && (
            <SmallBtn onClick={() => setBottomOpen(true)}>SHOW LOG</SmallBtn>
          )}
        </div>
        <div style={{ display: "flex", gap: 16 }}>
          <span style={{ color: TEXT_DIM }}>GPU: 7.2/8.0 GB</span>
          <span style={{ color: TEXT_DIM }}>BLE: 412 Hz</span>
          <span style={{ color: ACCENT }}>epoch 48/200</span>
        </div>
      </div>
    </div>
  );
}