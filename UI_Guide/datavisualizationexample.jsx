import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, LineChart, Line, PieChart, Pie,
  CartesianGrid, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, AreaChart, Area
} from "recharts";
import Papa from "papaparse";

// ============================================================
// NEO-SWISS DESIGN TOKENS
// ============================================================
const T = {
  bg0: "#0A0A0B",
  bg1: "#111113",
  bg2: "#1A1A1E",
  bg3: "#222228",
  border: "#2A2A32",
  borderLight: "#3A3A44",
  text0: "#E8E8EC",
  text1: "#A8A8B4",
  text2: "#6A6A78",
  green: "#5BE872",
  greenDim: "rgba(91,232,114,0.15)",
  red: "#FF4D4D",
  redDim: "rgba(255,77,77,0.12)",
  amber: "#FFB547",
  amberDim: "rgba(255,181,71,0.12)",
  blue: "#4D9FFF",
  blueDim: "rgba(77,159,255,0.12)",
  purple: "#A78BFA",
  cyan: "#22D3EE",
  chamfer: "0 0 0 12px",
  mono: "'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace",
  sans: "'DM Sans', 'Helvetica Neue', system-ui, sans-serif",
};

const CHART_COLORS = [T.green, T.blue, T.amber, T.purple, T.cyan, T.red, "#FF6B9D", "#50E3C2"];

// ============================================================
// SIMULATED FINGERPRINT ENGINE
// (In production this calls the Python backend)
// ============================================================
function investigateData(raw, columns) {
  const n = raw.length;
  const numCols = columns.filter(c => {
    const sample = raw.slice(0, 20).map(r => parseFloat(r[c]));
    return sample.filter(v => !isNaN(v)).length > sample.length * 0.6;
  });
  const catCols = columns.filter(c => !numCols.includes(c));

  // Distributions
  const distributions = numCols.slice(0, 30).map(col => {
    const vals = raw.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
    vals.sort((a, b) => a - b);
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
    const missing = raw.length - vals.length;
    return {
      name: col, dtype: "float64", count: raw.length, missing,
      missing_pct: Math.round(missing / raw.length * 10000) / 100,
      unique: new Set(vals).size,
      mean: Math.round(mean * 1000) / 1000,
      std: Math.round(std * 1000) / 1000,
      min: vals[0],
      q25: vals[Math.floor(vals.length * 0.25)],
      median: vals[Math.floor(vals.length * 0.5)],
      q75: vals[Math.floor(vals.length * 0.75)],
      max: vals[vals.length - 1],
      skewness: Math.round(vals.reduce((s, v) => s + ((v - mean) / (std || 1)) ** 3, 0) / vals.length * 1000) / 1000,
      kurtosis: Math.round((vals.reduce((s, v) => s + ((v - mean) / (std || 1)) ** 4, 0) / vals.length - 3) * 1000) / 1000,
    };
  });

  // Categorical distributions
  const catDistributions = catCols.slice(0, 10).map(col => {
    const counts = {};
    let missing = 0;
    raw.forEach(r => {
      const v = r[col];
      if (v === null || v === undefined || v === "") { missing++; return; }
      counts[v] = (counts[v] || 0) + 1;
    });
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return {
      name: col, dtype: "object", count: raw.length, missing,
      missing_pct: Math.round(missing / raw.length * 10000) / 100,
      unique: sorted.length,
      top_values: sorted.slice(0, 10).map(([value, count]) => ({
        value, count, pct: Math.round(count / (raw.length - missing) * 10000) / 100
      })),
    };
  });

  // Correlation (numeric pairs)
  const corrPairs = [];
  for (let i = 0; i < Math.min(numCols.length, 15); i++) {
    for (let j = i + 1; j < Math.min(numCols.length, 15); j++) {
      const vals1 = [], vals2 = [];
      raw.forEach(r => {
        const a = parseFloat(r[numCols[i]]), b = parseFloat(r[numCols[j]]);
        if (!isNaN(a) && !isNaN(b)) { vals1.push(a); vals2.push(b); }
      });
      if (vals1.length < 10) continue;
      const m1 = vals1.reduce((s, v) => s + v, 0) / vals1.length;
      const m2 = vals2.reduce((s, v) => s + v, 0) / vals2.length;
      const s1 = Math.sqrt(vals1.reduce((s, v) => s + (v - m1) ** 2, 0) / vals1.length);
      const s2 = Math.sqrt(vals2.reduce((s, v) => s + (v - m2) ** 2, 0) / vals2.length);
      if (s1 === 0 || s2 === 0) continue;
      const r_val = vals1.reduce((s, v, k) => s + (v - m1) * (vals2[k] - m2), 0) / (vals1.length * s1 * s2);
      if (Math.abs(r_val) > 0.3) {
        corrPairs.push({ feature_1: numCols[i], feature_2: numCols[j], correlation: Math.round(r_val * 10000) / 10000 });
      }
    }
  }
  corrPairs.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

  // Quality issues
  const issues = [];
  distributions.forEach(d => {
    if (d.missing_pct > 50) issues.push({ severity: "critical", category: "missing", message: `'${d.name}' has ${d.missing_pct}% missing` });
    else if (d.missing_pct > 10) issues.push({ severity: "warning", category: "missing", message: `'${d.name}' has ${d.missing_pct}% missing` });
  });

  const dupes = raw.length - new Set(raw.map(r => JSON.stringify(r))).size;
  if (dupes > 0) issues.push({ severity: dupes / raw.length > 0.05 ? "critical" : "warning", category: "duplicate", message: `${dupes} duplicate rows (${(dupes / raw.length * 100).toFixed(1)}%)` });

  corrPairs.filter(p => Math.abs(p.correlation) > 0.95).forEach(p => {
    issues.push({ severity: "warning", category: "leakage", message: `'${p.feature_1}' and '${p.feature_2}' have r=${p.correlation} -- possible leakage` });
  });

  // Scatter embedding (PCA-like: just use first two numeric cols)
  let embedding = null;
  if (numCols.length >= 2) {
    embedding = raw.slice(0, 2000).map(r => ({
      x: parseFloat(r[numCols[0]]) || 0,
      y: parseFloat(r[numCols[1]]) || 0,
    }));
  }

  // Quality score
  let qScore = 1.0;
  issues.forEach(i => { qScore -= i.severity === "critical" ? 0.15 : i.severity === "warning" ? 0.05 : 0.01; });
  qScore = Math.max(0, Math.min(1, qScore));

  return {
    archetype: "TABULAR",
    n_samples: n,
    n_features: columns.length,
    numeric_features: numCols.length,
    categorical_features: catCols.length,
    distributions,
    catDistributions,
    corrPairs,
    issues,
    quality_score: Math.round(qScore * 1000) / 1000,
    embedding,
    columns,
    numCols,
    catCols,
    raw: raw.slice(0, 5000),
  };
}

// ============================================================
// HISTOGRAM BUILDER
// ============================================================
function buildHistogram(values, bins = 30) {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / bins;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
    counts[idx]++;
  });
  return counts.map((count, i) => ({
    bin: Math.round((min + i * binWidth) * 100) / 100,
    count,
    label: `${(min + i * binWidth).toFixed(1)}`,
  }));
}

// ============================================================
// COMPONENTS
// ============================================================

// -- Status Bar --
function StatusBar({ fingerprint, activeView }) {
  if (!fingerprint) return null;
  const qColor = fingerprint.quality_score > 0.8 ? T.green : fingerprint.quality_score > 0.5 ? T.amber : T.red;
  return (
    <div style={{
      height: 28, background: T.bg0, borderTop: `1px solid ${T.border}`,
      display: "flex", alignItems: "center", padding: "0 12px", gap: 16,
      fontFamily: T.mono, fontSize: 11, color: T.text2, flexShrink: 0,
    }}>
      <span style={{ color: T.green }}>DATASET INVESTIGATOR</span>
      <span>{fingerprint.archetype}</span>
      <span>{fingerprint.n_samples.toLocaleString()} rows x {fingerprint.n_features} cols</span>
      <span>NUM: {fingerprint.numeric_features} | CAT: {fingerprint.categorical_features}</span>
      <span style={{ color: qColor }}>HEALTH: {(fingerprint.quality_score * 100).toFixed(1)}%</span>
      <span>{fingerprint.issues.length} issues</span>
      <span style={{ marginLeft: "auto", color: T.text2 }}>{activeView.toUpperCase()}</span>
    </div>
  );
}

// -- Quality Badge --
function QualityBadge({ score }) {
  const pct = Math.round(score * 100);
  const color = score > 0.8 ? T.green : score > 0.5 ? T.amber : T.red;
  const bg = score > 0.8 ? T.greenDim : score > 0.5 ? T.amberDim : T.redDim;
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "3px 10px", background: bg, borderRadius: `4px 4px 4px ${12}px`,
      fontFamily: T.mono, fontSize: 13, fontWeight: 600, color,
    }}>
      <svg width="10" height="10"><circle cx="5" cy="5" r="4" fill={color} opacity={0.8} /></svg>
      {pct}%
    </div>
  );
}

// -- Issue List --
function IssueList({ issues }) {
  const icons = { critical: "\u2716", warning: "\u26A0", info: "\u2139" };
  const colors = { critical: T.red, warning: T.amber, info: T.blue };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {issues.map((issue, i) => (
        <div key={i} style={{
          display: "flex", alignItems: "flex-start", gap: 8,
          padding: "6px 8px", background: T.bg2, borderRadius: `4px 4px 4px ${12}px`,
          borderLeft: `2px solid ${colors[issue.severity]}`,
          fontFamily: T.mono, fontSize: 11, color: T.text1, lineHeight: 1.4,
        }}>
          <span style={{ color: colors[issue.severity], flexShrink: 0 }}>{icons[issue.severity]}</span>
          <span>{issue.message}</span>
        </div>
      ))}
    </div>
  );
}

// ============================================================
// VISUALIZATION TEMPLATES
// ============================================================

function DistributionTemplate({ fingerprint, selectedFeature }) {
  const dist = fingerprint.distributions.find(d => d.name === selectedFeature) || fingerprint.distributions[0];
  if (!dist) return <div style={{ color: T.text2, padding: 20, fontFamily: T.mono, fontSize: 12 }}>No numeric features found</div>;

  const vals = fingerprint.raw.map(r => parseFloat(r[dist.name])).filter(v => !isNaN(v));
  const histData = buildHistogram(vals, 40);

  return (
    <div style={{ height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ fontFamily: T.mono, fontSize: 12, color: T.text0 }}>
          <span style={{ background: T.greenDim, padding: "2px 6px", borderRadius: `3px 3px 3px 8px` }}>{dist.name}</span>
        </div>
        <div style={{ fontFamily: T.mono, fontSize: 10, color: T.text2, display: "flex", gap: 12 }}>
          <span>m={dist.mean}</span>
          <span>s={dist.std}</span>
          <span>sk={dist.skewness}</span>
          <span>ku={dist.kurtosis}</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
          <XAxis dataKey="label" stroke={T.text2} fontSize={9} fontFamily={T.mono} angle={-45} textAnchor="end" height={40} interval="preserveStartEnd" />
          <YAxis stroke={T.text2} fontSize={9} fontFamily={T.mono} />
          <Tooltip contentStyle={{ background: T.bg1, border: `1px solid ${T.border}`, fontFamily: T.mono, fontSize: 11, color: T.text0, borderRadius: `4px 4px 4px 12px` }} />
          <Bar dataKey="count" fill={T.green} fillOpacity={0.7} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CorrelationTemplate({ fingerprint }) {
  const pairs = fingerprint.corrPairs.slice(0, 20);
  if (!pairs.length) return <div style={{ color: T.text2, padding: 20, fontFamily: T.mono, fontSize: 12 }}>No significant correlations found</div>;
  const data = pairs.map(p => ({
    pair: `${p.feature_1.slice(0, 8)}/${p.feature_2.slice(0, 8)}`,
    correlation: p.correlation,
    absCorr: Math.abs(p.correlation),
  }));
  return (
    <div style={{ height: "100%" }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
          <XAxis type="number" domain={[-1, 1]} stroke={T.text2} fontSize={9} fontFamily={T.mono} />
          <YAxis type="category" dataKey="pair" stroke={T.text2} fontSize={9} fontFamily={T.mono} width={76} />
          <Tooltip contentStyle={{ background: T.bg1, border: `1px solid ${T.border}`, fontFamily: T.mono, fontSize: 11, color: T.text0, borderRadius: `4px 4px 4px 12px` }} />
          <Bar dataKey="correlation" radius={[0, 2, 2, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.correlation > 0 ? T.green : T.red} fillOpacity={0.7} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ScatterTemplate({ fingerprint }) {
  if (!fingerprint.embedding || !fingerprint.numCols || fingerprint.numCols.length < 2) {
    return <div style={{ color: T.text2, padding: 20, fontFamily: T.mono, fontSize: 12 }}>Need at least 2 numeric columns for scatter</div>;
  }
  return (
    <div style={{ height: "100%" }}>
      <div style={{ fontFamily: T.mono, fontSize: 10, color: T.text2, marginBottom: 4 }}>
        {fingerprint.numCols[0]} vs {fingerprint.numCols[1]} (first 2000 samples)
      </div>
      <ResponsiveContainer width="100%" height="92%">
        <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
          <XAxis dataKey="x" stroke={T.text2} fontSize={9} fontFamily={T.mono} name={fingerprint.numCols[0]} />
          <YAxis dataKey="y" stroke={T.text2} fontSize={9} fontFamily={T.mono} name={fingerprint.numCols[1]} />
          <Tooltip contentStyle={{ background: T.bg1, border: `1px solid ${T.border}`, fontFamily: T.mono, fontSize: 11, color: T.text0, borderRadius: `4px 4px 4px 12px` }} />
          <Scatter data={fingerprint.embedding} fill={T.green} fillOpacity={0.4} r={2} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function CategoricalTemplate({ fingerprint }) {
  const dist = fingerprint.catDistributions[0];
  if (!dist || !dist.top_values) return <div style={{ color: T.text2, padding: 20, fontFamily: T.mono, fontSize: 12 }}>No categorical features found</div>;
  return (
    <div style={{ height: "100%" }}>
      <div style={{ fontFamily: T.mono, fontSize: 12, color: T.text0, marginBottom: 8 }}>
        <span style={{ background: T.blueDim, padding: "2px 6px", borderRadius: `3px 3px 3px 8px` }}>{dist.name}</span>
        <span style={{ color: T.text2, marginLeft: 8, fontSize: 10 }}>{dist.unique} unique</span>
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={dist.top_values} margin={{ top: 4, right: 8, bottom: 30, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
          <XAxis dataKey="value" stroke={T.text2} fontSize={9} fontFamily={T.mono} angle={-45} textAnchor="end" height={50} />
          <YAxis stroke={T.text2} fontSize={9} fontFamily={T.mono} />
          <Tooltip contentStyle={{ background: T.bg1, border: `1px solid ${T.border}`, fontFamily: T.mono, fontSize: 11, color: T.text0, borderRadius: `4px 4px 4px 12px` }} />
          <Bar dataKey="count" fill={T.blue} fillOpacity={0.7} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function MissingDataTemplate({ fingerprint }) {
  const data = [...fingerprint.distributions, ...fingerprint.catDistributions]
    .filter(d => d.missing_pct > 0)
    .sort((a, b) => b.missing_pct - a.missing_pct)
    .slice(0, 20)
    .map(d => ({ name: d.name.slice(0, 12), pct: d.missing_pct }));
  if (!data.length) return <div style={{ color: T.green, padding: 20, fontFamily: T.mono, fontSize: 12 }}>No missing data detected</div>;
  return (
    <div style={{ height: "100%" }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
          <XAxis type="number" domain={[0, 100]} stroke={T.text2} fontSize={9} fontFamily={T.mono} unit="%" />
          <YAxis type="category" dataKey="name" stroke={T.text2} fontSize={9} fontFamily={T.mono} width={76} />
          <Tooltip contentStyle={{ background: T.bg1, border: `1px solid ${T.border}`, fontFamily: T.mono, fontSize: 11, color: T.text0, borderRadius: `4px 4px 4px 12px` }} />
          <Bar dataKey="pct" radius={[0, 2, 2, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.pct > 50 ? T.red : d.pct > 10 ? T.amber : T.blue} fillOpacity={0.7} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function SummaryStatsTemplate({ fingerprint }) {
  const stats = fingerprint.distributions.slice(0, 15);
  if (!stats.length) return null;
  return (
    <div style={{ height: "100%", overflow: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: T.mono, fontSize: 10 }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${T.border}`, color: T.text2 }}>
            {["Feature", "Mean", "Std", "Min", "Median", "Max", "Miss%", "Skew"].map(h => (
              <th key={h} style={{ padding: "6px 8px", textAlign: "left", position: "sticky", top: 0, background: T.bg2 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stats.map((d, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${T.bg3}`, color: T.text1 }}>
              <td style={{ padding: "4px 8px", color: T.text0 }}>{d.name}</td>
              <td style={{ padding: "4px 8px" }}>{d.mean}</td>
              <td style={{ padding: "4px 8px" }}>{d.std}</td>
              <td style={{ padding: "4px 8px" }}>{d.min}</td>
              <td style={{ padding: "4px 8px" }}>{d.median}</td>
              <td style={{ padding: "4px 8px" }}>{d.max}</td>
              <td style={{ padding: "4px 8px", color: d.missing_pct > 10 ? T.amber : T.text2 }}>{d.missing_pct}%</td>
              <td style={{ padding: "4px 8px" }}>{d.skewness}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================
// TEMPLATE REGISTRY
// ============================================================
const TEMPLATES = [
  { id: "dist", label: "DISTRIBUTION", icon: "\u2581\u2582\u2583\u2585\u2587", component: DistributionTemplate },
  { id: "corr", label: "CORRELATION", icon: "\u25C9", component: CorrelationTemplate },
  { id: "scatter", label: "SCATTER", icon: "\u25CC", component: ScatterTemplate },
  { id: "cat", label: "CATEGORICAL", icon: "\u2588", component: CategoricalTemplate },
  { id: "missing", label: "MISSING DATA", icon: "\u2592", component: MissingDataTemplate },
  { id: "stats", label: "SUMMARY TABLE", icon: "\u2261", component: SummaryStatsTemplate },
];

// ============================================================
// CELL SPACE (Code Editor + Runner)
// ============================================================
function CellSpace({ fingerprint, isOpen, onClose }) {
  const [code, setCode] = useState(
`// Custom visualization cell
// Available: fingerprint (full investigation result)
// Return JSX or a data array for charting

// Example: top 5 features by standard deviation
const ranked = fingerprint.distributions
  .filter(d => d.std !== null)
  .sort((a, b) => b.std - a.std)
  .slice(0, 8);

return ranked.map(d => ({
  name: d.name,
  std: d.std,
  mean: d.mean,
}));`);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState(null);
  const [outputType, setOutputType] = useState("table");
  const textareaRef = useRef(null);

  const runCell = useCallback(() => {
    setError(null);
    setOutput(null);
    try {
      const fn = new Function("fingerprint", code);
      const result = fn(fingerprint);
      setOutput(result);
      if (Array.isArray(result) && result.length > 0 && typeof result[0] === "object") {
        setOutputType("chart");
      } else {
        setOutputType("raw");
      }
    } catch (e) {
      setError(e.message);
    }
  }, [code, fingerprint]);

  const handleKeyDown = useCallback((e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      runCell();
    }
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = textareaRef.current;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      setCode(code.substring(0, start) + "  " + code.substring(end));
      setTimeout(() => { ta.selectionStart = ta.selectionEnd = start + 2; }, 0);
    }
  }, [code, runCell]);

  if (!isOpen) return null;

  const chartKeys = output && Array.isArray(output) && output[0] ? Object.keys(output[0]).filter(k => typeof output[0][k] === "number") : [];
  const labelKey = output && Array.isArray(output) && output[0] ? Object.keys(output[0]).find(k => typeof output[0][k] === "string") : null;

  return (
    <div style={{
      width: 420, borderLeft: `1px solid ${T.border}`, background: T.bg1,
      display: "flex", flexDirection: "column", flexShrink: 0, overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        height: 36, display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 12px", borderBottom: `1px solid ${T.border}`, flexShrink: 0,
      }}>
        <div style={{ fontFamily: T.mono, fontSize: 11, color: T.green, display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 6, height: 6, background: T.green, borderRadius: 1, display: "inline-block" }} />
          CELL SPACE
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={runCell} style={{
            background: T.greenDim, border: `1px solid ${T.green}40`, color: T.green,
            fontFamily: T.mono, fontSize: 10, padding: "3px 10px", cursor: "pointer",
            borderRadius: `3px 3px 3px 8px`, fontWeight: 600,
          }}>
            RUN [Cmd+Enter]
          </button>
          <button onClick={onClose} style={{
            background: "none", border: "none", color: T.text2, cursor: "pointer",
            fontFamily: T.mono, fontSize: 14, padding: "0 4px",
          }}>
            x
          </button>
        </div>
      </div>

      {/* Editor */}
      <div style={{ flex: "1 1 50%", overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <div style={{ padding: "4px 12px 0", fontFamily: T.mono, fontSize: 9, color: T.text2, flexShrink: 0 }}>
          // fingerprint.distributions, .corrPairs, .issues, .raw, .numCols, .catCols
        </div>
        <textarea
          ref={textareaRef}
          value={code}
          onChange={e => setCode(e.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          style={{
            flex: 1, resize: "none", background: T.bg0, color: T.text0,
            fontFamily: T.mono, fontSize: 12, lineHeight: 1.6,
            border: "none", outline: "none", padding: 12,
            borderBottom: `1px solid ${T.border}`,
          }}
        />
      </div>

      {/* Output */}
      <div style={{ flex: "1 1 50%", overflow: "auto", background: T.bg2 }}>
        <div style={{
          padding: "4px 12px", borderBottom: `1px solid ${T.border}`,
          fontFamily: T.mono, fontSize: 9, color: T.text2, flexShrink: 0,
        }}>
          OUTPUT
        </div>
        {error && (
          <div style={{ padding: 12, fontFamily: T.mono, fontSize: 11, color: T.red, lineHeight: 1.5 }}>
            {error}
          </div>
        )}
        {output && !error && outputType === "chart" && (
          <div style={{ padding: 8, height: "calc(100% - 28px)" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={output} margin={{ top: 8, right: 8, bottom: 30, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
                {labelKey && <XAxis dataKey={labelKey} stroke={T.text2} fontSize={9} fontFamily={T.mono} angle={-30} textAnchor="end" height={40} />}
                <YAxis stroke={T.text2} fontSize={9} fontFamily={T.mono} />
                <Tooltip contentStyle={{ background: T.bg1, border: `1px solid ${T.border}`, fontFamily: T.mono, fontSize: 11, color: T.text0, borderRadius: `4px 4px 4px 12px` }} />
                {chartKeys.map((key, i) => (
                  <Bar key={key} dataKey={key} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.7} radius={[2, 2, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {output && !error && outputType === "raw" && (
          <pre style={{ padding: 12, fontFamily: T.mono, fontSize: 11, color: T.text1, whiteSpace: "pre-wrap", lineHeight: 1.5, margin: 0 }}>
            {typeof output === "string" ? output : JSON.stringify(output, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

// ============================================================
// IMPORT SCREEN
// ============================================================
function ImportScreen({ onDataLoaded }) {
  const [dragOver, setDragOver] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState(null);
  const fileRef = useRef(null);

  const handleFile = useCallback((file) => {
    setParsing(true);
    setParseError(null);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (results) => {
        if (results.errors.length > 5) {
          setParseError(`Parse errors: ${results.errors.slice(0, 3).map(e => e.message).join("; ")}`);
          setParsing(false);
          return;
        }
        if (results.data.length < 2) {
          setParseError("File contains fewer than 2 data rows");
          setParsing(false);
          return;
        }
        const columns = results.meta.fields || Object.keys(results.data[0]);
        const fingerprint = investigateData(results.data, columns);
        onDataLoaded(fingerprint, file.name);
        setParsing(false);
      },
      error: (err) => {
        setParseError(err.message);
        setParsing(false);
      },
    });
  }, [onDataLoaded]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      background: T.bg0, padding: 40,
    }}>
      {/* Title block */}
      <div style={{ marginBottom: 48, textAlign: "center" }}>
        <div style={{
          fontFamily: T.mono, fontSize: 10, color: T.green, letterSpacing: 3, marginBottom: 12,
          display: "inline-block", background: T.greenDim, padding: "4px 12px",
          borderRadius: `3px 3px 3px 12px`,
        }}>
          DATASET INVESTIGATOR
        </div>
        <h1 style={{ fontFamily: T.sans, fontSize: 32, color: T.text0, fontWeight: 300, margin: 0, letterSpacing: -0.5 }}>
          Import your data
        </h1>
        <p style={{ fontFamily: T.mono, fontSize: 12, color: T.text2, marginTop: 8 }}>
          CSV files supported. Drag and drop or click to browse.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => fileRef.current.click()}
        style={{
          width: 480, height: 240, border: `2px dashed ${dragOver ? T.green : T.border}`,
          borderRadius: `8px 8px 8px 24px`,
          background: dragOver ? T.greenDim : T.bg1,
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          cursor: "pointer", transition: "all 0.2s ease",
        }}
      >
        {parsing ? (
          <div style={{ fontFamily: T.mono, fontSize: 13, color: T.green }}>
            PARSING...
          </div>
        ) : (
          <>
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none" style={{ marginBottom: 16, opacity: 0.6 }}>
              <path d="M8 36V40C8 41.1 9.3 42 11 42H37C38.7 42 40 41.1 40 40V36" stroke={T.text2} strokeWidth="2" />
              <path d="M24 6V30M24 6L16 14M24 6L32 14" stroke={dragOver ? T.green : T.text2} strokeWidth="2" />
            </svg>
            <div style={{ fontFamily: T.mono, fontSize: 12, color: T.text1, marginBottom: 4 }}>
              Drop CSV here or click to browse
            </div>
            <div style={{ fontFamily: T.mono, fontSize: 10, color: T.text2 }}>
              Auto-detects archetype, runs full investigation pipeline
            </div>
          </>
        )}
      </div>

      <input ref={fileRef} type="file" accept=".csv,.tsv" style={{ display: "none" }}
        onChange={e => { if (e.target.files[0]) handleFile(e.target.files[0]); }} />

      {parseError && (
        <div style={{
          marginTop: 16, padding: "8px 16px", background: T.redDim,
          border: `1px solid ${T.red}40`, borderRadius: `4px 4px 4px 12px`,
          fontFamily: T.mono, fontSize: 11, color: T.red, maxWidth: 480,
        }}>
          {parseError}
        </div>
      )}

      {/* Demo data */}
      <button
        onClick={() => {
          const rows = [];
          for (let i = 0; i < 1500; i++) {
            rows.push({
              age: (30 + Math.random() * 40).toFixed(1),
              income: Math.exp(9.5 + Math.random() * 1.5).toFixed(0),
              score: (400 + Math.random() * 200).toFixed(1),
              region: ["North", "South", "East", "West"][Math.floor(Math.random() * 4)],
              churned: Math.random() > 0.75 ? "yes" : "no",
              signup_days: Math.floor(Math.random() * 1000).toString(),
              sessions: Math.floor(Math.random() * 200).toString(),
              nps_score: (Math.random() * 10).toFixed(1),
            });
          }
          if (Math.random() > 0.5) { for (let i = 0; i < 80; i++) rows[i].income = ""; }
          const cols = Object.keys(rows[0]);
          const fp = investigateData(rows, cols);
          onDataLoaded(fp, "demo_dataset.csv");
        }}
        style={{
          marginTop: 24, background: "none", border: `1px solid ${T.border}`,
          color: T.text2, fontFamily: T.mono, fontSize: 11, padding: "6px 16px",
          cursor: "pointer", borderRadius: `3px 3px 3px 8px`,
        }}
      >
        or load demo dataset
      </button>
    </div>
  );
}

// ============================================================
// INVESTIGATION VIEW
// ============================================================
function InvestigationView({ fingerprint, fileName }) {
  const [activeTemplates, setActiveTemplates] = useState(["dist", "corr", "scatter"]);
  const [cellOpen, setCellOpen] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState(
    fingerprint.distributions[0]?.name || ""
  );

  const toggleTemplate = (id) => {
    setActiveTemplates(prev =>
      prev.includes(id) ? prev.filter(t => t !== id) : [...prev, id]
    );
  };

  return (
    <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
      {/* LEFT SIDEBAR: Fingerprint Summary */}
      <div style={{
        width: 260, borderRight: `1px solid ${T.border}`, background: T.bg1,
        display: "flex", flexDirection: "column", overflow: "hidden", flexShrink: 0,
      }}>
        {/* File name header */}
        <div style={{
          padding: "10px 12px", borderBottom: `1px solid ${T.border}`,
          fontFamily: T.mono, fontSize: 11, color: T.text0,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{ width: 6, height: 6, background: T.green, borderRadius: 1, display: "inline-block" }} />
          {fileName}
        </div>

        <div style={{ flex: 1, overflow: "auto", padding: 12 }}>
          {/* Quality Score */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontFamily: T.mono, fontSize: 9, color: T.text2, letterSpacing: 1, marginBottom: 6 }}>HEALTH</div>
            <QualityBadge score={fingerprint.quality_score} />
          </div>

          {/* Shape */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontFamily: T.mono, fontSize: 9, color: T.text2, letterSpacing: 1, marginBottom: 6 }}>SHAPE</div>
            <div style={{ fontFamily: T.mono, fontSize: 12, color: T.text0 }}>
              {fingerprint.n_samples.toLocaleString()} x {fingerprint.n_features}
            </div>
            <div style={{ fontFamily: T.mono, fontSize: 10, color: T.text2, marginTop: 2 }}>
              {fingerprint.numeric_features} numeric / {fingerprint.categorical_features} categorical
            </div>
          </div>

          {/* Issues */}
          {fingerprint.issues.length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontFamily: T.mono, fontSize: 9, color: T.text2, letterSpacing: 1, marginBottom: 6 }}>
                ISSUES ({fingerprint.issues.length})
              </div>
              <IssueList issues={fingerprint.issues} />
            </div>
          )}

          {/* Feature selector */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontFamily: T.mono, fontSize: 9, color: T.text2, letterSpacing: 1, marginBottom: 6 }}>FEATURE FOCUS</div>
            <select
              value={selectedFeature}
              onChange={e => setSelectedFeature(e.target.value)}
              style={{
                width: "100%", background: T.bg2, border: `1px solid ${T.border}`,
                color: T.text0, fontFamily: T.mono, fontSize: 11, padding: "4px 8px",
                borderRadius: `3px 3px 3px 8px`, outline: "none",
              }}
            >
              {fingerprint.distributions.map(d => (
                <option key={d.name} value={d.name}>{d.name}</option>
              ))}
            </select>
          </div>

          {/* Template toggles */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontFamily: T.mono, fontSize: 9, color: T.text2, letterSpacing: 1, marginBottom: 6 }}>TEMPLATES</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {TEMPLATES.map(t => (
                <button
                  key={t.id}
                  onClick={() => toggleTemplate(t.id)}
                  style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "5px 8px", border: "none", cursor: "pointer",
                    background: activeTemplates.includes(t.id) ? T.greenDim : T.bg2,
                    borderLeft: `2px solid ${activeTemplates.includes(t.id) ? T.green : "transparent"}`,
                    borderRadius: `0 3px 3px 0`,
                    fontFamily: T.mono, fontSize: 10,
                    color: activeTemplates.includes(t.id) ? T.text0 : T.text2,
                    textAlign: "left",
                  }}
                >
                  <span style={{ fontSize: 11, width: 18, textAlign: "center" }}>{t.icon}</span>
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* Cell Space toggle */}
          <div>
            <button
              onClick={() => setCellOpen(o => !o)}
              style={{
                width: "100%", padding: "8px 12px", border: `1px solid ${cellOpen ? T.green : T.border}`,
                background: cellOpen ? T.greenDim : T.bg2, color: cellOpen ? T.green : T.text1,
                fontFamily: T.mono, fontSize: 10, cursor: "pointer",
                borderRadius: `3px 3px 3px 12px`, textAlign: "left",
                display: "flex", alignItems: "center", gap: 6,
              }}
            >
              <span style={{ fontSize: 13 }}>{cellOpen ? "\u25B6" : "\u25B7"}</span>
              CELL SPACE
              <span style={{ marginLeft: "auto", fontSize: 9, color: T.text2 }}>custom viz</span>
            </button>
          </div>
        </div>
      </div>

      {/* CENTER: Visualization Grid */}
      <div style={{
        flex: 1, overflow: "auto", background: T.bg0, padding: 16,
      }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: activeTemplates.length === 1 ? "1fr" : "repeat(auto-fill, minmax(440px, 1fr))",
          gap: 12,
        }}>
          {activeTemplates.map(tid => {
            const tmpl = TEMPLATES.find(t => t.id === tid);
            if (!tmpl) return null;
            const Comp = tmpl.component;
            return (
              <div key={tid} style={{
                background: T.bg1, border: `1px solid ${T.border}`,
                borderRadius: `6px 6px 6px 16px`,
                padding: 16, minHeight: 300,
                display: "flex", flexDirection: "column",
              }}>
                <div style={{
                  fontFamily: T.mono, fontSize: 9, color: T.text2,
                  letterSpacing: 2, marginBottom: 8, display: "flex",
                  justifyContent: "space-between", alignItems: "center",
                }}>
                  <span>{tmpl.icon} {tmpl.label}</span>
                  <button onClick={() => toggleTemplate(tid)} style={{
                    background: "none", border: "none", color: T.text2,
                    cursor: "pointer", fontFamily: T.mono, fontSize: 11, padding: "0 4px",
                  }}>x</button>
                </div>
                <div style={{ flex: 1 }}>
                  <Comp fingerprint={fingerprint} selectedFeature={selectedFeature} />
                </div>
              </div>
            );
          })}
        </div>

        {activeTemplates.length === 0 && (
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            height: "100%", fontFamily: T.mono, fontSize: 12, color: T.text2,
          }}>
            Select templates from the sidebar or open the Cell Space for custom visualizations
          </div>
        )}
      </div>

      {/* RIGHT: Cell Space */}
      <CellSpace fingerprint={fingerprint} isOpen={cellOpen} onClose={() => setCellOpen(false)} />
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================
export default function DatasetWorkbench() {
  const [fingerprint, setFingerprint] = useState(null);
  const [fileName, setFileName] = useState("");
  const [view, setView] = useState("import");

  useEffect(() => {
    document.body.style.margin = "0";
    document.body.style.padding = "0";
    document.body.style.background = T.bg0;
    document.body.style.overflow = "hidden";
    document.documentElement.style.background = T.bg0;
    const root = document.getElementById("root");
    if (root) {
      root.style.width = "100%";
      root.style.height = "100vh";
      root.style.overflow = "hidden";
    }
  }, []);

  const handleDataLoaded = useCallback((fp, name) => {
    setFingerprint(fp);
    setFileName(name);
    setView("investigate");
  }, []);

  const handleReset = useCallback(() => {
    setFingerprint(null);
    setFileName("");
    setView("import");
  }, []);

  return (
    <div style={{
      width: "100vw", height: "100vh", minHeight: "100vh",
      display: "flex", flexDirection: "column",
      background: T.bg0, color: T.text0, overflow: "hidden",
      position: "fixed", top: 0, left: 0,
    }}>
      {/* Top bar */}
      <div style={{
        height: 40, display: "flex", alignItems: "center",
        padding: "0 12px", borderBottom: `1px solid ${T.border}`,
        background: T.bg1, flexShrink: 0,
      }}>
        {/* Activity rail items */}
        <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
          <button
            onClick={handleReset}
            style={{
              background: view === "import" ? T.greenDim : "none",
              border: "none", color: view === "import" ? T.green : T.text2,
              fontFamily: T.mono, fontSize: 10, padding: "6px 12px",
              cursor: "pointer", borderBottom: view === "import" ? `2px solid ${T.green}` : "2px solid transparent",
            }}
          >
            IMPORT
          </button>
          <button
            disabled={!fingerprint}
            style={{
              background: view === "investigate" ? T.greenDim : "none",
              border: "none", color: view === "investigate" ? T.green : T.text2,
              fontFamily: T.mono, fontSize: 10, padding: "6px 12px",
              cursor: fingerprint ? "pointer" : "default",
              borderBottom: view === "investigate" ? `2px solid ${T.green}` : "2px solid transparent",
              opacity: fingerprint ? 1 : 0.4,
            }}
            onClick={() => fingerprint && setView("investigate")}
          >
            INVESTIGATE
          </button>
        </div>

        <div style={{ marginLeft: "auto", fontFamily: T.mono, fontSize: 10, color: T.text2 }}>
          cvLayer // Dataset Investigator
        </div>
      </div>

      {/* Content */}
      {view === "import" && <ImportScreen onDataLoaded={handleDataLoaded} />}
      {view === "investigate" && fingerprint && (
        <InvestigationView fingerprint={fingerprint} fileName={fileName} />
      )}

      {/* Status bar */}
      <StatusBar fingerprint={fingerprint} activeView={view} />
    </div>
  );
}