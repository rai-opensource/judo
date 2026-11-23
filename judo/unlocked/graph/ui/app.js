// Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  addEdge,
  reconnectEdge,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";

// --- auto-layout (dagre) ----------------------------------------------------
const NODE_W = 240;
const EMPTY_ARR = [];
const GROUP_PAD = 24;
const GROUP_HEAD = 26;
function nodeHeight(n) {
  if (n.type === "constant") return 44;
  const rows = Math.max(
    (n.data.inputs || []).length,
    (n.data.outputs || []).length,
    1
  );
  return 60 + rows * 20;
}

function layoutNodes(nodes, edges, direction = "LR") {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: direction, nodesep: 40, ranksep: 90, marginx: 40, marginy: 40 });
  nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: nodeHeight(n) }));
  edges.forEach((e) => {
    if (e.source !== e.target) g.setEdge(e.source, e.target);
  });
  dagre.layout(g);
  return nodes.map((n) => {
    const p = g.node(n.id);
    return { ...n, position: { x: p.x - NODE_W / 2, y: p.y - nodeHeight(n) / 2 } };
  });
}

// Hierarchical layout: when ``containerBy`` groups nodes into parent boxes,
// lay each bucket out locally first, treat the resulting bounding box as a
// super-node for the outer dagre pass, then translate the children back to
// absolute coordinates. This keeps siblings near each other.
function layoutHierarchical(nodes, edges, key, direction = "LR") {
  if (!key || key === "none") return layoutNodes(nodes, edges, direction);

  const buckets = {};
  const ungrouped = [];
  nodes.forEach((n) => {
    const tag = n.data && n.data[key];
    if (tag) (buckets[tag] = buckets[tag] || []).push(n);
    else ungrouped.push(n);
  });

  // 1) layout each bucket independently
  const laidById = {};
  const bucketSize = {}; // tag -> {w, h}
  Object.entries(buckets).forEach(([tag, members]) => {
    const memberIds = new Set(members.map((m) => m.id));
    const innerEdges = edges.filter((e) => memberIds.has(e.source) && memberIds.has(e.target));
    const laid = layoutNodes(members, innerEdges, direction);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    laid.forEach((n) => {
      laidById[n.id] = n;
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + NODE_W);
      maxY = Math.max(maxY, n.position.y + nodeHeight(n));
    });
    // Normalize bucket coordinates to start at (0,0).
    laid.forEach((n) => { n.position = { x: n.position.x - minX, y: n.position.y - minY }; });
    bucketSize[tag] = {
      w: (maxX - minX) + 2 * GROUP_PAD,
      h: (maxY - minY) + 2 * GROUP_PAD + GROUP_HEAD,
    };
  });

  // 2) outer dagre over super-nodes (one per bucket) + ungrouped real nodes.
  const outer = new dagre.graphlib.Graph();
  outer.setDefaultEdgeLabel(() => ({}));
  outer.setGraph({ rankdir: direction, nodesep: 80, ranksep: 140, marginx: 40, marginy: 40 });
  Object.entries(bucketSize).forEach(([tag, sz]) => {
    outer.setNode(`__group:${tag}`, { width: sz.w, height: sz.h });
  });
  ungrouped.forEach((n) => outer.setNode(n.id, { width: NODE_W, height: nodeHeight(n) }));

  // Map any edge to its super-node endpoint.
  const superFor = (id) => {
    const n = nodes.find((x) => x.id === id);
    const tag = n && n.data && n.data[key];
    return tag ? `__group:${tag}` : id;
  };
  const seen = new Set();
  edges.forEach((e) => {
    const s = superFor(e.source);
    const t = superFor(e.target);
    if (s === t) return;
    const k = `${s}->${t}`;
    if (seen.has(k)) return;
    seen.add(k);
    outer.setEdge(s, t);
  });
  dagre.layout(outer);

  // 3) translate ungrouped nodes; translate bucket members by bucket origin.
  const result = [];
  ungrouped.forEach((n) => {
    const p = outer.node(n.id);
    result.push({ ...n, position: { x: p.x - NODE_W / 2, y: p.y - nodeHeight(n) / 2 } });
  });
  Object.entries(buckets).forEach(([tag, members]) => {
    const p = outer.node(`__group:${tag}`);
    const sz = bucketSize[tag];
    const ox = p.x - sz.w / 2 + GROUP_PAD;
    const oy = p.y - sz.h / 2 + GROUP_PAD + GROUP_HEAD;
    members.forEach((m) => {
      const laid = laidById[m.id];
      result.push({ ...m, position: { x: ox + laid.position.x, y: oy + laid.position.y } });
    });
  });
  return result;
}

// Two-level nested layout for process ⊃ instance ⊃ node: lay each process out INTERNALLY with the
// (proven) instance-grouping layout so its instances cluster without overlapping, treat each
// process's bounding box as a super-node for an outer dagre pass, then translate back to absolute
// coords. Feeds wrapInContainers (instance boxes) + processFramesFromTagged (the process frame),
// which draw the boxes from these coords.
function layoutNestedProcessInstance(nodes, edges, direction = "LR") {
  const byProc = {};
  const noProc = [];
  nodes.forEach((n) => {
    const p = n.data && n.data.process;
    if (p) (byProc[p] = byProc[p] || []).push(n);
    else noProc.push(n);
  });

  // 1) per-process: instance-clustered layout (reuse the 2-level layout), normalized to (0,0).
  const procLaid = {};
  const procSize = {};
  Object.entries(byProc).forEach(([p, members]) => {
    const ids = new Set(members.map((m) => m.id));
    const inner = edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    const laid = layoutHierarchical(members, inner, "instance", direction);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    laid.forEach((n) => {
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + NODE_W);
      maxY = Math.max(maxY, n.position.y + nodeHeight(n));
    });
    procLaid[p] = laid.map((n) => ({ ...n, position: { x: n.position.x - minX, y: n.position.y - minY } }));
    // Extra head room so the process label sits above the inner instance boxes' labels.
    procSize[p] = { w: (maxX - minX) + 2 * GROUP_PAD, h: (maxY - minY) + 2 * GROUP_PAD + 2 * GROUP_HEAD };
  });

  // 2) outer dagre over process super-nodes + any process-less nodes.
  const outer = new dagre.graphlib.Graph();
  outer.setDefaultEdgeLabel(() => ({}));
  outer.setGraph({ rankdir: direction, nodesep: 100, ranksep: 200, marginx: 40, marginy: 40 });
  Object.entries(procSize).forEach(([p, sz]) => outer.setNode(`__proc:${p}`, { width: sz.w, height: sz.h }));
  noProc.forEach((n) => outer.setNode(n.id, { width: NODE_W, height: nodeHeight(n) }));
  const superOf = (id) => {
    const n = nodes.find((x) => x.id === id);
    const p = n && n.data && n.data.process;
    return p ? `__proc:${p}` : id;
  };
  const seen = new Set();
  edges.forEach((e) => {
    const s = superOf(e.source);
    const t = superOf(e.target);
    if (s === t) return;
    const k = `${s}->${t}`;
    if (seen.has(k)) return;
    seen.add(k);
    outer.setEdge(s, t);
  });
  dagre.layout(outer);

  // 3) compose absolute positions (process box header/pad offset applied per process).
  const result = [];
  noProc.forEach((n) => {
    const q = outer.node(n.id);
    result.push({ ...n, position: { x: q.x - NODE_W / 2, y: q.y - nodeHeight(n) / 2 } });
  });
  Object.entries(procLaid).forEach(([p, laid]) => {
    const q = outer.node(`__proc:${p}`);
    const sz = procSize[p];
    const ox = q.x - sz.w / 2 + GROUP_PAD;
    const oy = q.y - sz.h / 2 + GROUP_PAD + 2 * GROUP_HEAD;
    laid.forEach((n) => result.push({ ...n, position: { x: ox + n.position.x, y: oy + n.position.y } }));
  });
  return result;
}

// Shutdown-tier layout: arrange whole PROCESSES into columns by their shutdown tier (tier 0 -- stops
// first -- on the left, later tiers to the right), stacking the processes within a tier vertically.
// Each process's internal instances/nodes are laid out compactly (same 2-level layout as the nested
// layout); only the OUTER placement differs (tier columns instead of dagre). ``tiers`` is the
// array-of-tier-name-arrays from ``computeShutdownTiers``.
function layoutTiered(nodes, edges, tiers, direction = "LR") {
  const byProc = {};
  const noProc = [];
  nodes.forEach((n) => {
    const p = n.data && n.data.process;
    if (p) (byProc[p] = byProc[p] || []).push(n);
    else noProc.push(n);
  });

  // 1) per-process inner layout, normalized to (0,0) (identical to layoutNestedProcessInstance).
  const procLaid = {};
  const procSize = {};
  Object.entries(byProc).forEach(([p, members]) => {
    const ids = new Set(members.map((m) => m.id));
    const inner = edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    const laid = layoutHierarchical(members, inner, "instance", direction);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    laid.forEach((n) => {
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + NODE_W);
      maxY = Math.max(maxY, n.position.y + nodeHeight(n));
    });
    procLaid[p] = laid.map((n) => ({ ...n, position: { x: n.position.x - minX, y: n.position.y - minY } }));
    procSize[p] = { w: (maxX - minX) + 2 * GROUP_PAD, h: (maxY - minY) + 2 * GROUP_PAD + 2 * GROUP_HEAD };
  });

  // 2) one column per tier; append a trailing column for any processes not covered by a tier.
  const columns = (tiers || []).map((names) => names.filter((p) => procSize[p]));
  const placed = new Set(columns.flat());
  const leftover = Object.keys(procSize).filter((p) => !placed.has(p));
  if (leftover.length) columns.push(leftover);

  const COL_GAP = 160;  // horizontal gap between tier columns
  const ROW_GAP = 80;   // vertical gap between processes stacked within a tier
  const colHeights = columns.map((names) =>
    names.reduce((h, p) => h + procSize[p].h, 0) + Math.max(0, names.length - 1) * ROW_GAP);
  const maxColH = Math.max(0, ...colHeights);
  const procCenter = {};
  let x = 40;
  columns.forEach((names, ci) => {
    const colW = Math.max(0, ...names.map((p) => procSize[p].w));
    let y = 40 + (maxColH - colHeights[ci]) / 2;  // vertically center each column
    names.forEach((p) => {
      const sz = procSize[p];
      procCenter[p] = { x: x + colW / 2, y: y + sz.h / 2 };
      y += sz.h + ROW_GAP;
    });
    x += colW + COL_GAP;
  });

  // 3) compose absolute node positions from each process's center (same offset math as the nested layout).
  const result = [];
  Object.entries(procLaid).forEach(([p, laid]) => {
    const c = procCenter[p];
    const sz = procSize[p];
    const ox = c.x - sz.w / 2 + GROUP_PAD;
    const oy = c.y - sz.h / 2 + GROUP_PAD + 2 * GROUP_HEAD;
    laid.forEach((n) => result.push({ ...n, position: { x: ox + n.position.x, y: oy + n.position.y } }));
  });
  // Process-less nodes (rare): stack them in a trailing column.
  let ny = 40;
  noProc.forEach((n) => {
    result.push({ ...n, position: { x, y: ny } });
    ny += nodeHeight(n) + ROW_GAP;
  });
  return result;
}

const POLICIES = ["lossless", "latest", "keep_latest"];

// --- deterministic color assignment for groups ------------------------------
const PALETTE = [
  "#e57373", "#f06292", "#ba68c8", "#9575cd", "#7986cb",
  "#64b5f6", "#4fc3f7", "#4dd0e1", "#4db6ac", "#81c784",
  "#aed581", "#dce775", "#fff176", "#ffd54f", "#ffb74d",
  "#ff8a65", "#a1887f", "#90a4ae",
];
function hashColor(key) {
  if (!key) return null;
  let h = 0;
  for (let i = 0; i < key.length; i++) h = ((h << 5) - h + key.charCodeAt(i)) | 0;
  return PALETTE[Math.abs(h) % PALETTE.length];
}

// Assign each distinct name its own palette color by sorted position. Unlike
// hashColor (which can hash two different names to the same slot -- e.g.
// spot_control_process and policy_process both landing on green), this never
// collides until there are more names than palette entries.
function distinctColorMap(names) {
  const uniq = Array.from(new Set(names.filter(Boolean))).sort();
  const map = {};
  uniq.forEach((name, i) => { map[name] = PALETTE[i % PALETTE.length]; });
  return map;
}

function shortTarget(t) {
  if (!t) return "";
  const colon = t.lastIndexOf(":");
  return colon >= 0 ? t.slice(colon + 1) : t;
}

// Pretty-print a frequency for the node-card timer chip. Avoid scientific
// notation for the common 10/100/333/1000 Hz cases and trim trailing zeros.
function formatHz(hz) {
  if (!Number.isFinite(hz)) return "?";
  if (hz >= 100) return hz.toFixed(0);
  if (hz >= 10) return hz.toFixed(1).replace(/\.0$/, "");
  return hz.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

// --- node component ---------------------------------------------------------
// Externally driven node types (mirror of judo.unlocked.node.NodeType). Keyed by the spec's ``node_type``
// string; "unlocked" is absent because those nodes ARE scheduler-driven and get no tint. Each
// entry names the CSS modifier applied to .node-card and the tooltip explaining the drive method.
const EXTERNAL_NODE_TYPE = {
  generator: {
    className: "node-type-generator",
    title: "NodeType.GENERATOR \u2014 no worker thread; spun by an external consumer iterating Node.generator",
  },
  callback: {
    className: "node-type-callback",
    title: "NodeType.CALLBACK \u2014 no worker thread; spun by an external producer calling Node.callback",
  },
};

function UnlockedNode({ id, data }) {
  const inputs = data.inputs || [];
  const outputs = data.outputs || [];
  const bound = data.bound_inputs || {};
  const rows = Math.max(inputs.length, outputs.length, 1);
  const procColor = data.procColor || hashColor(data.process);
  const style = {};
  if (data.highlightBg) {
    // Tint the whole card with its (distinct) process color so every node in
    // the selected node's process reads as one background-colored group.
    style.background = data.highlightBg + "22";
    style.borderColor = data.highlightBg;
  }
  if (data.highlight === "primary") {
    style.boxShadow = "0 0 0 3px #1e88e5";
  } else if (data.highlight) {
    style.boxShadow = `0 0 0 2px ${data.highlight}`;
  }
  // Externally driven nodes get a colored outline + warm fill so it is obvious at a glance which
  // nodes the scheduler does NOT give a worker thread (they are spun from outside the graph), and
  // which of the two drive methods applies: orange for generator, yellow for callback.
  const external = EXTERNAL_NODE_TYPE[data.node_type];
  return React.createElement(
    "div",
    {
      className: external ? `node-card ${external.className}` : "node-card",
      style,
      title: external ? external.title : undefined,
    },
    React.createElement(
      "div",
      { className: "head" },
      React.createElement("div", { className: "title", title: data.label }, data.label),
      React.createElement(
        "div",
        { className: "target", title: data.target },
        shortTarget(data.target)
      ),
      (data.process || data.rate_hz != null) &&
        React.createElement(
          "div",
          { className: "badges" },
          data.rate_hz != null && React.createElement(
            "span",
            {
              className: "badge",
              style: { background: "#37474f", color: "#fff" },
              title: `tick rate ${data.rate_hz} Hz`,
            },
            `${formatHz(data.rate_hz)} Hz`
          ),
          data.process && React.createElement(
            "span",
            { className: "badge", style: { background: procColor } },
            data.process
          )
        )
    ),
    React.createElement(
      "div",
      { className: "rows" },
      Array.from({ length: rows }).map((_, i) => {
        const inName = inputs[i];
        const isBound = inName != null && Object.prototype.hasOwnProperty.call(bound, inName);
        return React.createElement(
          "div",
          { key: i, className: "row" },
          React.createElement(
            "div",
            { className: "cell in" + (isBound ? " bound" : "") },
            inName != null
              ? [
                  React.createElement(Handle, {
                    key: "h", type: "target", position: Position.Left,
                    id: `in:${inName}`,
                  }),
                  React.createElement("span", { key: "l", className: "label" }, inName),
                ]
              : null
          ),
          React.createElement(
            "div",
            { className: "cell out" },
            outputs[i] != null
              ? [
                  React.createElement("span", { key: "l", className: "label" }, outputs[i]),
                  React.createElement(Handle, {
                    key: "h", type: "source", position: Position.Right,
                    id: `out:${outputs[i]}`,
                  }),
                ]
              : null
          )
        );
      })
    )
  );
}

// --- group container node (parent wrapper) -----------------------------------
function GroupNode({ data }) {
  const color = data.color || hashColor(data.key) || "#999";
  const threads = data.kind === "proc" && Array.isArray(data.threads) ? data.threads : null;
  return React.createElement(
    "div",
    {
      className: "group-card",
      style: { borderColor: color, background: color + "22", position: "relative" },
    },
    React.createElement(
      "div",
      { className: "group-head", style: { background: color } },
      React.createElement("span", { className: "group-kind" }, data.kind || data.kindLabel || ""),
      React.createElement("span", { className: "group-label" }, data.label),
      React.createElement("span", { className: "group-count" }, `(${data.count})`)
    ),
    // Non-node background threads for this process (threads not shown as nodes): small pills in the
    // frame's bottom margin, below the instance boxes. A solid ↻ pill with a colored background
    // freezes on Pause; a dashed ⚠ pill with a white background won't (a potential rogue).
    // Double-click opens the thread's source.
    threads && threads.length
      ? React.createElement(
          "div",
          {
            style: {
              position: "absolute", left: 8, bottom: 8, right: 8,
              display: "flex", flexWrap: "wrap", gap: 4, pointerEvents: "auto",
            },
          },
          threads.map((t) => {
            // ``managed`` (pause-aware) threads freeze on Pause; the rest keep running (a potential
            // rogue). Distinguish with the glyph + tooltip so both are visible but tell-apart.
            const freezes = !!t.managed;
            const glyph = freezes ? "\u21BB" : "\u26A0"; // ↻ freezes on pause · ⚠ won't freeze
            const pauseNote = freezes ? "freezes on Pause" : "won't freeze on Pause";
            return React.createElement(
              "div",
              {
                key: t.name,
                className: "nodrag nopan",
                title:
                  (t.owner
                    ? `background thread — ${pauseNote}\nspawned by ${t.owner}`
                    : `background thread — ${pauseNote}`) +
                  (t.file ? `\n\ndouble-click to open ${t.file}:${t.line || 1} in VS Code` : ""),
                onDoubleClick: (e) => {
                  e.stopPropagation();
                  e.preventDefault();
                  if (!t.file) { console.warn("thread pill: no source file resolved for", t.name); return; }
                  // Ask the server (which runs on the remote in a Remote-SSH session) to open the
                  // file via its remote-aware ``code`` CLI, so it lands in the window the user is
                  // actually looking at. Only fall back to a ``vscode://file`` URI (local VS Code)
                  // when the server itself is unreachable -- for a Remote-SSH setup that URI opens
                  // the wrong (local) machine, so we don't use it when the server merely reports a
                  // failure; we surface the reason instead.
                  fetch("/open-in-editor", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ file: t.file, line: t.line || 1 }),
                  })
                    .then((r) => r.json())
                    .then((j) => {
                      if (!j || !j.ok) console.warn("open in VS Code failed:", (j && j.msg) || "unknown", "-", t.file);
                    })
                    .catch(() => window.open(`vscode://file/${t.file}:${t.line || 1}`, "_blank"));
                },
                style: {
                  display: "inline-flex", alignItems: "center", gap: 5,
                  padding: "3px 10px", borderRadius: 12,
                  border: `2px ${freezes ? "solid" : "dashed"} ${color}`,
                  background: freezes ? `${color}55` : "#fff",
                  fontSize: 12, fontWeight: 700, color: "#111", whiteSpace: "nowrap",
                  cursor: t.file ? "pointer" : "default",
                },
              },
              React.createElement("span", { style: { color, fontWeight: 700 } }, glyph),
              React.createElement("span", null, t.name)
            );
          })
        )
      : null
  );
}

const nodeTypes = { unlocked: UnlockedNode, group: GroupNode, constant: ConstantNode };

// --- constant node (visualizes a bound_input) -------------------------------
function ConstantNode({ data }) {
  const v = data.value;
  const text = JSON.stringify(v);
  // Strip surrounding quotes from short strings for a cleaner one-line look.
  const short = typeof v === "string" ? v : text;
  return React.createElement(
    "div",
    { className: "const-card mono", title: `${data.port} = ${text}` },
    React.createElement("span", { className: "const-value" }, short),
    React.createElement(Handle, {
      type: "source", position: Position.Right, id: "out:value",
    })
  );
}

// --- spec <-> flow conversion ----------------------------------------------
// Heuristic: derive a display rate (Hz) from a node's kwargs when the
// node's target looks like a timer. Supports periodic_timer / subordinate_timer
// (``frequency=`` / ``period_s=``) and master_clock (``resolution_s=``).
function deriveRateHz(target, kwargs) {
  if (!target || !kwargs) return null;
  const t = String(target);
  const isTimer = /timer|master_clock/i.test(t);
  if (!isTimer) return null;
  if (typeof kwargs.frequency === "number" && kwargs.frequency > 0) return kwargs.frequency;
  if (typeof kwargs.period_s === "number" && kwargs.period_s > 0) return 1 / kwargs.period_s;
  if (typeof kwargs.resolution_s === "number" && kwargs.resolution_s > 0) return 1 / kwargs.resolution_s;
  // Defaults the factory itself uses when the kwarg is omitted:
  // make_master_clock_node(resolution_s=0.001) -> 1000 Hz.
  if (/master_clock/i.test(t)) return 1000;
  return null;
}

// Expected output rate (Hz) per node, for the edge flow-stat warning colors. Timer nodes seed their
// own derived rate (``data.rate_hz``); that rate then propagates downstream along edges -- a node
// fed by a 50 Hz timer is *expected* to produce at ~50 Hz. Processed fastest-timer-first so a node
// reachable from several timers takes the fastest driving rate. Heuristic (ignores downsamplers /
// per-input policies), but good enough to flag an edge that's clearly slower than its driver.
function computeExpectedRates(nodes, edges) {
  const rate = {};
  nodes.forEach((n) => { if (n.data && n.data.rate_hz > 0) rate[n.id] = n.data.rate_hz; });
  const outAdj = {};
  edges.forEach((e) => {
    if (e.data && e.data.isConstEdge) return;
    (outAdj[e.source] = outAdj[e.source] || []).push(e.target);
  });
  Object.keys(rate).sort((a, b) => rate[b] - rate[a]).forEach((seed) => {
    const r = rate[seed];
    const queue = [seed];
    const seen = new Set([seed]);
    while (queue.length) {
      const cur = queue.shift();
      (outAdj[cur] || []).forEach((tgt) => {
        if (seen.has(tgt)) return;
        seen.add(tgt);
        if (rate[tgt] == null) rate[tgt] = r; // inherit the driving timer's rate; keep a node's own
        queue.push(tgt);
      });
    }
  });
  return rate;
}

// Stable per-edge key for the persisted expected-frequency config: ``src.outport->dst.inport``.
// Stable across graph rebuilds (unlike the synthetic ``e0``/``e1`` ids).
function edgeKey(e) {
  if (!e) return "";
  return `${e.source}.${stripPrefix(e.sourceHandle, "out:")}->${e.target}.${stripPrefix(e.targetHandle, "in:")}`;
}

// Anchor each constant chip beside its owner node's left edge, vertically
// centered on the input-port row it binds. Constants are deliberately NOT part
// of the dagre layout -- as first-class nodes dagre spaces them a full rank
// (100px+) away from their gate -- so after every (re)layout of the real nodes
// we pin each chip to its owner here. Shared by specToFlow (initial load) and
// relayoutNodes (auto-layout button, containerBy switch, hide/unhide reflow) so
// the chips always track their gate. Owners absent from ``laidRealNodes`` (e.g.
// hidden) leave their chip untouched.
function positionConstNodes(laidRealNodes, constNodes) {
  const byId = Object.fromEntries(laidRealNodes.map((n) => [n.id, n]));
  return constNodes.map((c) => {
    const owner = byId[c.data.ownerId];
    if (!owner) return c;
    const inputs = (owner.data.inputs) || [];
    const idx = inputs.indexOf(c.data.port);
    // Match the CSS in index.html: .head padding 6+6, 1px bottom border,
    // title line-height 16, target line-height 14, badges (when present)
    // margin-top 3 + line-height 12 = 15 extra.
    const headerH = 6 + 16 + 14 + 6 + 1 + (owner.data.process ? 15 : 0);
    const ROW_H = 20, CONST_H = 24;
    // CONST_W must match the fixed width set in CSS (.const-card width).
    const CONST_W = 53, GAP = 30;
    // Port handle sits at the vertical center of its row; nudge down a few
    // pixels to compensate for the const card's box-shadow/border so the
    // straight edge sits exactly on the handle.
    const portY = headerH + (idx >= 0 ? idx : 0) * ROW_H + ROW_H / 2 + 4;
    return {
      ...c,
      position: {
        x: owner.position.x - CONST_W - GAP,
        y: owner.position.y + portY - CONST_H / 2,
      },
    };
  });
}

// Re-run the hierarchical layout over a live node array and re-anchor the
// constant chips -- the same two-step specToFlow does on load, so EVERY later
// relayout keeps the chips glued to their gates. Lays out ONLY the real nodes
// (constants would be flung a rank away by dagre), drops const edges + edges to
// absent nodes (so a filtered/hidden subset never spawns phantom dagre nodes),
// then pins the chips back beside their owners via positionConstNodes.
function relayoutNodes(allNodes, allEdges, containerBy, tiers) {
  const realNodes = allNodes.filter((n) => n.type !== "constant");
  const constNodes = allNodes.filter((n) => n.type === "constant");
  const presentIds = new Set(realNodes.map((n) => n.id));
  const realEdges = allEdges.filter(
    (e) => !(e.data && e.data.isConstEdge) && presentIds.has(e.source) && presentIds.has(e.target)
  );
  const laidReal = tiers
    ? layoutTiered(realNodes, realEdges, tiers)
    : containerBy === "nested"
      ? layoutNestedProcessInstance(realNodes, realEdges)
      : layoutHierarchical(realNodes, realEdges, containerBy);
  return [...laidReal, ...positionConstNodes(laidReal, constNodes)];
}

function specToFlow(spec, containerBy) {
  const nodes = spec.nodes.map((n) => ({
    id: n.id,
    type: "unlocked",
    position: { x: 0, y: 0 },
    data: {
      label: n.id,
      target: n.target,
      inputs: n.inputs,
      outputs: n.outputs,
      kwargs: n.kwargs,
      warmup: n.warmup,
      pre_open: n.pre_open || null,
      post_close: n.post_close || null,
      process: n.process || null,
      instance: n.instance || null,
      bound_inputs: n.bound_inputs || {},
      node_type: n.node_type || "unlocked",
      close_order: n.close_order || 0,
      rate_hz: deriveRateHz(n.target, n.kwargs),
    },
  }));
  const loopCounters = {};
  const edges = spec.edges.map((e, i) => {
    const sDot = e.from.lastIndexOf(".");
    const tDot = e.to.lastIndexOf(".");
    const src = e.from.slice(0, sDot);
    const dst = e.to.slice(0, tDot);
    const isSelfLoop = src === dst;
    const loopIndex = isSelfLoop ? (loopCounters[src] = (loopCounters[src] || 0) + 1) - 1 : 0;
    return makeEdge({
      id: `e${i}`,
      source: src,
      sourceHandle: `out:${e.from.slice(sDot + 1)}`,
      target: dst,
      targetHandle: `in:${e.to.slice(tDot + 1)}`,
      policy: e.policy,
      extra_args: e.extra_args || [],
      policy_kwargs: e.policy_kwargs || {},
      isSelfLoop,
      loopIndex,
    });
  });
  // Emit synthetic constant nodes + edges for each bound input. These are
  // visualization-only and are stripped out by flowToSpec; bound_inputs
  // continues to live on the owner node's data.
  const constNodes = [];
  spec.nodes.forEach((n) => {
    Object.entries(n.bound_inputs || {}).forEach(([port, value]) => {
      const cid = `__const:${n.id}:${port}`;
      constNodes.push({
        id: cid,
        type: "constant",
        position: { x: 0, y: 0 },
        data: { ownerId: n.id, port, value },
        selectable: true,
        draggable: true,
      });
      edges.push({
        id: `__const_edge:${n.id}:${port}`,
        source: cid,
        sourceHandle: "out:value",
        target: n.id,
        targetHandle: `in:${port}`,
        type: "straight",
        zIndex: 1000,
        interactionWidth: 12,
        data: { isConstEdge: true },
        className: "edge-const",
        deletable: false,
        reconnectable: false,
        selectable: false,
      });
    });
  });
  // Lay out the real nodes and pin each constant chip beside its owner's input
  // gate (positionConstNodes) -- the same two-step every later relayout reuses,
  // so the chips always track their gate instead of dagre flinging them a rank
  // away.
  return { nodes: relayoutNodes([...nodes, ...constNodes], edges, containerBy), edges };
}

function makeEdge({ id, source, sourceHandle, target, targetHandle, policy, extra_args, policy_kwargs, isSelfLoop, loopIndex }) {
  const ea = extra_args || [];
  // extra_args is legacy; bound constants now live on the node as bound_inputs
  // and are rendered on the port itself. Only show ea in the label if non-empty
  // (older specs may still carry values).
  const lbl = ea.length ? `${policy} · ${ea.join(",")}` : policy;
  return {
    id, source, sourceHandle, target, targetHandle,
    type: "default",
    zIndex: 1000,
    interactionWidth: 24,
    label: lbl,
    data: { policy, extra_args: ea, policy_kwargs: policy_kwargs || {}, loopIndex: loopIndex || 0, isSelfLoop: !!isSelfLoop },
    className: `edge-${policy}`,
  };
}

function stripPrefix(s, p) {
  return s && s.startsWith(p) ? s.slice(p.length) : s;
}

// Wrap regular nodes inside React Flow parent "group" nodes so each process
// (or instance) shows up as a labelled container box. Children's positions are
// converted to be relative to the parent. Nodes without the chosen tag are
// rendered ungrouped at the top level.
function wrapInContainers(nodes, key /* "process" | "instance" */) {
  // Bucket children by the chosen tag.
  const buckets = {};
  const ungrouped = [];
  nodes.forEach((n) => {
    const tag = n.data && n.data[key];
    if (tag) (buckets[tag] = buckets[tag] || []).push(n);
    else ungrouped.push(n);
  });

  const parents = [];
  const children = [];

  Object.entries(buckets).forEach(([tag, members]) => {
    // Bounding box of the children in absolute coordinates.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    members.forEach((n) => {
      const w = NODE_W;
      const h = nodeHeight(n);
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + w);
      maxY = Math.max(maxY, n.position.y + h);
    });
    const parentId = `__group:${key}:${tag}`;
    const parentPos = { x: minX - GROUP_PAD, y: minY - GROUP_PAD - GROUP_HEAD };
    parents.push({
      id: parentId,
      type: "group",
      position: parentPos,
      // ReactFlow uses style.width/height to size group nodes.
      style: {
        width:  (maxX - minX) + 2 * GROUP_PAD,
        height: (maxY - minY) + 2 * GROUP_PAD + GROUP_HEAD,
      },
      data: { label: tag, count: members.length, kind: key === "process" ? "proc" : "inst", key: tag, _members: members.map((m) => m.id) },
      selectable: true,
      draggable: true,
      zIndex: 0,
    });
    members.forEach((n) => {
      children.push({
        ...n,
        parentId,
        extent: "parent",
        position: { x: n.position.x - parentPos.x, y: n.position.y - parentPos.y },
      });
    });
  });

  // Parents must come before children in the array.
  return [...parents, ...ungrouped, ...children];
}

// Process frames: a labelled box drawn BEHIND the instance boxes of each process, framing them as
// "process ⊃ instances". Intentionally a decorative sibling (not a React Flow parent of the instance
// boxes), so the proven single-level instance⊃node parenting + drag/const logic stays untouched. The
// process-clustered layout (layoutNestedProcessInstance) keeps each process's instances adjacent so
// the frame is tight. Computed from the pre-wrap absolute node positions (which the instance boxes
// are also derived from, so the frame lines up around them).
function processFramesFromTagged(tagged) {
  const byProc = {};
  tagged.forEach((n) => {
    const proc = n.data && n.data.process;
    if (!proc || n.type === "constant") return;
    const b = (byProc[proc] = byProc[proc] || { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity, members: [] });
    b.members.push(n.id);
    b.minX = Math.min(b.minX, n.position.x);
    b.minY = Math.min(b.minY, n.position.y);
    b.maxX = Math.max(b.maxX, n.position.x + NODE_W);
    b.maxY = Math.max(b.maxY, n.position.y + nodeHeight(n));
  });
  // Pad OUTSIDE the instance boxes (which already extend GROUP_PAD + GROUP_HEAD beyond their nodes).
  const OUT_PAD = 2 * GROUP_PAD;
  const OUT_HEAD = 2 * GROUP_HEAD;
  return Object.entries(byProc).map(([proc, b]) => ({
    id: `__group:process:${proc}`,
    type: "group",
    position: { x: b.minX - OUT_PAD, y: b.minY - OUT_PAD - OUT_HEAD },
    style: {
      width: (b.maxX - b.minX) + 2 * OUT_PAD,
      height: (b.maxY - b.minY) + 2 * OUT_PAD + OUT_HEAD + GROUP_HEAD,
    },
    // ``_members`` (this process's real nodes) wires the frame into the group-drag logic in
    // onNodesChange: grabbing the frame's header drags every node in the process by the same delta;
    // the instance boxes + this frame then re-hug the moved nodes on the next render. The frame's
    // position/size stay auto-computed from the node bbox, so it never drifts out of sync.
    data: { label: proc, kind: "proc", key: proc, _members: b.members },
    selectable: false,
    draggable: true,
    zIndex: 0,
  }));
}

function flowToSpec(name, nodes, edges, instances, processes) {
  // Drop synthetic visualization-only nodes/edges (constant bound inputs).
  const realNodes = nodes.filter((n) => n.type !== "constant");
  const realEdges = edges.filter((e) => !(e.data && e.data.isConstEdge));
  // Recompute process.members from current node tags.
  const processMembers = {};
  realNodes.forEach((n) => {
    const p = n.data.process;
    if (!p) return;
    (processMembers[p] = processMembers[p] || []).push(n.id);
  });
  const outProcesses = Object.keys(processMembers).map((pname) => ({
    name: pname,
    kind: (processes[pname] && processes[pname].kind) || "Threaded",
    members: processMembers[pname],
    // Shutdown-ordering config (see ShutdownTierPanel / GraphBuilder.process). Carried through so a
    // Run honors the tiers even when they were only edited in the browser.
    tier: (processes[pname] && processes[pname].tier != null) ? processes[pname].tier : null,
    grace_s: (processes[pname] && processes[pname].grace_s != null) ? processes[pname].grace_s : null,
    role: (processes[pname] && processes[pname].role) || null,
    stop_before: (processes[pname] && processes[pname].stop_before) || [],
  }));
  return {
    name,
    nodes: realNodes.map((n) => ({
      id: n.id,
      target: n.data.target,
      kwargs: n.data.kwargs || {},
      warmup: n.data.warmup || [],
      inputs: n.data.inputs || [],
      outputs: n.data.outputs || [],
      pre_open: n.data.pre_open || null,
      post_close: n.data.post_close || null,
      process: n.data.process || null,
      instance: n.data.instance || null,
      bound_inputs: n.data.bound_inputs || {},
      node_type: n.data.node_type || "unlocked",
      close_order: n.data.close_order || 0,
    })),
    edges: realEdges.map((e) => ({
      from: `${e.source}.${stripPrefix(e.sourceHandle, "out:")}`,
      to: `${e.target}.${stripPrefix(e.targetHandle, "in:")}`,
      policy: (e.data && e.data.policy) || "lossless",
      extra_args: (e.data && e.data.extra_args) || [],
      policy_kwargs: (e.data && e.data.policy_kwargs) || {},
    })),
    instances: Object.values(instances),
    processes: outProcesses,
  };
}

// ---------------------------------------------------------------------------
// Shutdown tiers (mirror of GraphBuilder._shutdown_tiers / _grace_for). Lower tier stops first;
// processes in a tier stop in parallel. Explicit ``tier`` seeds the layering; ``stop_before`` edges
// push a process one tier after each predecessor. Grace defaults by role.
// ---------------------------------------------------------------------------
const SHUTDOWN_GRACE_DEFAULTS = { safety_critical: 30, recorder: 120, "": 3 };

function shutdownGraceFor(cfg) {
  if (cfg && cfg.grace_s != null) return cfg.grace_s;
  return SHUTDOWN_GRACE_DEFAULTS[(cfg && cfg.role) || ""];
}

function computeShutdownTiers(processes, nodes) {
  const names = new Set();
  nodes.forEach((n) => { if (n.data && n.data.process) names.add(n.data.process); });
  Object.keys(processes).forEach((p) => names.add(p));
  const edges = [];
  Object.values(processes).forEach((p) => {
    (p.stop_before || []).forEach((other) => {
      if (names.has(p.name) && names.has(other)) edges.push([p.name, other]);
    });
  });
  const tier = {};
  names.forEach((n) => {
    tier[n] = processes[n] && processes[n].tier != null ? processes[n].tier : 0;
  });
  for (let i = 0; i <= names.size; i++) {
    let changed = false;
    for (const [a, b] of edges) {
      if (tier[b] <= tier[a]) { tier[b] = tier[a] + 1; changed = true; }
    }
    if (!changed) break;
  }
  const byTier = {};
  Object.entries(tier).forEach(([n, t]) => { (byTier[t] = byTier[t] || []).push(n); });
  return Object.keys(byTier).map(Number).sort((a, b) => a - b).map((t) => byTier[t].sort());
}

// Floating panel: view + edit the shutdown-tier DAG (per-process tier / grace / role / stop-before).
function ShutdownTierPanel({ processes, setProcesses, nodes, onClose }) {
  const tiers = React.useMemo(() => computeShutdownTiers(processes, nodes), [processes, nodes]);
  const nodeCount = React.useMemo(() => {
    const c = {};
    nodes.forEach((n) => { const p = n.data && n.data.process; if (p) c[p] = (c[p] || 0) + 1; });
    return c;
  }, [nodes]);
  const allNames = React.useMemo(() => {
    const s = new Set(Object.keys(processes));
    nodes.forEach((n) => { if (n.data && n.data.process) s.add(n.data.process); });
    return Array.from(s).sort();
  }, [processes, nodes]);

  const setCfg = (name, patch) =>
    setProcesses((ps) => ({
      ...ps,
      [name]: { name, kind: "Threaded", members: [], ...(ps[name] || {}), ...patch },
    }));

  const row = (name) => {
    const cfg = processes[name] || {};
    return React.createElement(
      "div",
      { key: name, className: "tier-row",
        style: { display: "flex", alignItems: "center", gap: 8, padding: "3px 0" } },
      React.createElement("span", {
        style: { width: 12, height: 12, borderRadius: 3, background: hashColor(name), flex: "0 0 auto" },
      }),
      React.createElement("span", { style: { flex: 1, fontSize: 12 } },
        name, React.createElement("span", { style: { color: "#999" } }, ` · ${nodeCount[name] || 0} nodes`)),
      React.createElement("label", { style: { fontSize: 10, color: "#666" } }, "tier ",
        React.createElement("input", {
          type: "number", value: cfg.tier != null ? cfg.tier : "", placeholder: "0",
          style: { width: 44 },
          onChange: (e) => setCfg(name, { tier: e.target.value === "" ? null : Number(e.target.value) }),
        })),
      React.createElement("label", { style: { fontSize: 10, color: "#666" } }, "grace ",
        React.createElement("input", {
          type: "number", min: 0, value: cfg.grace_s != null ? cfg.grace_s : "",
          placeholder: `${shutdownGraceFor(cfg)}`, style: { width: 52 },
          onChange: (e) => setCfg(name, { grace_s: e.target.value === "" ? null : Number(e.target.value) }),
        }), "s"),
      React.createElement("select", {
        value: cfg.role || "", style: { fontSize: 11 },
        onChange: (e) => setCfg(name, { role: e.target.value || null }),
      }, ["", "safety_critical", "recorder"].map((r) =>
        React.createElement("option", { key: r || "none", value: r }, r || "—"))),
    );
  };

  return React.createElement(
    "div",
    { className: "tier-panel",
      style: { position: "absolute", top: 52, right: 12, width: 460, maxHeight: "80vh", overflow: "auto",
               background: "#fff", border: "1px solid #ccc", borderRadius: 6, padding: 12, zIndex: 2000,
               boxShadow: "0 4px 16px rgba(0,0,0,0.15)" } },
    React.createElement("div", { style: { display: "flex", justifyContent: "space-between", marginBottom: 6 } },
      React.createElement("strong", null, "Shutdown tiers"),
      React.createElement("button", { onClick: onClose }, "✕")),
    React.createElement("div", { style: { fontSize: 11, color: "#666", marginBottom: 8 } },
      "Lower tier stops first; a tier stops in parallel. Consumers should stop before the producers ",
      "they read. Edits are honored on the next Run."),
    tiers.map((names, idx) =>
      React.createElement("div", { key: idx, className: "tier-lane",
        style: { border: "1px solid #eee", borderRadius: 5, padding: "6px 8px", marginBottom: 6 } },
        React.createElement("div", { style: { fontSize: 11, fontWeight: 600, color: "#37474f", marginBottom: 2 } },
          `Tier ${idx}`,
          idx === 0 ? " — stops first" : "",
          ` · grace ${Math.max(...names.map((n) => shutdownGraceFor(processes[n])))}s`),
        names.map(row))),
    allNames.length === 0 && React.createElement("div", { style: { color: "#999" } }, "no processes"),
  );
}

// ---------------------------------------------------------------------------
// Persisted visibility: hidden node/instance ids survive reloads and runs by
// round-tripping through localStorage as JSON arrays.
const LS_HIDDEN_NODES = "unlockedHiddenNodeIds";
const LS_HIDDEN_INSTANCES = "unlockedHiddenInstanceIds";
// Selected Debug Step timer set (names). Persisted across runs/reloads.
const LS_STEP_TIMERS = "unlockedStepTimers";
// Last-discovered non-node background threads (list of {name,process,owner,file,line}). Persisted
// so the process boxes keep showing their thread pills when idle; each Run/Debug refreshes it.
const LS_THREADS = "unlockedDiscoveredThreads";

function loadHiddenSet(key) {
  try {
    const arr = JSON.parse(localStorage.getItem(key) || "[]");
    return new Set(Array.isArray(arr) ? arr : []);
  } catch (e) {
    return new Set();
  }
}

function saveHiddenSet(key, set) {
  try {
    localStorage.setItem(key, JSON.stringify([...set]));
  } catch (e) {
    /* ignore quota / disabled storage */
  }
}

function loadThreads() {
  try {
    const arr = JSON.parse(localStorage.getItem(LS_THREADS) || "[]");
    return Array.isArray(arr) ? arr : [];
  } catch (e) {
    return [];
  }
}

function saveThreads(list) {
  try {
    localStorage.setItem(LS_THREADS, JSON.stringify(Array.isArray(list) ? list : []));
  } catch (e) {
    /* ignore quota / disabled storage */
  }
}

// ---------------------------------------------------------------------------
// Error boundary: a render error in one panel (e.g. the edge inspector on an odd
// tap value) must NOT white-screen the whole editor. Catch it, show the message
// + stack in place, and keep the rest of the app alive. ``resetKey`` lets the
// parent clear the error when the offending selection changes.
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, info) {
    console.error("[editor] render error:", error, info && info.componentStack);
  }
  componentDidUpdate(prev) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) this.setState({ error: null });
  }
  render() {
    if (this.state.error) {
      const e = this.state.error;
      return React.createElement(
        "div",
        { className: this.props.className, style: { padding: 12, color: "#b71c1c", fontFamily: "ui-monospace, monospace", fontSize: 12, whiteSpace: "pre-wrap", overflow: "auto", background: "#fff", minWidth: 260 } },
        React.createElement("div", { style: { fontWeight: 700, marginBottom: 6 } }, this.props.label || "This panel crashed."),
        React.createElement("div", { style: { marginBottom: 8, color: "#555" } }, "The rest of the editor is still usable. Please report this:"),
        React.createElement("pre", { style: { background: "#fff3f3", border: "1px solid #ef9a9a", borderRadius: 6, padding: 10, margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word" } },
          String((e && e.stack) || e)),
      );
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
function App() {
  const [name, setName] = useState("graph");
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [instances, setInstances] = useState({}); // id -> {id, factory, kwargs}
  const [processes, setProcesses] = useState({}); // name -> {name, kind, members}
  const [selectedId, setSelectedId] = useState(null);
  const [selectedInstanceId, setSelectedInstanceId] = useState(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState(null);
  // { nodeId, port } of a clicked bound-constant chip, or null.
  const [selectedBound, setSelectedBound] = useState(null);
  // factory string -> {loading?: true} | {error?: string} | introspection result
  const [instanceStates, setInstanceStates] = useState({});
  // instance id -> {loading?: true} | {error?: string} | { fields: [...] }.
  // Keyed per-instance (not per-factory) because the schema depends on the
  // current ctor kwargs -- e.g. SMPC's ``optimizer_name`` selects which
  // optimizer sub-form (cem / mppi / ps) the server returns.
  const [instanceSchemas, setInstanceSchemas] = useState({});
  // edge.id -> { loading? | error? | { in: {...}, out: {...} } }
  const [edgeTypes, setEdgeTypes] = useState({});
  // Per-edge expected frequency (Hz) overrides, persisted server-side in a repo config file so they
  // survive runs/restarts. Keyed by ``edgeKey``. Empty entries fall back to the derived timer rate.
  const [expectedOverrides, setExpectedOverrides] = useState({});
  useEffect(() => {
    fetch("/edge_config").then((r) => r.json()).then((d) => setExpectedOverrides(d.expected_hz || {})).catch(() => {});
  }, []);
  const setExpectedHz = useCallback((key, hz) => {
    if (!key) return;
    setExpectedOverrides((cur) => {
      const next = { ...cur };
      const v = Number(hz);
      if (hz === "" || hz == null || !(v > 0)) delete next[key];
      else next[key] = v;
      return next;
    });
  }, []);
  const [expectedSaved, setExpectedSaved] = useState(false);
  const saveExpectedHz = useCallback(() => {
    fetch("/edge_config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_hz: expectedOverrides }),
    }).then((r) => r.json()).then((d) => {
      if (d && d.expected_hz) setExpectedOverrides(d.expected_hz);
      setExpectedSaved(true);
      setTimeout(() => setExpectedSaved(false), 1500);
    }).catch(() => {});
  }, [expectedOverrides]);
  // ---- run-config / play state ----
  // Whether the Run graph panel is open. Closing it (its window "close"
  // button) hides the panel entirely; the toolbar "Run graph" toggle reopens
  // it. Run state lives here in App, so a run keeps going while the panel is
  // closed and its live state reappears on reopen.
  const [runPanelOpen, setRunPanelOpen] = useState(true);
  // Whether the Shutdown-tiers editor panel is open (view/edit per-process stop order + grace).
  const [tierOpen, setTierOpen] = useState(false);
  // Builder-driven kwargs UI: server-introspected params + current values.
  const [builderInfo, setBuilderInfo] = useState({ spec: "", params: [], kwargs: {}, meta: { vision_tasks: [] } });
  const [builderKwargs, setBuilderKwargs] = useState({});
  const [builderError, setBuilderError] = useState("");
  const [runState, setRunState] = useState({ running: false, pid: null, exit_code: null, cmd: [], lines: [], offset: 0, debug: true });
  // True from Stop/Kill press until status poll confirms the child exited.
  // Keeps builder inputs + Play available while the child finishes shutting
  // down (otherwise users see disabled fields for the 1-2s teardown window).
  const [stopRequested, setStopRequested] = useState(false);
  const runOffsetRef = useRef(0);
  const logRef = useRef(null);
  // Auto-scroll only while the user is parked at the bottom. Any manual scroll
  // up flips this false (stop following); scrolling back to the bottom re-arms
  // it. Starts true so a fresh run tails output.
  const stickToBottomRef = useRef(true);
  // Live edge taps: Set<edge_id> + map of latest sample by edge_id.
  const [tapEdgeIds, setTapEdgeIds] = useState(() => new Set());
  const [tapSamples, setTapSamples] = useState({});  // edge_id -> { ts, value, v }
  const tapSinceRef = useRef(0);
  // Pause/step control (Phase B+C) mirrored from server /control.
  const [ctrl, setCtrl] = useState({ paused: false, global_epoch: 0, node_epochs: {}, controller_running: false, health: { ready: false, started: 0, total: 0, pending: [], errors: [] }, shutdown_requested: false, shutdown_ready: false, version: 0 });
  // Time-travel (Phase D): edge_id -> samples[] history (always populated for tapped edges).
  // ``scrubTime`` is a global wall-clock time (unix seconds) or null for live.
  const [historyByEdge, setHistoryByEdge] = useState({});
  const [scrubTime, setScrubTime] = useState(null);
  // Debug Step target: the set of subordinate-timer names Step runs until one of
  // them fires (then re-pauses). Empty = every timer (step to the next tick of
  // anything). Persisted so the selection survives runs and reloads.
  const [stepTimers, setStepTimers] = useState(() => loadHiddenSet(LS_STEP_TIMERS));
  useEffect(() => { saveHiddenSet(LS_STEP_TIMERS, stepTimers); }, [stepTimers]);
  // Non-node background threads discovered during a Run/Debug (with source file/line). Persisted so
  // the process boxes keep rendering their thread pills when idle; each Run/Debug re-discovers and
  // refreshes this via the health snapshot below.
  const [persistedThreads, setPersistedThreads] = useState(() => loadThreads());
  // Graph editing is disabled while an app run is active so the diagram stays
  // in sync with the running topology. ``lockedRef`` is consulted from inside
  // useCallback closures to avoid re-creating them on every run state change.
  const lockedRef = useRef(false);
  useEffect(() => { lockedRef.current = !!runState.running; }, [runState.running]);
  const locked = !!runState.running;
  // Hover tooltip state: { id, x, y } | null
  const [edgeHover, setEdgeHover] = useState(null);
  const [groupBy, setGroupBy] = useState("instance");
  // Container mode: ``nested`` draws process boxes with instance boxes inside (process ⊃ instance ⊃
  // node); the layout + wrapping helpers switch on this. Kept as a const so all the memoized
  // ``containerBy`` deps stay unchanged.
  const containerBy = "nested";
  const [hiddenNodeIds, setHiddenNodeIds] = useState(() => loadHiddenSet(LS_HIDDEN_NODES)); // node ids hidden from the canvas (view-only)
  // Instances the user explicitly hid (via an instance eye). Remembered across
  // graph rebuilds: when a rebuild adds new nodes to one of these instances, the
  // new nodes come in hidden and the instance stays greyed out. ``hiddenNodeIds``
  // is what the canvas filters on; this is the persistent "which instances" memory.
  const [hiddenInstanceIds, setHiddenInstanceIds] = useState(() => loadHiddenSet(LS_HIDDEN_INSTANCES));
  // Live mirrors so callbacks (applyBuiltSpec, layout) can read the current sets
  // without re-creating themselves on every hide/unhide.
  const hiddenNodeIdsRef = useRef(hiddenNodeIds);
  useEffect(() => { hiddenNodeIdsRef.current = hiddenNodeIds; }, [hiddenNodeIds]);
  const hiddenInstanceIdsRef = useRef(hiddenInstanceIds);
  useEffect(() => { hiddenInstanceIdsRef.current = hiddenInstanceIds; }, [hiddenInstanceIds]);
  // Persist visibility so hidden nodes/instances survive reloads and runs.
  useEffect(() => { saveHiddenSet(LS_HIDDEN_NODES, hiddenNodeIds); }, [hiddenNodeIds]);
  useEffect(() => { saveHiddenSet(LS_HIDDEN_INSTANCES, hiddenInstanceIds); }, [hiddenInstanceIds]);

  // Reconcile the persisted/known hidden set against a freshly built flow and
  // lay out ONLY the visible subset -- so hidden nodes never push the visible
  // ones apart (the whole point of hiding). Returns the laid-out node list
  // (visible reflowed + hidden appended with their prior positions) and the
  // reconciled hidden-id set. Shared by the initial load, kwargs-rebuilds and
  // Run so every path that regenerates the diagram respects visibility.
  const layoutWithHidden = useCallback((f) => {
    const alive = new Set(f.nodes.map((n) => n.id));
    const hiddenInsts = hiddenInstanceIdsRef.current || new Set();
    const nextHidden = new Set();
    // Keep still-present individually-hidden ids; drop dead ones.
    hiddenNodeIdsRef.current.forEach((id) => { if (alive.has(id)) nextHidden.add(id); });
    // (Re)hide every node belonging to a memorized-hidden instance so new
    // members come in hidden and the instance stays greyed out.
    if (hiddenInsts.size) {
      f.nodes.forEach((n) => {
        const inst = n.data && n.data.instance;
        if (inst && hiddenInsts.has(inst)) nextHidden.add(n.id);
      });
    }
    const isHidden = (n) =>
      nextHidden.has(n.id) ||
      (n.type === "constant" && n.data && nextHidden.has(n.data.ownerId));
    const visible = f.nodes.filter((n) => !isHidden(n));
    const hidden = f.nodes.filter(isHidden);
    return { nodes: [...relayoutNodes(visible, f.edges, containerBy), ...hidden], hidden: nextHidden };
  }, [containerBy]);

  useEffect(() => {
    fetch("/graph.json")
      .then((r) => r.json())
      .then((spec) => {
        setName(spec.name || "graph");
        const f = specToFlow(spec, containerBy);
        const laid = layoutWithHidden(f);
        setNodes(laid.nodes);
        setEdges(f.edges);
        setHiddenNodeIds(laid.hidden);
        const insts = {};
        (spec.instances || []).forEach((i) => { insts[i.id] = i; });
        setInstances(insts);
        const procs = {};
        (spec.processes || []).forEach((p) => { procs[p.name] = p; });
        setProcesses(procs);
      })
      .catch((err) => console.error("load graph:", err));
  }, []);

  // Debounced autosave: whenever the graph changes (after the initial load),
  // POST the current spec to /graph.json so a reload preserves edits without
  // requiring the user to hit "Save to server" first. Skips while a run is
  // active (graph is locked) and during the initial empty render.
  const autosaveSeededRef = useRef(false);
  useEffect(() => {
    if (locked) return undefined;
    // Don't autosave the empty pre-load state, and don't autosave the very
    // first render that simply mirrors the server's current graph.
    if (!nodes.length && !edges.length && !Object.keys(instances).length) return undefined;
    if (!autosaveSeededRef.current) { autosaveSeededRef.current = true; return undefined; }
    const handle = setTimeout(() => {
      const spec = flowToSpec(name, nodes, edges, instances, processes);
      fetch("/graph.json", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
      }).catch((err) => console.error("autosave:", err));
    }, 500);
    return () => clearTimeout(handle);
  }, [name, nodes, edges, instances, processes, locked]);

  // Track the most recently rendered parent ("group") positions so we can
  // translate child nodes by the drag delta when a parent is moved.
  const groupPosRef = useRef({});           // parentId -> {x, y}
  const groupMembersRef = useRef({});       // parentId -> [childId, ...]
  const childToParentRef = useRef({});      // childId -> parentId
  // Track owner-node positions so dragging an owner node moves its attached
  // constant-input visualization nodes by the same delta.
  const ownerPosRef = useRef({});           // ownerId -> {x, y}
  const ownerToConstsRef = useRef({});      // ownerId -> [constId, ...]

  // Rebuild owner->consts index whenever the nodes list changes. Always
  // refresh ownerPosRef with the latest absolute owner positions — not just
  // seed it — so that subsequent direct drags compute their delta from the
  // current position (e.g. after a container has been dragged, which moves
  // owners through childDeltas without firing per-owner position events).
  useEffect(() => {
    const idx = {};
    const pos = {};
    nodes.forEach((n) => {
      if (n.type === "constant" && n.data && n.data.ownerId) {
        (idx[n.data.ownerId] = idx[n.data.ownerId] || []).push(n.id);
      } else if (n.type === "unlocked") {
        pos[n.id] = { x: n.position.x, y: n.position.y };
      }
    });
    ownerToConstsRef.current = idx;
    ownerPosRef.current = pos;
  }, [nodes]);

  const onNodesChange = useCallback((c) => {
    if (lockedRef.current) {
      c = c.filter((ch) => ch.type === "position" || ch.type === "dimensions" || ch.type === "select");
      if (!c.length) return;
    }
    // Apply drag deltas from parent ("group") nodes to their children's
    // absolute positions in the underlying `nodes` state.
    const childDeltas = {};                 // childId -> {dx, dy}
    c.forEach((ch) => {
      if (ch.type !== "position" || !ch.id || !ch.id.startsWith("__group:")) return;
      if (!ch.position) return;
      const prev = groupPosRef.current[ch.id];
      if (!prev) {
        // First time we see this group move; baseline so subsequent deltas work.
        groupPosRef.current[ch.id] = { ...ch.position };
        return;
      }
      const dx = ch.position.x - prev.x;
      const dy = ch.position.y - prev.y;
      groupPosRef.current[ch.id] = { ...ch.position };
      if (dx === 0 && dy === 0) return;
      const members = groupMembersRef.current[ch.id] || [];
      members.forEach((cid) => {
        const cur = childDeltas[cid] || { dx: 0, dy: 0 };
        childDeltas[cid] = { dx: cur.dx + dx, dy: cur.dy + dy };
      });
    });

    // For position changes on owner nodes that have attached constant nodes,
    // propagate the same delta to those constants so they move together.
    c.forEach((ch) => {
      if (ch.type !== "position" || !ch.id) return;
      if (ch.id.startsWith("__group:") || ch.id.startsWith("__const:")) return;
      const consts = ownerToConstsRef.current[ch.id];
      if (!consts || !consts.length) return;
      // Resolve absolute position. RF sometimes omits positionAbsolute on
      // child-of-parent events; in that case translate ch.position by the
      // parent's absolute position. Never use ch.position raw for an in-group
      // node — it's relative and would produce huge bogus deltas.
      let abs = ch.positionAbsolute;
      if (!abs) {
        const parentId = childToParentRef.current[ch.id];
        if (parentId && ch.position) {
          const p = groupPosRef.current[parentId];
          if (p) abs = { x: ch.position.x + p.x, y: ch.position.y + p.y };
        } else if (ch.position) {
          abs = ch.position;
        }
      }
      if (!abs) return;
      const prev = ownerPosRef.current[ch.id];
      ownerPosRef.current[ch.id] = { x: abs.x, y: abs.y };
      if (!prev) return;
      const dx = abs.x - prev.x;
      const dy = abs.y - prev.y;
      if (dx === 0 && dy === 0) return;
      consts.forEach((cid) => {
        const cur = childDeltas[cid] || { dx: 0, dy: 0 };
        childDeltas[cid] = { dx: cur.dx + dx, dy: cur.dy + dy };
      });
    });

    // Non-group changes. For position changes on nodes that are children of a
    // rendered group, ReactFlow reports a position *relative to the parent*;
    // our underlying `nodes` state stores absolute positions, so prefer the
    // `positionAbsolute` field. Drop the bare `position` to avoid corrupting
    // absolute state. If ``positionAbsolute`` is missing (older RF events),
    // translate by adding the parent's absolute position.
    const filtered = [];
    c.forEach((ch) => {
      if (ch.id && ch.id.startsWith("__group:")) return;
      if (ch.type === "position" && ch.id) {
        const parentId = childToParentRef.current[ch.id];
        if (parentId) {
          if (ch.positionAbsolute) {
            filtered.push({ ...ch, position: ch.positionAbsolute });
          } else if (ch.position) {
            const p = groupPosRef.current[parentId];
            if (p) {
              filtered.push({ ...ch, position: { x: ch.position.x + p.x, y: ch.position.y + p.y } });
            } else {
              filtered.push(ch);
            }
          } else {
            filtered.push(ch);
          }
          return;
        }
      }
      filtered.push(ch);
    });

    if (Object.keys(childDeltas).length === 0 && filtered.length === 0) return;

    setNodes((ns) => {
      let next = ns;
      if (Object.keys(childDeltas).length) {
        next = next.map((n) => {
          const d = childDeltas[n.id];
          if (!d) return n;
          return { ...n, position: { x: n.position.x + d.dx, y: n.position.y + d.dy } };
        });
      }
      if (filtered.length) next = applyNodeChanges(filtered, next);
      return next;
    });
  }, []);
  const onEdgesChange = useCallback((c) => {
    if (lockedRef.current) {
      c = c.filter((ch) => ch.type !== "remove" && ch.type !== "add" && ch.type !== "reset");
      if (!c.length) return;
    }
    setEdges((es) => applyEdgeChanges(c, es));
  }, []);
  const onConnect = useCallback(
    (params) => {
      if (lockedRef.current) return;
      setEdges((es) => {
        const isSelfLoop = params.source === params.target;
        const loopIndex = isSelfLoop
          ? es.filter((e) => e.source === params.source && e.target === params.target).length
          : 0;
        const edge = makeEdge({
          id: `e_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
          source: params.source, sourceHandle: params.sourceHandle,
          target: params.target, targetHandle: params.targetHandle,
          policy: "lossless", extra_args: [], policy_kwargs: {},
          isSelfLoop, loopIndex,
        });
        return addEdge(edge, es);
      });
    },
    []
  );

  // Resolve an edge's source-output and target-input types (cached).
  const fetchEdgeTypes = useCallback((edge) => {
    if (!edge || !edge.id) return;
    if (edgeTypes[edge.id] && !edgeTypes[edge.id].error) return;
    const src = nodes.find((n) => n.id === edge.source);
    const dst = nodes.find((n) => n.id === edge.target);
    if (!src || !dst) return;
    const outPort = stripPrefix(edge.sourceHandle, "out:");
    const inPort  = stripPrefix(edge.targetHandle, "in:");
    const outIdx = (src.data.outputs || []).indexOf(outPort);
    const inIdx  = (dst.data.inputs  || []).indexOf(inPort);
    setEdgeTypes((m) => ({ ...m, [edge.id]: { ...(m[edge.id] || {}), loading: true } }));
    const params = new URLSearchParams();
    const promises = [];
    if (src.data.target && outIdx >= 0) {
      promises.push(
        fetch(`/port_type?target=${encodeURIComponent(src.data.target)}&side=out&index=${outIdx}`)
          .then((r) => (r.ok ? r.json() : null))
          .catch(() => null)
          .then((info) => ["out", info, outPort])
      );
    } else {
      promises.push(Promise.resolve(["out", null, outPort]));
    }
    if (dst.data.target && inIdx >= 0) {
      promises.push(
        fetch(`/port_type?target=${encodeURIComponent(dst.data.target)}&side=in&index=${inIdx}`)
          .then((r) => (r.ok ? r.json() : null))
          .catch(() => null)
          .then((info) => ["in", info, inPort])
      );
    } else {
      promises.push(Promise.resolve(["in", null, inPort]));
    }
    Promise.all(promises).then((results) => {
      const next = { out: null, in: null };
      results.forEach(([k, info, port]) => { next[k] = info ? { ...info, port } : { port }; });
      setEdgeTypes((m) => ({ ...m, [edge.id]: next }));
    });
  }, [nodes, edgeTypes]);

  const onEdgeClick = useCallback((e, edge) => {
    e.stopPropagation();
    if (edge.data && edge.data.isConstEdge) return;
    setSelectedId(null);
    setSelectedInstanceId(null);
    setSelectedBound(null);
    setSelectedEdgeId(edge.id);
    fetchEdgeTypes(edge);
    // Tap interactions:
    //   shift-click  -> toggle this edge in the tap set (compare several at once).
    //   plain click  -> while running, tap ONLY this edge (replace). Tapping every edge floods the
    //                   tap ingest with high-rate queues (2 kHz physics, etc.) and starves the
    //                   control plane, making Pause/Step/Stop laggy -- so we keep it to one.
    if (e.shiftKey) {
      setTapEdgeIds((cur) => {
        const next = new Set(cur);
        if (next.has(edge.id)) next.delete(edge.id); else next.add(edge.id);
        return next;
      });
    } else if (runState.running) {
      setTapEdgeIds((cur) => (cur.size === 1 && cur.has(edge.id) ? cur : new Set([edge.id])));
    }
  }, [fetchEdgeTypes, runState.running]);

  const onEdgeMouseEnter = useCallback((e, edge) => {
    if (edge.data && edge.data.isConstEdge) return;
    setEdgeHover({ id: edge.id, x: e.clientX, y: e.clientY });
    fetchEdgeTypes(edge);
  }, [fetchEdgeTypes]);
  const onEdgeMouseMove = useCallback((e, edge) => {
    if (edge.data && edge.data.isConstEdge) return;
    setEdgeHover({ id: edge.id, x: e.clientX, y: e.clientY });
  }, []);
  const onEdgeMouseLeave = useCallback(() => setEdgeHover(null), []);

  const setEdgePolicy = useCallback((edgeId, newPolicy) => {
    setEdges((es) =>
      es.map((e) => {
        if (e.id !== edgeId) return e;
        const ea = (e.data && e.data.extra_args) || [];
        const lbl = ea.length ? `${newPolicy} · ${ea.join(",")}` : newPolicy;
        return { ...e, data: { ...e.data, policy: newPolicy }, label: lbl, className: `edge-${newPolicy}` };
      })
    );
  }, []);

  const deleteEdge = useCallback((edgeId) => {
    if (lockedRef.current) return;
    setEdges((es) => es.filter((e) => e.id !== edgeId));
    setSelectedEdgeId((cur) => (cur === edgeId ? null : cur));
  }, []);

  // Reject connections to an input port that already has a real (non-const) edge.
  const MAX_EDGES_PER_INPUT = 1;
  const isValidConnection = useCallback((conn) => {
    if (lockedRef.current) return false;
    if (!conn || !conn.target || !conn.targetHandle) return true;
    const count = edges.reduce((n, e) => {
      if (e.data && e.data.isConstEdge) return n;
      return n + (e.target === conn.target && e.targetHandle === conn.targetHandle ? 1 : 0);
    }, 0);
    return count < MAX_EDGES_PER_INPUT;
  }, [edges]);

  const onReconnect = useCallback(
    (oldEdge, newConnection) => {
      if (lockedRef.current) return;
      setEdges((es) => reconnectEdge(oldEdge, newConnection, es));
    },
    []
  );

  const onSave = useCallback(() => {
    const spec = flowToSpec(name, nodes, edges, instances, processes);
    fetch("/graph.json", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec, null, 2),
    })
      .then((r) => r.text())
      .then((t) => console.log("saved:", t))
      .catch((err) => console.error("save:", err));
  }, [name, nodes, edges, instances, processes]);

  // Re-arrange on demand (the toolbar "Auto-layout" button). We deliberately do NOT
  // relayout on every hide/unhide -- the user rearranges when they want to. This lays
  // out only the VISIBLE subset so the remaining nodes reflow to fill the gaps hidden
  // nodes leave; hidden nodes keep their old positions (they're off-canvas via
  // ``highlightedNodes``) and get a fresh spot whenever they're unhidden and a later
  // Auto-layout runs. ``relayoutNodes`` drops edges to hidden nodes (no phantom dagre
  // nodes) and re-pins each constant chip beside its owner gate.
  const onRelayout = useCallback(() => {
    setNodes((ns) => {
      if (!ns.length) return ns;
      const isHidden = (n) =>
        hiddenNodeIds.has(n.id) ||
        (n.type === "constant" && n.data && hiddenNodeIds.has(n.data.ownerId));
      const visible = ns.filter((n) => !isHidden(n));
      const hidden = ns.filter(isHidden);
      return [...relayoutNodes(visible, edges, containerBy), ...hidden];
    });
  }, [edges, containerBy, hiddenNodeIds]);

  // Arrange whole processes into shutdown-tier columns (tier 0 stops first -> leftmost). Mirrors
  // ``onRelayout`` but feeds the computed tiers to ``relayoutNodes`` so the outer placement is by
  // tier instead of dagre. Tiers come from the same computation the shutdown-tier panel shows.
  const onTierLayout = useCallback(() => {
    setNodes((ns) => {
      if (!ns.length) return ns;
      const tiers = computeShutdownTiers(processes, ns);
      const isHidden = (n) =>
        hiddenNodeIds.has(n.id) ||
        (n.type === "constant" && n.data && hiddenNodeIds.has(n.data.ownerId));
      const visible = ns.filter((n) => !isHidden(n));
      const hidden = ns.filter(isHidden);
      return [...relayoutNodes(visible, edges, containerBy, tiers), ...hidden];
    });
  }, [edges, containerBy, hiddenNodeIds, processes]);

  // Re-layout automatically when the user switches container grouping so
  // children of the same group end up adjacent.
  useEffect(() => {
    setNodes((ns) => (ns.length ? relayoutNodes(ns, edges, containerBy) : ns));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerBy]);

  const onNodeDoubleClick = useCallback((_, node) => {
    let target = null;
    if (node.id && node.id.startsWith("__group:")) {
      if (node.data && node.data.kind === "inst") {
        const inst = instances[node.data.key];
        target = inst && inst.factory;
      }
    } else {
      target = node.data && node.data.target;
    }
    if (!target) return;
    fetch(`/resolve?target=${encodeURIComponent(target)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(({ file, line }) => { window.location.href = `vscode://file/${file}:${line}`; })
      .catch((err) => console.error("resolve", target, err));
  }, [instances]);

  const onExport = useCallback(() => {
    const spec = flowToSpec(name, nodes, edges, instances, processes);
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.json`;
    a.click();
  }, [name, nodes, edges, instances, processes]);

  // ---- run / stop the app ------------------------------------------------
  const stopApp = useCallback((kill = false) => {
    setStopRequested(true);
    fetch("/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kill }),
    }).catch((err) => console.error("/stop failed", err));
  }, []);

  // Fetch builder param introspection once on mount. Seeds ``builderKwargs``
  // from server-side defaults so the Play button can be pressed without
  // touching any field.
  useEffect(() => {
    fetch("/builder/args")
      .then((r) => r.json())
      .then((info) => {
        setBuilderInfo(info || { spec: "", params: [], kwargs: {} });
        const seed = {};
        (info && info.params || []).forEach((p) => {
          if (info.kwargs && Object.prototype.hasOwnProperty.call(info.kwargs, p.name)) {
            seed[p.name] = info.kwargs[p.name];
          } else if (p.default !== null && p.default !== undefined) {
            seed[p.name] = p.default;
          }
        });
        // ``use_mocap`` has a task-aware default in build() (bool|None): non-vision tasks require
        // mocap. Its static signature default is null, so reflect the resolved default here for the
        // initial (non-vision) task, keeping the checkbox in sync with the graph the server builds.
        const visionTasks = (info && info.meta && info.meta.vision_tasks) || [];
        if (seed.task !== undefined && !visionTasks.includes(seed.task) &&
            (seed.use_mocap === undefined || seed.use_mocap === null)) {
          seed.use_mocap = true;
        }
        // A network_sim backend talks to the sim + desktop over loopback: default both
        // host knobs so a seeded sim graph is remote-controlled (no local viser) out of the box.
        if (seed.backend === "network_sim") {
          const simHost = (info && info.meta && info.meta.sim_default_host) || "127.0.0.1";
          if (!seed.sim_host) seed.sim_host = simHost;
          if (!seed.commander_host) seed.commander_host = simHost;
        } else if (Object.prototype.hasOwnProperty.call(seed, "sim_host")) {
          seed.sim_host = "";
        }
        setBuilderKwargs(seed);
      })
      .catch((err) => console.error("/builder/args failed", err));
  }, []);

  // Apply a freshly-built graph spec to the editor (nodes/edges/instances/
  // processes), so the diagram reflects current kwargs.
  const applyBuiltSpec = useCallback((spec) => {
    if (!spec) return;
    setName(spec.name || "graph");
    const f = specToFlow(spec, containerBy);
    // Reconcile hidden nodes with the rebuilt graph and lay out only the visible
    // subset, so hiding an instance keeps the compact layout across Run/rebuild
    // (previously the fresh spec was laid out with hidden nodes still present,
    // spreading everything back out as if nothing were hidden).
    const laid = layoutWithHidden(f);
    setNodes(laid.nodes);
    setEdges(f.edges);
    setHiddenNodeIds(laid.hidden);
    const insts = {};
    (spec.instances || []).forEach((i) => { insts[i.id] = i; });
    setInstances(insts);
    const procs = {};
    (spec.processes || []).forEach((p) => { procs[p.name] = p; });
    setProcesses(procs);
  }, [containerBy, layoutWithHidden]);

  // Rebuild the graph diagram (without running) whenever kwargs change while
  // idle. Debounced so rapid edits don't spam the server. Skipped during a
  // run since the editor is locked to the currently-running spec.
  useEffect(() => {
    if (runState.running) return undefined;
    if (!builderInfo || !(builderInfo.params || []).length) return undefined;
    const id = setTimeout(() => {
      fetch("/builder/build", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kwargs: builderKwargs }),
      })
        .then(async (r) => {
          const body = await r.json().catch(() => ({}));
          if (!r.ok || body.ok === false) {
            throw new Error(body.error || `build failed (HTTP ${r.status})`);
          }
          setBuilderError("");
          applyBuiltSpec(body.graph);
        })
        .catch((err) => setBuilderError(String(err.message || err)));
    }, 250);
    return () => clearTimeout(id);
  }, [builderKwargs, builderInfo, runState.running, applyBuiltSpec]);

  // Run/Debug: rebuild the graph from the builder + current kwargs, refresh the
  // editor diagram from the new spec, then launch ``run_graph`` against it.
  // ``debug`` selects the run flavor: Debug installs the tap (edge taps +
  // master-clock gating + Pause/Step); plain Run is lightweight (no tap, no
  // clock gating -- only Stop/Kill). The server resets the control plane before
  // the child starts, so the high-level controller is parked on Debug until
  // Play. ``pauseOnStart`` is kept for backward compatibility but the server no
  // longer freezes the graph at startup.
  const playGraph = useCallback((debug = true, pauseOnStart = false) => {
    runOffsetRef.current = 0;
    stickToBottomRef.current = true;  // fresh run: re-arm auto-follow
    setRunState((s) => ({ ...s, lines: [], offset: 0, exit_code: null, debug }));
    setBuilderError("");
    setStopRequested(false);
    fetch("/builder/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kwargs: builderKwargs }),
    })
      .then(async (r) => {
        const body = await r.json().catch(() => ({}));
        if (!r.ok || body.ok === false) {
          throw new Error(body.error || `build failed (HTTP ${r.status})`);
        }
        return body.graph;
      })
      .then((spec) => {
        // Refresh the editor diagram so the user sees the wiring they just
        // chose (e.g. mujoco-block vs hardware-block) before/while it runs.
        applyBuiltSpec(spec);
        return fetch("/run/graph", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pause_on_start: pauseOnStart, debug }),
        });
      })
      .then((r) => r && r.json())
      .catch((err) => {
        console.error("play failed", err);
        setBuilderError(String(err.message || err));
      });
  }, [builderKwargs, applyBuiltSpec]);

  // Build: bazel-build the run_graph binary (via pixel) so subsequent Runs take
  // the fast direct-launch path. Streams build output into the console.
  const buildGraph = useCallback(() => {
    runOffsetRef.current = 0;
    stickToBottomRef.current = true;  // tail the build output
    setRunState((s) => ({ ...s, lines: [], offset: 0, exit_code: null }));
    setBuilderError("");
    setStopRequested(false);
    fetch("/build/graph", { method: "POST" })
      .then(async (r) => {
        const body = await r.json().catch(() => ({}));
        if (!r.ok || body.ok === false) {
          throw new Error(body.msg || `build failed to start (HTTP ${r.status})`);
        }
      })
      .catch((err) => {
        console.error("build failed", err);
        setBuilderError(String(err.message || err));
      });
  }, []);

  // Poll log status while the run panel is open or a run is active.
  useEffect(() => {
    let cancelled = false;
    // Guard against overlapping requests: a request only advances the offset in
    // its .then(), so if a second tick fired while one was in flight it would
    // re-request the same offset and the server would re-send (and we'd append)
    // the same lines. Skipping ticks while a request is outstanding keeps the
    // offset monotonic and the console duplicate-free.
    let inFlight = false;
    const tick = () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      fetch(`/run/status?offset=${runOffsetRef.current}`)
        .then((r) => r.json())
        .then((s) => {
          if (cancelled) return;
          runOffsetRef.current = s.offset;
          if (!s.running) setStopRequested(false);
          setRunState((cur) => ({
            running: s.running,
            pid: s.pid,
            exit_code: s.exit_code,
            cmd: s.cmd,
            offset: s.offset,
            debug: s.debug !== undefined ? s.debug : cur.debug,
            lines: s.lines && s.lines.length ? cur.lines.concat(s.lines) : cur.lines,
          }));
        })
        .catch(() => {})
        .finally(() => { inFlight = false; });
    };
    tick();
    const id = setInterval(tick, 500);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Auto-scroll log to bottom on new lines — but only while the user is still
  // parked at the bottom (stickToBottomRef). If they scrolled up to read an
  // error we leave the scroll position alone so it doesn't yank back down.
  useEffect(() => {
    if (stickToBottomRef.current && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [runState.lines]);

  // Track whether the console is scrolled to (near) the bottom. A manual scroll
  // up disables auto-follow; returning to the bottom re-enables it.
  const onLogScroll = useCallback((e) => {
    const el = e.currentTarget;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  }, []);

  // Download the full console log as a text file for offline investigation.
  const onDownloadLog = useCallback(() => {
    const text = (runState.lines || []).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    a.download = `run_graph_console_${ts}.log`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, [runState.lines]);

  // ---- live edge taps ---------------------------------------------------
  // Translate the current tapEdgeIds into [{edge_id, queue_name}] pairs the
  // server can map to producer-side queue names (``{node}_output_{index}``).
  const computeTapList = useCallback((ids) => {
    const list = [];
    edges.forEach((e) => {
      if (!ids.has(e.id)) return;
      if (e.data && e.data.isConstEdge) return;
      const src = nodes.find((n) => n.id === e.source);
      if (!src) return;
      const portName = (e.sourceHandle || "").replace(/^out:/, "");
      const idx = (src.data.outputs || []).indexOf(portName);
      if (idx < 0) return;
      list.push({ edge_id: e.id, queue_name: `${src.id}_output_${idx}` });
    });
    return list;
  }, [edges, nodes]);

  // Push the current tap list to the server whenever it changes.
  useEffect(() => {
    const taps = computeTapList(tapEdgeIds);
    // Guard: never wipe an active subscription with an empty list just because the edges momentarily
    // failed to resolve (e.g. a transient graph refresh where node outputs aren't populated yet).
    // Only send an empty list when the user genuinely cleared all taps. Otherwise a single bad render
    // would deregister the tap on the worker and the flow stats would freeze ("stuck after a while").
    if (tapEdgeIds.size > 0 && taps.length === 0) return;
    fetch("/tap/edges", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ taps }),
    }).catch(() => {});
  }, [tapEdgeIds, computeTapList]);

  // When the tap SET changes (switching OR re-tapping an edge), start a FRESH measurement: clear
  // cached live samples, history buffers and any scrub position. Otherwise a re-tapped edge would
  // immediately show its previous session's frozen last value (from stale history/tapSamples) and
  // never re-measure. The server likewise drops a re-subscribed edge's old samples (set_taps), so
  // both ends reset together and the panel shows "measuring…" until fresh data arrives.
  useEffect(() => {
    tapSinceRef.current = 0;
    setTapSamples({});
    setHistoryByEdge({});
    setScrubTime(null);
  }, [tapEdgeIds]);

  // Auto-clear taps when a run finishes or stops.
  useEffect(() => {
    if (!runState.running) {
      tapSinceRef.current = 0;
      setTapSamples({});
      setHistoryByEdge({});
      setScrubTime(null);
      setCtrl({ paused: false, global_epoch: 0, node_epochs: {}, controller_running: false, health: { ready: false, started: 0, total: 0, pending: [], errors: [] }, shutdown_requested: false, shutdown_ready: false, version: 0 });
    }
  }, [runState.running]);

  // Poll latest samples while a run is active and we have at least one tap.
  useEffect(() => {
    if (!runState.running || tapEdgeIds.size === 0) return undefined;
    let cancelled = false;
    let inFlight = false;  // never stack requests -> avoids ERR_INSUFFICIENT_RESOURCES under load
    const tick = () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      const ac = new AbortController();
      const to = setTimeout(() => ac.abort(), 4000);  // self-recover if a request hangs
      // Cursorless poll (``since=0``): return the CURRENT latest sample for every tapped edge every
      // cycle, exactly like ``/tap/history``. The ``since`` cursor was fragile across tap switches --
      // once it advanced past a stale global version, a freshly-tapped edge's samples could sit below
      // it and never surface (flow stats froze while the timeline kept updating). We instead dedup by
      // sample version ``v`` so unchanged samples don't trigger needless re-renders.
      fetch(`/tap/latest?since=0`, { signal: ac.signal })
        .then((r) => r.json())
        .then((data) => {
          if (cancelled) return;
          const samples = data.samples || [];
          if (!samples.length) return;
          setTapSamples((cur) => {
            let changed = false;
            const next = { ...cur };
            samples.forEach((s) => {
              const prev = cur[s.edge_id];
              if (!prev || prev.v !== s.v) { next[s.edge_id] = s; changed = true; }
            });
            return changed ? next : cur;
          });
        })
        .catch(() => {})
        .finally(() => { clearTimeout(to); inFlight = false; });
    };
    tick();
    const id = setInterval(tick, 200);
    return () => { cancelled = true; clearInterval(id); };
  }, [runState.running, tapEdgeIds]);

  const toggleTap = useCallback((edgeId) => {
    setTapEdgeIds((cur) => {
      const next = new Set(cur);
      if (next.has(edgeId)) next.delete(edgeId); else next.add(edgeId);
      return next;
    });
  }, []);
  const clearTaps = useCallback(() => setTapEdgeIds(new Set()), []);
  const tapAll = useCallback(() => {
    setTapEdgeIds(new Set(edges.filter((e) => !(e.data && e.data.isConstEdge)).map((e) => e.id)));
  }, [edges]);

  // Tap ONLY the edge the user selects (see onEdgeClick), in BOTH Debug and plain Run. Debug streams
  // the full value; a plain Run streams STATS-ONLY (count/rate/staleness) for that one edge so the
  // user can see where data is blocked or slow. Clear the tap set only when the run ends.
  useEffect(() => {
    if (!runState.running) {
      setTapEdgeIds((cur) => (cur.size ? new Set() : cur));
    }
  }, [runState.running]);

  // ----- pause / step / time-travel ----------------------------------------
  useEffect(() => {
    if (!runState.running) return undefined;
    let cancelled = false;
    let inFlight = false;  // never stack requests -> avoids ERR_INSUFFICIENT_RESOURCES under load
    const tick = () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      fetch("/control")
        .then((r) => r.json())
        .then((d) => { if (!cancelled) setCtrl(d); })
        .catch(() => {})
        .finally(() => { inFlight = false; });
    };
    tick();
    const id = setInterval(tick, 400);
    return () => { cancelled = true; clearInterval(id); };
  }, [runState.running]);

  const postPause = useCallback((paused) => {
    fetch("/control/pause", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paused }),
    }).then((r) => r.json()).then(setCtrl).catch(() => {});
    // Resuming or stepping should always snap the preview back to live.
    if (!paused) {
      setScrubTime(null);
      fetch("/control/replay", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ts: null }),
      }).catch(() => {});
    }
  }, []);
  const postStep = useCallback((node) => {
    fetch("/control/step", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(node ? { node } : {}),
    }).then((r) => r.json()).then(setCtrl).catch(() => {});
    // Stepping advances one live tick — snap the preview back to live.
    setScrubTime(null);
    fetch("/control/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ts: null }),
    }).catch(() => {});
  }, []);
  const postReplay = useCallback((ts) => {
    fetch("/control/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ts: ts == null ? null : ts }),
    }).then((r) => r.json()).then(setCtrl).catch(() => {});
  }, []);

  // Toggle a subordinate timer in/out of the Step group.
  const toggleStepTimer = useCallback((name) => {
    setStepTimers((cur) => {
      const next = new Set(cur);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  }, []);

  // Push the selected Step group to the server (Debug runs only). Fires on run
  // start (restoring the persisted selection) and whenever it changes. An empty
  // selection sends ``null`` = every timer (step to the next tick of anything).
  useEffect(() => {
    if (!runState.running || runState.debug === false) return;
    const group = stepTimers.size ? [...stepTimers] : null;
    fetch("/control/clock/step_group", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group }),
    }).catch(() => {});
  }, [runState.running, runState.debug, stepTimers]);
  const togglePause = useCallback(() => postPause(!ctrl.paused), [ctrl.paused, postPause]);

  // High-level controller (judo-style Play/Pause). Independent of the global
  // pause: the rest of the graph keeps ticking so the low-level joint-control
  // stream stays alive. The SplineFollower switches between MPC-driven
  // commands and DEFAULT_POLICY_CMD based on this flag.
  const postController = useCallback((running) => {
    fetch("/control/controller", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ running: !!running }),
    }).then((r) => r.json()).then(setCtrl).catch(() => {});
  }, []);
  const toggleController = useCallback(
    () => postController(!ctrl.controller_running),
    [ctrl.controller_running, postController],
  );

  // Refresh history for the currently-selected edge so the Inspector can show
  // its values over time (a live tail) and support scrubbing. Only the selected
  // edge is polled: with every edge tapped in Debug, fetching all 256-sample
  // histories each tick would be needlessly heavy.
  useEffect(() => {
    if (!runState.running || !selectedEdgeId) return undefined;
    const id = selectedEdgeId;
    let cancelled = false;
    let inFlight = false;  // never stack requests -> avoids ERR_INSUFFICIENT_RESOURCES under load
    const fetchHist = () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      const ac = new AbortController();
      const to = setTimeout(() => ac.abort(), 4000);  // self-recover if a request hangs
      fetch(`/tap/history?edge=${encodeURIComponent(id)}`, { signal: ac.signal })
        .then((r) => r.json())
        .then((d) => {
          if (cancelled) return;
          setHistoryByEdge((cur) => ({ ...cur, [id]: d.samples || [] }));
        })
        .catch(() => {})
        .finally(() => { clearTimeout(to); inFlight = false; });
    };
    fetchHist();
    const t = setInterval(fetchHist, 200);
    return () => { cancelled = true; clearInterval(t); };
  }, [runState.running, selectedEdgeId]);

  // Fetch the typed config schema for an instance's ctor kwargs. Seeded with
  // the current kwargs so the server can tailor the form (e.g. pick the
  // optimizer sub-schema matching ``optimizer_name``). Keeps the previous
  // fields visible while re-fetching so an optimizer switch doesn't flash empty.
  const fetchInstanceSchema = useCallback((instId, factory, kwargs) => {
    if (!factory) { setInstanceSchemas((s) => ({ ...s, [instId]: { fields: [] } })); return; }
    setInstanceSchemas((s) => ({
      ...s,
      [instId]: { loading: true, fields: (s[instId] && s[instId].fields) || undefined },
    }));
    const q = encodeURIComponent(JSON.stringify(kwargs || {}));
    fetch(`/instance_schema?target=${encodeURIComponent(factory)}&kwargs=${q}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((res) => setInstanceSchemas((s) => ({ ...s, [instId]: { fields: (res && res.fields) || [] } })))
      .catch((err) => {
        console.error("instance_schema", factory, err);
        setInstanceSchemas((s) => ({ ...s, [instId]: { error: String(err) } }));
      });
  }, []);
  const onNodeClick = useCallback((_, node) => {
    if (node.type === "constant") {
      setSelectedId(null);
      setSelectedInstanceId(null);
      setSelectedEdgeId(null);
      setSelectedBound({ nodeId: node.data.ownerId, port: node.data.port });
      return;
    }
    if (node.id && node.id.startsWith("__group:")) {
      if (node.data && node.data.kind === "inst") {
        const instId = node.data.key;
        setSelectedId(null);
        setSelectedEdgeId(null);
        setSelectedBound(null);
        setSelectedInstanceId(instId);
        const inst = instances[instId];
        const factory = inst && inst.factory;
        if (factory && !instanceStates[factory]) {
          setInstanceStates((s) => ({ ...s, [factory]: { loading: true } }));
          fetch(`/instance_state?target=${encodeURIComponent(factory)}`)
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
            .then((info) => setInstanceStates((s) => ({ ...s, [factory]: info })))
            .catch((err) => {
              console.error("instance_state", factory, err);
              setInstanceStates((s) => ({ ...s, [factory]: { error: String(err) } }));
            });
        }
        fetchInstanceSchema(instId, factory, (inst && inst.kwargs) || {});
      }
      return;
    }
    setSelectedInstanceId(null);
    setSelectedEdgeId(null);
    setSelectedBound(null);
    setSelectedId(node.id);
  }, [instances, instanceStates, fetchInstanceSchema]);
  const onPaneClick  = useCallback(() => { setSelectedId(null); setSelectedInstanceId(null); setSelectedEdgeId(null); setSelectedBound(null); }, []);

  // Assign each process/instance its own palette color by sorted position so
  // distinct names never collide the way a plain hash can (which is why
  // spot_control_process and policy_process used to share the same green).
  const procColors = useMemo(
    () => distinctColorMap(nodes.map((n) => n.data && n.data.process)),
    [nodes]
  );
  const instColors = useMemo(
    () => distinctColorMap(nodes.map((n) => n.data && n.data.instance)),
    [nodes]
  );

  // ---- highlight the selected node ----
  const highlightedNodes = useMemo(() => {
    const sel = nodes.find((n) => n.id === selectedId);
    // Drop hidden nodes (and any constant chips whose owner is hidden) before
    // grouping so each container box reflows around only the visible nodes.
    const visible = nodes.filter((n) => {
      if (hiddenNodeIds.has(n.id)) return false;
      if (n.type === "constant" && n.data && hiddenNodeIds.has(n.data.ownerId)) return false;
      return true;
    });
    const tagged = visible.map((n) => {
      const procColor = (n.data && n.data.process) ? procColors[n.data.process] : null;
      // Only the selected node is highlighted. Process/instance membership is already conveyed by the
      // container boxes, so sibling nodes no longer get a highlight tint (no same-process lighting).
      return {
        ...n,
        data: { ...n.data, procColor, highlightBg: null, highlight: sel && n.id === sel.id ? "primary" : null },
      };
    });
    const wrapped =
      containerBy === "none"
        ? tagged
        : containerBy === "nested"
          ? [...processFramesFromTagged(tagged), ...wrapInContainers(tagged, "instance")]
          : wrapInContainers(tagged, containerBy);
    // Give the container boxes the same distinct colors as the node badges.
    return wrapped.map((n) => {
      if (n.id && n.id.startsWith("__group:") && n.data) {
        const cmap = n.data.kind === "proc" ? procColors : instColors;
        const color = cmap[n.data.key];
        if (color) return { ...n, data: { ...n.data, color } };
      }
      return n;
    });
  }, [nodes, selectedId, containerBy, hiddenNodeIds, procColors, instColors]);

  // Attach the non-node background threads to their process frame's data so the process box
  // (``GroupNode``) renders them as looped pills. Threads are DISCOVERED by the health snapshot
  // during a Run/Debug; we persist the latest discovery so the pills keep showing when idle (shown
  // all the time), and each new Run/Debug refreshes them to stay consistent. Done here (not in the
  // layout-heavy ``highlightedNodes`` memo) so the ~1 Hz health poll only re-tags frames.
  const liveThreads =
    runState.running && ctrl && ctrl.health && Array.isArray(ctrl.health.unmanaged_threads)
      ? ctrl.health.unmanaged_threads
      : EMPTY_ARR;
  const liveSig = useMemo(() => JSON.stringify(liveThreads), [liveThreads]);
  // Accumulate the UNION of every background thread seen during the current run (keyed by
  // process+name). On Stop the graph tears down gracefully -- some threads (e.g. the command stream)
  // exit first, so the final health polls report a shrunken set; persisting that would drop them.
  // The union keeps everything discovered during the run so the pills survive Stop. Reset when a new
  // run starts so each Run/Debug re-discovers from scratch.
  const runThreadsRef = useRef({});
  const wasRunningRef = useRef(false);
  useEffect(() => {
    if (runState.running && !wasRunningRef.current) runThreadsRef.current = {};
    wasRunningRef.current = runState.running;
  }, [runState.running]);
  useEffect(() => {
    if (!liveThreads.length) return;
    const acc = runThreadsRef.current;
    let changed = false;
    liveThreads.forEach((t) => {
      const rec = typeof t === "string" ? { name: t } : t;
      const key = `${rec.process || ""}\u0000${rec.name}`;
      if (!acc[key] || JSON.stringify(acc[key]) !== JSON.stringify(rec)) {
        acc[key] = rec; // add or refresh (managed flag / file may resolve after the grace period)
        changed = true;
      }
    });
    if (changed) {
      const merged = Object.values(acc);
      setPersistedThreads(merged);
      saveThreads(merged);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveSig]);
  // Show the accumulated union while running (so a thread that stops mid-run doesn't flicker out)
  // and the last persisted discovery when idle -- pills render all the time.
  const unmanagedThreads = persistedThreads;
  const displayNodes = useMemo(() => {
    if (!unmanagedThreads.length) return highlightedNodes;
    const byProc = {};
    unmanagedThreads.forEach((t) => {
      const rec = typeof t === "string" ? { name: t } : t;
      const p = rec.process || null;
      (byProc[p] = byProc[p] || []).push(rec);
    });
    return highlightedNodes.map((n) =>
      n.id && n.id.startsWith("__group:process:") && n.data && byProc[n.data.key]
        ? { ...n, data: { ...n.data, threads: byProc[n.data.key] } }
        : n
    );
  }, [highlightedNodes, unmanagedThreads]);

  // Keep the group-position / group-members refs in sync with what is rendered
  // so onNodesChange can translate drag deltas into child-position updates.
  useEffect(() => {
    const pos = {};
    const mem = {};
    const c2p = {};
    highlightedNodes.forEach((n) => {
      if (n.id && n.id.startsWith("__group:")) {
        pos[n.id] = { x: n.position.x, y: n.position.y };
        const members = ((n.data && n.data._members) || []).slice();
        // True RF children that need relative<->absolute translation.
        members.forEach((cid) => { c2p[cid] = n.id; });
        // Constants whose owner is inside this group must visually follow the
        // group when it's dragged, even though they live at the top level
        // (not real RF children of the group).
        const owners = members.slice();
        owners.forEach((ownerId) => {
          const cs = ownerToConstsRef.current[ownerId] || [];
          cs.forEach((cid) => { if (!members.includes(cid)) members.push(cid); });
        });
        mem[n.id] = members;
      }
    });
    groupPosRef.current = pos;
    groupMembersRef.current = mem;
    childToParentRef.current = c2p;
  }, [highlightedNodes]);

  // ---- groupings for the side panel ----
  const grouped = useMemo(() => {
    const groups = {};
    nodes.forEach((n) => {
      if (n.type === "constant") return; // visualization-only pseudo-nodes, not real graph nodes
      let key = "—";
      if (groupBy === "process")  key = n.data.process  || "—";
      else if (groupBy === "instance") key = n.data.instance || "—";
      else if (groupBy === "target")   key = n.data.target   || "—";
      (groups[key] = groups[key] || []).push(n);
    });
    return groups;
  }, [nodes, groupBy]);

  // ---- node / instance visibility (the inspector eye toggles) ----
  // ``hiddenNodeIds`` is the single source of truth; ``highlightedNodes`` and
  // ``decoratedEdges`` filter against it, so hiding a node (or every node of an
  // instance) just drops it from the canvas and its container box reflows to fit
  // what remains. Hiding is view-only -- it never touches the saved graph spec.
  const toggleNodeHidden = useCallback((id) => {
    setHiddenNodeIds((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);
  // Hide the whole group when any member is still visible, else reveal them all
  // (one click hides an instance's nodes; a second click brings them back).
  const toggleGroupHidden = useCallback((ids) => {
    setHiddenNodeIds((cur) => {
      const next = new Set(cur);
      const allHidden = ids.length > 0 && ids.every((id) => next.has(id));
      ids.forEach((id) => (allHidden ? next.delete(id) : next.add(id)));
      return next;
    });
  }, []);
  // Instance eye: like ``toggleGroupHidden`` but also records the instance in
  // ``hiddenInstanceIds`` so the hide survives graph rebuilds (new member nodes
  // are re-hidden in ``applyBuiltSpec``). Direction is driven by the memorized
  // state so the two sets never desync.
  const toggleInstanceHidden = useCallback((instId, memberIds) => {
    const wasHidden = hiddenInstanceIdsRef.current.has(instId);
    setHiddenInstanceIds((cur) => {
      const next = new Set(cur);
      if (wasHidden) next.delete(instId); else next.add(instId);
      return next;
    });
    setHiddenNodeIds((cur) => {
      const next = new Set(cur);
      (memberIds || []).forEach((id) => (wasHidden ? next.delete(id) : next.add(id)));
      return next;
    });
  }, []);
  const showAllHidden = useCallback(() => {
    setHiddenNodeIds(new Set());
    setHiddenInstanceIds(new Set());
  }, []);

  const selected = nodes.find((n) => n.id === selectedId);
  const selectedEdge = edges.find((e) => e.id === selectedEdgeId) || null;
  // Expected output rate per node (timer rate propagated downstream) -> edge flow-warning colors.
  const expectedRates = useMemo(() => computeExpectedRates(nodes, edges), [nodes, edges]);
  // Visual highlight for the selected edge + a "TAP" badge for live taps + a rate-warning color.
  const decoratedEdges = useMemo(() => {
    return edges
      // Drop edges touching a hidden node (const edges too: their target is the
      // owner node, so a hidden owner hides its constant's edge as well).
      .filter((e) => !hiddenNodeIds.has(e.source) && !hiddenNodeIds.has(e.target))
      .map((e) => {
        const isSel = e.id === selectedEdgeId;
        const isTap = tapEdgeIds.has(e.id);
        // Rate warning: for a tapped edge, compare measured Hz against the source node's expected
        // (driving-timer) rate -- yellow if >20% slow, red if >50% slow or clearly stalled.
        let rateColor = null;
        if (isTap) {
          const st = tapSamples[e.id] && tapSamples[e.id].stats;
          const exp = expectedOverrides[edgeKey(e)] || expectedRates[e.source];
          if (st && st.age_s != null && st.age_s > 1.0) {
            rateColor = "#c62828"; // no data for >1 s -> blocked
          } else if (st && st.hz != null && exp) {
            const ratio = st.hz / exp;
            if (ratio < 0.5) rateColor = "#c62828";
            else if (ratio < 0.8) rateColor = "#f9a825";
          }
        }
        if (!isSel && !isTap && !rateColor) return e;
        const style = { ...(e.style || {}) };
        if (isSel) {
          style.strokeWidth = 4;
          style.filter = "drop-shadow(0 0 4px #1e88e5)";
        }
        if (isTap) {
          style.stroke = rateColor || "#d81b60";
          if (!isSel) style.strokeWidth = rateColor ? 4 : 3;
          if (rateColor) style.filter = `drop-shadow(0 0 4px ${rateColor})`;
        }
        const label = isTap
          ? `● ${e.label || ""}`.trim()
          : e.label;
        const cls = `${e.className || ""}${isTap ? " edge-tapped" : ""}`.trim();
        return { ...e, style, label, className: cls };
      });
  }, [edges, selectedEdgeId, tapEdgeIds, hiddenNodeIds, tapSamples, expectedRates, expectedOverrides]);
  const selectedInstance = selectedInstanceId ? instances[selectedInstanceId] || { id: selectedInstanceId } : null;
  const selectedInstanceMembers = useMemo(() => {
    if (!selectedInstanceId) return [];
    return nodes.filter((n) => n.data.instance === selectedInstanceId).map((n) => n.id);
  }, [nodes, selectedInstanceId]);
  const selectedInstanceInfo = selectedInstance && selectedInstance.factory
    ? instanceStates[selectedInstance.factory] || null
    : null;
  const selectedInstanceSchema = selectedInstanceId ? instanceSchemas[selectedInstanceId] || null : null;

  // ---- inspector handlers ----
  const updateSelected = useCallback((patch) => {
    if (!selectedId) return;
    setNodes((ns) => ns.map((n) => (n.id === selectedId ? { ...n, data: { ...n.data, ...patch } } : n)));
  }, [selectedId]);

  const assignProcess = useCallback((procName) => {
    updateSelected({ process: procName || null });
    if (procName) setProcesses((ps) => ps[procName] ? ps : { ...ps, [procName]: { name: procName, kind: "Threaded", members: [] } });
  }, [updateSelected]);

  const assignInstance = useCallback((instName) => {
    updateSelected({ instance: instName || null });
  }, [updateSelected]);

  // Patch the kwargs of the currently-selected node.
  const updateSelectedKwargs = useCallback((kwargs) => {
    if (!selectedId) return;
    setNodes((ns) => ns.map((n) => (n.id === selectedId ? { ...n, data: { ...n.data, kwargs } } : n)));
  }, [selectedId]);

  // Patch kwargs on a shared instance (factory + kwargs is what build() consumes).
  const updateInstanceKwargs = useCallback((instId, kwargs) => {
    setInstances((all) => {
      const cur = all[instId];
      if (!cur) return all;
      return { ...all, [instId]: { ...cur, kwargs } };
    });
  }, []);

  // Re-fetch an instance's schema after an ``affects_schema`` field changed
  // (e.g. SMPC ``optimizer_name``), using the freshly-edited kwargs so the
  // returned sub-form matches the new selection.
  const reschemaInstance = useCallback((instId, kwargs) => {
    const inst = instances[instId];
    fetchInstanceSchema(instId, inst && inst.factory, kwargs);
  }, [instances, fetchInstanceSchema]);

  // Patch the bound_inputs map on the currently-selected node.
  const updateSelectedBoundInputs = useCallback((bound_inputs) => {
    if (!selectedId) return;
    setNodes((ns) => ns.map((n) =>
      n.id === selectedId ? { ...n, data: { ...n.data, bound_inputs } } : n));
  }, [selectedId]);

  const allProcesses = useMemo(() => {
    const s = new Set(Object.keys(processes));
    nodes.forEach((n) => n.data.process && s.add(n.data.process));
    return Array.from(s).sort();
  }, [processes, nodes]);

  const allInstances = useMemo(() => {
    const s = new Set(Object.keys(instances));
    nodes.forEach((n) => n.data.instance && s.add(n.data.instance));
    return Array.from(s).sort();
  }, [instances, nodes]);

  return React.createElement(
    React.Fragment,
    null,
    React.createElement(
      "div",
      { className: "toolbar" },
      React.createElement("strong", null, "unLocked: "),
      React.createElement("input", {
        value: name,
        onChange: (e) => setName(e.target.value),
        style: { padding: "2px 6px", width: 160 },
      }),
      React.createElement("button", { onClick: onRelayout }, "Auto-layout"),
      React.createElement("button", { onClick: onTierLayout, title: "Arrange processes in shutdown-tier columns (tier 0 stops first, on the left)" }, "Tier layout"),
      React.createElement("button", { onClick: onSave }, "Save to server"),
      React.createElement("button", { onClick: onExport }, "Download JSON"),
      React.createElement(
        "button",
        {
          onClick: () => {
            // Open (or switch to the already-open) viser tab. Like the Foxglove button, the named
            // window target reuses the same tab: a second click focuses it instead of spawning
            // duplicates. Points at this run's viser server (localhost:8080, port-forwarded in
            // Remote-SSH).
            window.open("http://localhost:8080", "viser");
          },
          title: "Open viser (or switch to its tab) at http://localhost:8080",
        },
        "🧊 viser",
      ),
      React.createElement(
        "button",
        {
          onClick: () => {
            // Open (or switch to the already-open) Foxglove tab. The named window target reuses the
            // same tab: a second click focuses it instead of spawning duplicates. Connects Foxglove
            // to this run's websocket bridge (localhost:8765, port-forwarded in Remote-SSH).
            const ws = "ws://localhost:8765";
            window.open(
              `https://app.foxglove.dev/~/view?ds=foxglove-websocket&ds.url=${encodeURIComponent(ws)}`,
              "foxglove",
            );
          },
          title: "Open Foxglove (or switch to its tab) connected to ws://localhost:8765",
        },
        "🦊 foxglove",
      ),
      React.createElement(
        "button",
        {
          onClick: () => setRunPanelOpen((v) => !v),
          style: { background: runPanelOpen ? "#e3f2fd" : undefined },
          title: runPanelOpen ? "Hide the Run graph panel" : "Show the Run graph panel",
        },
        "▶ Run graph",
      ),
      React.createElement(
        "button",
        {
          onClick: () => setTierOpen((v) => !v),
          style: { background: tierOpen ? "#ede7f6" : undefined },
          title: "View / edit the process shutdown-tier order + grace",
        },
        "⇊ Tiers",
      ),
      React.createElement(
        "label",
        { style: { fontSize: 11, color: "#444", marginLeft: 8 } },
        "group by: ",
        React.createElement(
          "select",
          { value: groupBy, onChange: (e) => setGroupBy(e.target.value) },
          ["none", "process", "instance", "target"].map((g) =>
            React.createElement("option", { key: g, value: g }, g)
          )
        )
      ),
      React.createElement(
        "span",
        { style: { fontSize: 11, color: "#666" } },
        "click edge: inspect · dbl-click node: open in VS Code"
      )
    ),
    tierOpen && React.createElement(ShutdownTierPanel, {
      processes,
      setProcesses,
      nodes,
      onClose: () => setTierOpen(false),
    }),
    React.createElement(
      ReactFlow,
      {
        nodes: displayNodes,
        edges: decoratedEdges,
        nodeTypes,
        onNodesChange,
        onEdgesChange,
        onConnect,
        onReconnect,
        onNodeClick,
        onPaneClick,
        onNodeDoubleClick,
        isValidConnection,
        deleteKeyCode: locked ? [] : ["Delete", "Backspace"],
        edgesReconnectable: !locked,
        reconnectRadius: 20,
        onEdgeClick,
        onEdgeMouseEnter,
        onEdgeMouseMove,
        onEdgeMouseLeave,
        fitView: true,
      },
      React.createElement(Background, null),
      React.createElement(Controls, null),
      React.createElement(MiniMap, null)
    ),
    edgeHover && edgeTypes[edgeHover.id] && React.createElement(EdgeTooltip, {
      x: edgeHover.x, y: edgeHover.y, info: edgeTypes[edgeHover.id],
    }),
    React.createElement(ErrorBoundary, { className: "inspector", label: "The inspector panel crashed (likely rendering this selection).", resetKey: `${selectedId}|${selectedEdgeId}|${selectedInstanceId}|${selectedBound && selectedBound.nodeId}` },
    React.createElement(Inspector, {
      selected, allProcesses, allInstances, instances,
      onAssignProcess: assignProcess,
      onAssignInstance: assignInstance,
      onUpdateSelected: updateSelected,
      onUpdateSelectedKwargs: updateSelectedKwargs,
      onUpdateSelectedBoundInputs: updateSelectedBoundInputs,
      onUpdateInstanceKwargs: updateInstanceKwargs,
      selectedConnectedPorts: selectedId ? new Set(
        edges.filter((e) => e.target === selectedId && !(e.data && e.data.isConstEdge))
             .map((e) => stripPrefix(e.targetHandle, "in:"))
      ) : new Set(),
      grouped, groupBy,
      hiddenNodeIds,
      hiddenInstanceIds,
      onToggleNodeHidden: toggleNodeHidden,
      onToggleGroupHidden: toggleGroupHidden,
      onToggleInstanceHidden: toggleInstanceHidden,
      onShowAllHidden: showAllHidden,
      onPickNode: (id) => { setSelectedInstanceId(null); setSelectedEdgeId(null); setSelectedBound(null); setSelectedId(id); },
      selectedInstance, selectedInstanceMembers, selectedInstanceInfo, selectedInstanceSchema,
      onReschema: reschemaInstance,
      selectedEdge, selectedEdgeTypes: selectedEdgeId ? edgeTypes[selectedEdgeId] : null,
      expectedRates,
      expectedOverrides, onSetExpectedHz: setExpectedHz, onSaveExpectedHz: saveExpectedHz, expectedSaved,
      onSetEdgePolicy: setEdgePolicy,
      onDeleteEdge: deleteEdge,
      selectedBound,
      selectedBoundNode: selectedBound ? nodes.find((n) => n.id === selectedBound.nodeId) || null : null,
      locked,
      tapEdgeIds, tapSamples, onToggleTap: toggleTap, onClearTaps: clearTaps, onTapAll: tapAll,
      runState,
      ctrl, onTogglePause: togglePause, onStep: postStep,
      historyByEdge, scrubTime, onSetScrubTime: setScrubTime,
    })),
    React.createElement(TimelinePanel, {
      historyByEdge, tapEdgeIds, scrubTime, onSetScrubTime: setScrubTime, onPostReplay: postReplay,
    }),
    runPanelOpen && React.createElement(BuilderPanel, {
      info: builderInfo,
      kwargs: builderKwargs,
      setKwargs: setBuilderKwargs,
      error: builderError,
      runState, logRef,
      onLogScroll, onDownloadLog,
      stopRequested,
      onBuild: buildGraph,
      onDebug: () => playGraph(true),
      onRun: () => playGraph(false),
      onPlay: () => postController(true),
      onStop: () => stopApp(false),
      onKill: () => stopApp(true),
      onClose: () => setRunPanelOpen(false),
      // Master-clock Play/Pause + Step. Pressing Play resumes the master
      // clock (and every subordinate timer with it); Pause stops it; Step
      // advances by ``min(period)`` over the registered subordinate timers
      // (default group = all of them). The high-level MPC controller toggle
      // lives on the per-edge tap menu and is independent.
      ctrl, onTogglePause: togglePause, onStep: postStep,
      stepTimers, onToggleStepTimer: toggleStepTimer,
    }),
  );
}

function TimelinePanel({ historyByEdge, tapEdgeIds, scrubTime, onSetScrubTime, onPostReplay }) {
  // Aggregate the global time range across all tapped edges' histories.
  const { tMin, tMax, totalCount, perEdgeCount } = useMemo(() => {
    let lo = Infinity, hi = -Infinity, total = 0;
    const per = {};
    tapEdgeIds.forEach((id) => {
      const h = historyByEdge[id] || [];
      per[id] = h.length;
      total += h.length;
      if (h.length) {
        if (h[0].ts < lo) lo = h[0].ts;
        if (h[h.length - 1].ts > hi) hi = h[h.length - 1].ts;
      }
    });
    return { tMin: lo, tMax: hi, totalCount: total, perEdgeCount: per };
  }, [historyByEdge, tapEdgeIds]);

  const hasData = isFinite(tMin) && isFinite(tMax) && tMax > tMin;
  const isLive = scrubTime == null;
  const cur = isLive ? tMax : Math.max(tMin, Math.min(tMax, scrubTime));
  const fmt = (t) => isFinite(t) ? new Date(t * 1000).toLocaleTimeString([], { hour12: false }) + "." + String(Math.floor((t % 1) * 1000)).padStart(3, "0") : "—";
  const span = hasData ? (tMax - tMin) : 0;
  const STEPS = 1000; // slider granularity
  const sliderVal = hasData ? Math.round(((cur - tMin) / span) * STEPS) : STEPS;

  return React.createElement("div", { className: "timeline-panel" },
    React.createElement("div", { className: "timeline-row" },
      React.createElement("strong", { style: { fontSize: 11 } }, "⏱ timeline"),
      React.createElement("span", { className: "dim mono", style: { fontSize: 10 } },
        hasData ? `${fmt(tMin)} → ${fmt(tMax)}` : "(waiting for samples…)"),
      React.createElement("span", { style: { marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" } },
        React.createElement("span", { className: "mono", style: { fontSize: 11 } },
          isLive ? "● LIVE" : `@ ${fmt(cur)}`),
        React.createElement("button", {
          onClick: () => { onSetScrubTime(null); onPostReplay && onPostReplay(null); },
          disabled: isLive,
          title: "snap back to live",
          style: { fontSize: 11, padding: "2px 8px" },
        }, "live"),
      ),
    ),
    React.createElement("input", {
      type: "range", min: 0, max: STEPS, value: sliderVal,
      disabled: !hasData,
      onChange: (e) => {
        const v = parseInt(e.target.value, 10);
        if (v >= STEPS) { onSetScrubTime(null); onPostReplay && onPostReplay(null); return; }
        const ts = tMin + (v / STEPS) * span;
        onSetScrubTime(ts);
        onPostReplay && onPostReplay(ts);
      },
      style: { width: "100%" },
    }),
    React.createElement("div", { className: "timeline-row dim", style: { fontSize: 10 } },
      `${tapEdgeIds.size} tapped edge${tapEdgeIds.size === 1 ? "" : "s"}, ${totalCount} samples — scrub to inspect past values (live system is unaffected)`,
    ),
  );
}

// --- live tap renderers ------------------------------------------------------
function LiveValue({ sample }) {
  if (!sample) {
    return React.createElement("div", { className: "live-empty" }, "no sample yet");
  }
  if (sample.value == null) {
    return React.createElement("div", { className: "live-empty" },
      "value not captured — stats-only Run. Use Debug (or open the mcap) to see the data.");
  }
  const v = sample.value || {};
  const ts = sample.ts ? new Date(sample.ts * 1000).toLocaleTimeString() : "";
  return React.createElement(
    "div",
    { className: "live-card" },
    React.createElement("div", { className: "live-head" },
      React.createElement("span", { className: "live-kind" }, v.kind || "?"),
      React.createElement("span", { className: "live-ts dim" }, ts)),
    renderValueByKind(v),
  );
}

function renderValueByKind(v) {
  switch (v && v.kind) {
    case "image":  return React.createElement(ImageValue,  { v });
    case "vector": return React.createElement(VectorValue, { v });
    case "scalar": return React.createElement("div", { className: "live-scalar mono" }, String(v.value));
    case "text":   return React.createElement("pre",  { className: "live-text mono" }, v.value);
    case "bytes":  return React.createElement("div", { className: "live-text mono" }, `${v.size} bytes`);
    case "json":   return React.createElement("pre", { className: "live-text mono" }, JSON.stringify(v.value, null, 2));
    case "struct": return React.createElement(StructValue, { v });
    case "ndarray":return React.createElement("div", { className: "live-text mono" },
                     `ndarray ${(v.shape || []).join("×")} ${v.dtype}  min=${v.stats?.min?.toFixed?.(3)} max=${v.stats?.max?.toFixed?.(3)}`);
    default:       return React.createElement("pre", { className: "live-text mono" }, JSON.stringify(v, null, 2));
  }
}

function StructValue({ v }) {
  const fields = v.fields || {};
  const rows = Object.entries(fields).map(([k, fv]) =>
    React.createElement("div", { className: "live-struct-row", key: k },
      React.createElement("div", { className: "live-struct-key mono" }, k),
      React.createElement("div", { className: "live-struct-val" }, renderValueByKind(fv)),
    )
  );
  return React.createElement("div", { className: "live-struct" },
    React.createElement("div", { className: "dim live-meta mono" }, v.type || "struct"),
    ...rows,
  );
}

function ImageValue({ v }) {
  const src = v.png ? `data:image/png;base64,${v.png}` : null;
  return React.createElement(
    "div", { className: "live-img-wrap" },
    src
      ? React.createElement("img", { key: src.length + ":" + (src.slice(-32) || ""), src, alt: "live", className: "live-img" })
      : React.createElement("div", { className: "dim" }, "no image data"),
    React.createElement("div", { className: "dim live-meta mono" },
      `${(v.shape || []).join("×")} ${v.dtype}`),
  );
}

function VectorValue({ v }) {
  const ref = useRef(null);
  useEffect(() => {
    const cvs = ref.current;
    if (!cvs) return;
    const values = v.values || [];
    const w = cvs.width = cvs.clientWidth;
    const h = cvs.height = 80;
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!values.length) return;
    let lo = Infinity, hi = -Infinity;
    values.forEach((x) => { if (x < lo) lo = x; if (x > hi) hi = x; });
    if (lo === hi) { lo -= 1; hi += 1; }
    const scale = (y) => h - ((y - lo) / (hi - lo)) * (h - 4) - 2;
    ctx.strokeStyle = "#1e88e5";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    values.forEach((y, i) => {
      const px = (i / Math.max(1, values.length - 1)) * (w - 2) + 1;
      const py = scale(y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }, [v]);
  const stats = v.stats || {};
  return React.createElement(
    "div", { className: "live-vec-wrap" },
    React.createElement("canvas", { ref, className: "live-vec" }),
    React.createElement("div", { className: "dim live-meta mono" },
      `n=${stats.n ?? (v.values || []).length}  min=${(stats.min ?? 0).toFixed(3)}  max=${(stats.max ?? 0).toFixed(3)}  mean=${(stats.mean ?? 0).toFixed(3)}`),
  );
}

// --- run / play panel -------------------------------------------------------
// Classify a console line by log level / traceback so the run-log can color it.
// Matches Python logging's uppercase level names + common exception / traceback
// markers; ordinary output returns "" (default light text).
function stripAnsi(line) {
  // Remove ANSI SGR color escapes (e.g. "\x1b[35m", "\x1b[0m"). Some pipes drop
  // the leading ESC byte, leaving a bare "[35m", so the ESC is optional. Without
  // this, a colored prefix like "\x1b[35mWARNING:" puts a letter ("m") right
  // before WARNING and breaks the level regex, misclassifying it as an error.
  return (line || "").replace(/\x1b?\[[0-9;]*m/g, "");
}

function logLineClass(line) {
  const s = stripAnsi(line);
  // Explicit log-level markers win over the content heuristic below, so a
  // WARNING line that merely mentions "Error"/"Exception" (e.g. "WARNING:
  // retrying after ConnectionError") stays yellow instead of turning red.
  if (/(^|[^A-Za-z])(WARN|WARNING)([^A-Za-z]|$)/.test(s)) return "log-warn";
  if (/(^|[^A-Za-z])(DEBUG)([^A-Za-z]|$)/.test(s)) return "log-debug";
  if (/(^|[^A-Za-z])(ERROR|CRITICAL|FATAL)([^A-Za-z]|$)|Traceback \(most recent call last\)|[A-Za-z_.]*(Error|Exception)\b/.test(s)) return "log-error";
  return "";
}

// Debug + Paused: choose which subordinate timers Step advances to. Step runs
// until one selected timer fires, then re-pauses. Empty selection = every timer
// (step to the next tick of anything). ``periods`` is {name: period_ns}.
function StepTimerPicker({ periods, selected, onToggle }) {
  const names = Object.keys(periods || {}).sort();
  const fmtHz = (ns) => {
    if (!ns || ns <= 0) return "";
    const hz = 1e9 / ns;
    return hz >= 1 ? `${hz.toFixed(hz >= 10 ? 0 : 1)} Hz` : `${(hz * 1000).toFixed(0)} mHz`;
  };
  return React.createElement(
    "div", { className: "step-timers" },
    React.createElement("div", { className: "step-timers-head" },
      React.createElement("span", null, "Step timers"),
      React.createElement("span", { className: "dim", style: { fontSize: 10, fontWeight: 400 } },
        selected.size ? `Step runs until one of ${selected.size} fires` : "Step runs to the next tick of any timer")),
    names.length === 0
      ? React.createElement("div", { className: "dim", style: { fontSize: 11 } }, "no timers registered yet…")
      : React.createElement("div", { className: "step-timers-list" },
          names.map((name) => React.createElement("label", { key: name, className: "step-timer-item", title: name },
            React.createElement("input", {
              type: "checkbox",
              checked: selected.has(name),
              onChange: () => onToggle(name),
            }),
            React.createElement("span", { className: "step-timer-name mono" }, name),
            React.createElement("span", { className: "dim step-timer-hz" }, fmtHz(periods[name]))))),
  );
}

function BuilderPanel({ info, kwargs, setKwargs, error, runState, logRef, onLogScroll, onDownloadLog, stopRequested, onBuild, onDebug, onRun, onPlay, onStop, onKill, ctrl, onTogglePause, onStep, stepTimers, onToggleStepTimer, onClose }) {
  const set = (k) => (v) => setKwargs((f) => ({ ...f, [k]: v }));
  const [maximized, setMaximized] = useState(false);
  // Minimized collapses the panel to just its title bar + run/play/pause/step
  // controls, hiding the kwargs form and the console.
  const [minimized, setMinimized] = useState(false);
  // Whether the advanced (non-core) params are expanded. The Run-graph window shows a handful of
  // core knobs up front (``param_meta.core_params``) and hides the rest behind this "More" toggle
  // so the panel stays compact unless the user asks for everything (or maximizes).
  const [showMore, setShowMore] = useState(false);
  const params = (info && info.params || []);
  // Effective "running" for UI: false once Stop/Kill was pressed, even if
  // the child is still shutting down. Inputs unlock, Play reappears.
  const uiRunning = !!runState.running && !stopRequested;
  // Vision vs SMPC is a property of the task: a vision task (task.SINGLE_ROBOT_VISION) runs the skill brain
  // and needs a skill_policy; a state-based task runs SMPC and ignores it. Gate the skill_policy
  // field accordingly so the app-level config stays self-consistent (the builder also enforces this
  // server-side). ``vision_tasks`` comes from the builder's ``param_meta``.
  const visionTasks = (info && info.meta && info.meta.vision_tasks) || [];
  const isVisionTask = visionTasks.includes(kwargs.task);
  // Backend-specific knobs and the core/advanced split come from ``param_meta``. Robot names
  // identify the unit only: a live identity can run on either hardware or a simulation backend.
  const coreParams = (info && info.meta && info.meta.core_params) || [];
  const mujocoOnlyParams = (info && info.meta && info.meta.mujoco_only_params) || [];
  const simulationOnlyParams = (info && info.meta && info.meta.simulation_only_params) || [];
  const runRobot = kwargs.robot ?? (info && info.params || []).find((p) => p.name === "robot")?.default;
  const runBackend = kwargs.backend ?? params.find((p) => p.name === "backend")?.default;
  const isHardware = runBackend === "hardware";
  const isSimulation = runBackend === "mujoco" || runBackend === "network_sim";
  // Task/backend-driven visibility (the task dictates single-vs-multi robot; the backend dictates
  // sim-link). ``policy_params`` (the single skill_policy) collapse into ONE synthetic "policy" row
  // routed by task; ``multi_robot_params`` show only for a multi-robot task; the ``sim_link_params``
  // show only for network_sim.
  const multiRobotTasks = (info && info.meta && info.meta.multi_robot_tasks) || [];
  const policyParams = (info && info.meta && info.meta.policy_params) || [];
  const multiRobotParams = (info && info.meta && info.meta.multi_robot_params) || [];
  const simLinkParams = (info && info.meta && info.meta.sim_link_params) || [];
  const simDefaultHost = (info && info.meta && info.meta.sim_default_host) || "127.0.0.1";
  const taskIsMultiRobot = multiRobotTasks.includes(kwargs.task);
  const isNetworkSim = runBackend === "network_sim";
  const isPlaceholder = (robot) => robot === "spot0" || robot === "spot1";
  const g1Robots = (info && info.meta && info.meta.g1_robots) || [];
  const showPolicy = taskIsMultiRobot || isVisionTask;
  // The one ``skill_policy`` build param the unified "policy" row writes to; the task decides whether
  // it is loaded as the multi-robot brain or the vision skill.
  const policyTargetKey = "skill_policy";
  // A param hidden by the current task/robot (never rendered). The two policy params are always
  // hidden here -- the synthetic "policy" row stands in for them.
  const isHiddenParam = (name) => {
    if (policyParams.includes(name)) return true;
    if (multiRobotParams.includes(name) && !taskIsMultiRobot) return true;
    if (name === "partner" && !taskIsMultiRobot) return true;
    if ((name === "sim_host" || simLinkParams.includes(name)) && !isNetworkSim) return true;
    return false;
  };
  const renderParam = (p) => {
    const k = p.name;
    // Unified policy row: ONE text box writing the single ``skill_policy`` param, loaded as the
    // multi-robot brain or the vision skill depending on the task.
    if (p.synthetic && k === "policy") {
      const pv = kwargs[policyTargetKey] ?? "";
      return React.createElement(
        "label", { className: "run-row", key: "policy", title: `policy \u2014 writes ${policyTargetKey} for this task` },
        React.createElement("span", { className: "run-k" }, "policy"),
        React.createElement("input", {
          className: "run-v",
          value: pv,
          disabled: uiRunning,
          placeholder: taskIsMultiRobot ? "multi-robot ONNX uri/path" : "vision skill checkpoint",
          onChange: (e) => set(policyTargetKey)(e.target.value),
        }),
      );
    }
    const val = kwargs[k];
    const common = { className: "run-v" };
    const isSkill = k === "skill_policy";
    const skillDisabled = isSkill && !isVisionTask;
    const mujocoDisabled = mujocoOnlyParams.includes(k) && runBackend !== "mujoco";
    const simulationDisabled = simulationOnlyParams.includes(k) && !isSimulation;
    const backendDisabled = mujocoDisabled || simulationDisabled;
    const disabled = uiRunning || skillDisabled || backendDisabled;
    let input;
    if (p.kind === "bool" && p.nullable) {
      input = React.createElement(
        "select",
        {
          ...common,
          value: val === true ? "on" : val === false ? "off" : "auto",
          disabled,
          onChange: (e) => set(k)(e.target.value === "auto" ? null : e.target.value === "on"),
        },
        React.createElement("option", { value: "auto" }, "Auto"),
        React.createElement("option", { value: "on" }, "On"),
        React.createElement("option", { value: "off" }, "Off"),
      );
    } else if (p.kind === "bool") {
      input = React.createElement("input", {
        type: "checkbox",
        checked: !!val,
        disabled,
        onChange: (e) => set(k)(e.target.checked),
      });
    } else if (p.choices && p.choices.length) {
      // Changing the task also clears skill_policy when the new task is state-based, so a stale skill
      // brain can't linger on an SMPC task. Backend changes preserve live robot identities.
      const onChange = k === "task"
        ? (e) => {
            const t = e.target.value;
            setKwargs((f) => {
              const next = { ...f, task: t };
              // A state-based (non-vision) task runs SMPC (no skill brain) and REQUIRES mocap to
              // observe object poses -- clear skill_policy and default use_mocap on (matches build()).
              if (!visionTasks.includes(t)) {
                next.skill_policy = "";
                next.use_mocap = true;
              } else {
                // Vision task: refill the skill policy last used for this task (persisted
                // server-side), so switching back restores your choice.
                const remembered = (info && info.skill_policies) || {};
                if (remembered[t]) next.skill_policy = remembered[t];
              }
              return next;
            });
          }
        : k === "backend"
        ? (e) => {
            const backend = e.target.value;
            setKwargs((f) => {
              const next = { ...f, backend };
              if (backend === "network_sim") {
                if (!next.sim_host) next.sim_host = simDefaultHost;
                if (taskIsMultiRobot && !next.commander_host) next.commander_host = simDefaultHost;
              } else {
                next.sim_host = "";
                if (backend === "mujoco") next.commander_host = "";
              }
              if (backend === "hardware") {
                // Only placeholders need replacement; never rewrite a selected live identity.
                for (const key of ["robot", "partner"]) {
                  if (!isPlaceholder(next[key])) continue;
                  const available = params.find((p) => p.name === key)?.choices || [];
                  next[key] = available.find((c) => c && !isPlaceholder(c) &&
                    !g1Robots.includes(c) &&
                    c !== next[key === "robot" ? "partner" : "robot"]) || "";
                }
              }
              return next;
            });
          }
        : k === "robot"
        ? (e) => setKwargs((f) => ({ ...f, robot: e.target.value,
            partner: f.partner === e.target.value ? "" : f.partner }))
        : (e) => set(k)(e.target.value);
      // Both live and placeholder identities work in simulation. Hardware excludes placeholders;
      // a partner must also differ from the primary robot (its choices are Spot-only server-side).
      const choices = k === "robot" || k === "partner"
        ? p.choices.filter((c) => (!isHardware || !isPlaceholder(c)) &&
            (k !== "partner" || c !== runRobot))
        : p.choices;
      input = React.createElement(
        "select",
        { ...common, value: val ?? "", disabled, onChange },
        k === "partner" && !choices.includes("")
          ? React.createElement("option", { value: "" }, "Select partner") : null,
        choices.map((c) => React.createElement("option", { key: String(c), value: c }, String(c)))
      );
    } else if (p.kind === "int" || p.kind === "float") {
      input = React.createElement("input", {
        ...common,
        type: "number",
        step: p.kind === "float" ? "any" : "1",
        value: val ?? "",
        disabled,
        onChange: (e) => set(k)(e.target.value === "" ? null : Number(e.target.value)),
      });
    } else if (p.kind === "str") {
      input = React.createElement("input", {
        ...common,
        value: skillDisabled ? "" : (val ?? ""),
        disabled,
        placeholder: skillDisabled ? "n/a \u2014 state-based task (SMPC)" : "",
        onChange: (e) => set(k)(e.target.value),
      });
    } else {
      // JSON fallback for list/dict/Any.
      const displayed = val === undefined || val === null ? "" :
        typeof val === "string" ? val : JSON.stringify(val);
      input = React.createElement("input", {
        ...common,
        value: displayed,
        disabled,
        title: "JSON value (string, number, list, dict, null, true/false)",
        onChange: (e) => {
          const s = e.target.value;
          if (s.trim() === "") { set(k)(null); return; }
          try { set(k)(JSON.parse(s)); } catch (_) { set(k)(s); }
        },
      });
    }
    const hint = p.annotation ? ` (${p.annotation})` : "";
    const title = isSkill && skillDisabled
      ? `${k} \u2014 only vision tasks (SINGLE_ROBOT_VISION) use a skill_policy`
      : mujocoDisabled
        ? `${k} \u2014 local mujoco backend only`
        : simulationDisabled
        ? `${k} \u2014 simulation backends only`
        : `${k}${hint}`;
    return React.createElement(
      "label", { className: "run-row" + (backendDisabled ? " is-disabled" : ""), key: k, title },
      React.createElement("span", { className: "run-k" }, k),
      input,
    );
  };
  const cmdStr = (runState.cmd || []).join(" ");
  // A Build (``pixel build //...``) also runs through the shared runner, so
  // ``runState.running`` is true while it compiles. Distinguish it from a graph
  // run by its command shape (``[pixel, "build", target]`` -> second token is
  // "build"; graph runs use "run"). During a build the graph-control buttons
  // (Play / Pause / Step) are meaningless -- only Stop / Kill apply.
  const isBuilding = (runState.cmd || [])[1] === "build";
  // Status line reflects: idle / starting (N/M nodes ready) / controller
  // off / on / stopping / exited. The high-level controller (MPC) is
  // independent of the graph's process lifetime: Run starts the subprocess
  // (which powers on the robot and stands on hardware); when every node's
  // ``pre_open`` has completed the health gate flips ``ready``, the user
  // can then press Play to start the controller; Pause stops the controller
  // but keeps the low-level command stream alive.
  // Play/Pause now toggles the *master clock* (graph-wide pause). The
  // high-level MPC controller is auto-started via STARFISH_AUTOSTART_CONTROLLER
  // and remains on; pausing the master simply stops every subordinate timer
  // (including ``controller_timer``) which naturally stops MPC from firing.
  const controllerOn = !!(ctrl && ctrl.controller_running);
  const paused = !!(ctrl && (ctrl.clock_paused || ctrl.paused));
  const health = (ctrl && ctrl.health) || { ready: true, started: 0, total: 0, pending: [], errors: [] };
  const healthErrors = Array.isArray(health.errors) ? health.errors : [];
  const hasInitErrors = healthErrors.length > 0;
  const systemReady = !!health.ready;
  const shutdownRequested = !!(ctrl && ctrl.shutdown_requested);
  const shutdownReady = !!(ctrl && ctrl.shutdown_ready);
  const startingStatus = health.total > 0
    ? (hasInitErrors
        ? `init errors (${healthErrors.length})`
        : `starting (${health.started}/${health.total} ready)`)
    : "starting";
  // Stop-button labels reflect the safe-shutdown handshake: the user
  // sees "stopping (safe-sit)" while the mujoco animation is playing,
  // then "stopping" once the runner has actually sent SIGINT.
  const stoppingLabel = shutdownRequested
    ? (shutdownReady ? "stopping (sit-down done)" : "stopping (safe-sit in progress)")
    : "stopping";
  const headStatus = runState.running
    ? (isBuilding
        ? `building… (pid ${runState.pid})`
        : runState.debug === false
        ? `running (pid ${runState.pid})`
        : stopRequested || shutdownRequested
        ? `${stoppingLabel} (pid ${runState.pid})`
        : (!systemReady
            ? `${startingStatus} (pid ${runState.pid})`
            : paused
              ? `paused (pid ${runState.pid}) — press Resume`
              : controllerOn
                ? `running (pid ${runState.pid})`
                : `running, controller off (pid ${runState.pid})`))
    : runState.exit_code !== null ? `exited ${runState.exit_code}` : "idle";
  // Backend, never robot identity, drives debugger availability. Generic builders without a
  // backend parameter retain their existing debug controls.
  const debugDisabled = runBackend !== undefined && !isSimulation;
  // Play/Pause toggles the high-level controller (MPC). The rest of the
  // graph keeps ticking either way so the joint-control stream stays
  // alive — when the controller is off SplineFollower streams a safe
  // DEFAULT_POLICY_CMD (zero velocity, standing, stowed arm). Single
  // toggle works on both mujoco and hardware. Step / Kill remain mujoco-
  // only debug controls (Step uses the global pause + epoch machinery).
  // The health gate is informational — see headStatus for the
  // "starting (N/M ready)" hint. Play is always clickable so the user
  // can arm MPC the moment they decide the system is ready (matches
  // judo's behaviour and avoids races caused by per-process health
  // aggregation lag).
  const playDisabled = false;
  const playTitle = paused
    ? "Resume the master clock (every subordinate timer starts ticking)"
    : "Pause the master clock (every subordinate timer stops)";
  const playPauseToggle = React.createElement("button", {
    key: "play",
    onClick: onTogglePause,
    disabled: playDisabled,
    title: playTitle,
    style: {
      background: paused ? "#e8f5e9" : "#fff3e0",
      opacity: playDisabled ? 0.6 : 1,
    },
  }, paused ? "\u25B6 Resume" : "\u23F8 Pause");
  const stopBtn = React.createElement("button", { key: "stop", onClick: onStop, title: "SIGINT" }, "■ Stop");
  const stepBtn = React.createElement("button", {
    key: "step",
    onClick: () => onStep && onStep(null),
    disabled: !paused,
    title: paused ? "Step: run until one selected timer fires, then pause" : "Step is only available while Paused",
  }, "⏭ Step");
  const killBtn = React.createElement("button", { key: "kill", onClick: onKill, title: "SIGKILL" }, "✕ Kill");
  // Debug run: full controls (Pause/Resume + Step). Plain Run: lightweight, so
  // Pause/Step don't apply -- only Stop / Kill.
  const debugRun = runState.debug !== false;
  const runningButtons = !debugRun
    ? [stopBtn, killBtn]
    : debugDisabled
      ? [playPauseToggle, stopBtn]
      : [playPauseToggle, stepBtn, stopBtn, killBtn];
  // While building, Play/Pause/Step don't apply -- show a disabled Pause plus
  // Stop / Kill (Stop/Kill abort the bazel build).
  const buildingPause = React.createElement("button", {
    key: "play",
    disabled: true,
    title: "Play/Pause is unavailable during Build",
    style: { background: "#fff3e0", opacity: 0.5 },
  }, "\u23F8 Pause");
  const buildingButtons = [buildingPause, stopBtn, killBtn];
  const activeButtons = isBuilding ? buildingButtons : runningButtons;
  return React.createElement(
    "div", { className: "run-panel" + (minimized ? " minimized" : (maximized ? " maximized" : "")) },
    React.createElement("div", { className: "run-head" },
      React.createElement("strong", null, "Run graph"),
      info && info.spec && React.createElement("span", { className: "dim mono", style: { marginLeft: 6, fontSize: 11 } }, info.spec),
      React.createElement("span", { className: "dim run-status", style: { marginLeft: 8 }, title: headStatus }, headStatus),
      React.createElement("span", { style: { marginLeft: "auto", display: "flex", gap: 6 } },
        !uiRunning
          ? React.createElement(React.Fragment, null,
              React.createElement("button", {
                key: "build",
                onClick: onBuild,
                title: "Bazel-build the run_graph binary (pixel build) so Run/Debug launch instantly",
                style: { background: "#ede7f6" },
              }, "\uD83D\uDD28 Build"),
              React.createElement("button", {
                key: "debug",
                onClick: onDebug,
                disabled: debugDisabled,
                title: debugDisabled
                  ? "Debug (pause/step) is simulation-only — it freezes background threads, which is unsafe on a real robot. Select a simulation backend to debug."
                  : "Run with the debugger: all edges tapped, Pause/Resume + Step, and edge-value inspection",
                style: { background: debugDisabled ? "#eee" : "#fff3e0", opacity: debugDisabled ? 0.5 : 1 },
              }, "\uD83D\uDC1E Debug"),
              React.createElement("button", {
                key: "run",
                onClick: onRun,
                title: "Lightweight run (no taps, no clock gating) — only Stop/Kill. Fast direct launch if already built.",
                style: { background: "#e8f5e9" },
              }, "\u25B6 Run"))
          : React.createElement(React.Fragment, null, ...activeButtons)),
      React.createElement("span", { className: "win-btns" },
        React.createElement("button", {
          className: "win-btn win-min",
          onClick: () => { setMaximized(false); setMinimized((m) => !m); },
          title: minimized ? "Expand panel" : "Minimize (collapse to controls)",
          "aria-label": minimized ? "Expand panel" : "Minimize panel",
        }),
        React.createElement("button", {
          className: "win-btn win-max" + (maximized ? " is-max" : ""),
          onClick: () => { setMinimized(false); setMaximized((m) => !m); },
          title: maximized ? "Restore panel size" : "Maximize panel",
          "aria-label": maximized ? "Restore panel size" : "Maximize panel",
        }),
        React.createElement("button", {
          className: "win-btn win-close",
          onClick: onClose,
          title: "Close panel (reopen from the top toolbar)",
          "aria-label": "Close panel",
        }))),
    !minimized && React.createElement(React.Fragment, null,
      // Debug fidelity warning: threads not under scheduler/pacer control won't
      // freeze on Pause (e.g. a user-spawned raw loop). Surfaced so the user
      // knows Debug can't fully stop them. Only the UNMANAGED (non-pause-aware)
      // background threads count here -- managed ones (e.g. the command stream)
      // freeze on Pause and are shown as pills but don't warrant a warning.
      (() => {
        const rogue = debugRun && runState.running
          ? (((ctrl && ctrl.health && ctrl.health.unmanaged_threads) || []).filter((t) => typeof t === "string" || !t.managed))
          : [];
        return rogue.length
          ? React.createElement("div", { className: "unmanaged-warn" },
              React.createElement("span", { className: "unmanaged-warn-icon" }, "\u26A0"),
              React.createElement("span", null,
                `${rogue.length} thread(s) run outside the scheduler and won't freeze on Pause: `,
                React.createElement("span", { className: "mono" },
                  rogue.map((t) => (typeof t === "string" ? t : t.name)).join(", ")),
                ". Debug fidelity is reduced for these — convert them to graph nodes for full pause/step control."))
          : null;
      })(),
      params.length
        ? (() => {
            // Core knobs render up front (in meta order -- TASK first), the rest hide behind a
            // "More" toggle. The two policy params collapse into ONE synthetic "policy" row, and
            // params the current task/robot hides (partner on a single-robot task, sim ports on a
            // hardware robot) are dropped entirely. Every host + port lives in "More".
            const orderNames = coreParams.length ? coreParams : params.map((p) => p.name);
            const coreNameSet = new Set(orderNames);
            const coreList = [];
            let policyPlaced = false;
            orderNames.forEach((name) => {
              if (policyParams.includes(name)) {
                if (!policyPlaced && showPolicy) {
                  coreList.push({ name: "policy", kind: "str", synthetic: true });
                  policyPlaced = true;
                }
                return;
              }
              if (isHiddenParam(name)) return;
              const p = params.find((pp) => pp.name === name);
              if (p) coreList.push(p);
            });
            const advList = params.filter((p) => !coreNameSet.has(p.name) && !isHiddenParam(p.name));
            return React.createElement(React.Fragment, null,
              React.createElement("div", { className: "run-form" }, coreList.map(renderParam)),
              advList.length ? React.createElement("button", {
                className: "run-more-toggle",
                onClick: () => setShowMore((m) => !m),
                title: showMore ? "Hide the advanced options" : "Show all advanced options",
              }, showMore ? "\u25B4 Less" : `\u25BE More (${advList.length})`) : null,
              showMore && advList.length
                ? React.createElement("div", { className: "run-form run-adv" }, advList.map(renderParam))
                : null,
            );
          })()
        : React.createElement("div", { className: "dim", style: { padding: "8px 10px" } },
            "Builder takes no arguments."),
      // Debug: pick which timers Step runs until one fires. Shown for the whole debug run (not just
      // while paused) so the panel doesn't resize every time you Pause/Step/Resume; the selection
      // just pre-configures the Step group and takes effect the next time you Step.
      debugRun && runState.running && React.createElement(StepTimerPicker, {
        periods: (ctrl && ctrl.subordinate_periods_ns) || {},
        selected: stepTimers || new Set(),
        onToggle: onToggleStepTimer,
      }),
      error && React.createElement("div", { className: "run-cmd", style: { color: "#c62828" } }, error),
      cmdStr && React.createElement("div", { className: "run-cmd mono" }, "$ " + cmdStr),
      React.createElement("div", { className: "run-log-bar" },
        React.createElement("span", { className: "dim", style: { fontSize: 11 } },
          `${(runState.lines || []).length} line${(runState.lines || []).length === 1 ? "" : "s"}`),
        React.createElement("button", {
          className: "log-dl",
          onClick: onDownloadLog,
          disabled: !((runState.lines || []).length),
          title: "Download the full console log as a text file",
        }, "\u2b73 Download log")),
      React.createElement("pre", { className: "run-log", ref: logRef, onScroll: onLogScroll },
        (runState.lines || []).map((line, i) => {
          const clean = stripAnsi(line);
          return React.createElement("div", { key: i, className: "log-line " + logLineClass(line) },
            clean && clean.length ? clean : "\u00a0");
        })),
    ),
  );
}

// --- kwargs / bound-inputs editors (used by Inspector) ----------------------
// Renders a small JSON value as a one-liner. Strings are unquoted in the input
// so users can paste values like 0.5 / "spot" / [1,2] / true and we serialize
// them as JSON on save (with a string fallback so plain text doesn't reject).
function _jsonValueToInput(v) {
  if (v === undefined) return "";
  if (typeof v === "string") return JSON.stringify(v);
  try { return JSON.stringify(v); } catch (_) { return String(v); }
}
function _inputToJsonValue(s) {
  const t = (s || "").trim();
  if (t === "") return "";
  try { return JSON.parse(t); } catch (_) { return s; }
}

function KwargsEditor({ kwargs, onChange, disabled }) {
  const entries = Object.entries(kwargs || {});
  const update = (newEntries) => {
    const out = {};
    newEntries.forEach(([k, v]) => { if (k) out[k] = v; });
    onChange(out);
  };
  const rows = entries.map(([k, v], i) =>
    React.createElement(
      "div",
      { key: `kw-${i}`, className: "kv" },
      React.createElement("input", {
        className: "k",
        defaultValue: k,
        disabled,
        style: { padding: "2px 4px" },
        onBlur: (e) => {
          const next = entries.slice();
          next[i] = [e.target.value, v];
          update(next);
        },
      }),
      React.createElement(
        "span",
        { className: "v" },
        React.createElement("input", {
          className: "mono",
          defaultValue: _jsonValueToInput(v),
          disabled,
          style: { padding: "2px 4px", flex: 1 },
          onBlur: (e) => {
            const next = entries.slice();
            next[i] = [k, _inputToJsonValue(e.target.value)];
            update(next);
          },
        }),
        React.createElement(
          "button",
          {
            disabled,
            title: "remove",
            onClick: () => {
              const next = entries.slice();
              next.splice(i, 1);
              update(next);
            },
          },
          "×"
        )
      )
    )
  );
  rows.push(
    React.createElement(
      "div",
      { key: "kw-add", className: "kv" },
      React.createElement("span", { className: "k dim" }, "+ add"),
      React.createElement(
        "span",
        { className: "v" },
        React.createElement("button", {
          disabled,
          onClick: () => {
            const name = window.prompt("kwarg name?");
            if (!name) return;
            update([...entries, [name, ""]]);
          },
        }, "add kwarg")
      )
    )
  );
  if (!entries.length) {
    rows.unshift(React.createElement("div", { key: "kw-empty", className: "hint" }, "(none)"));
  }
  return React.createElement("div", { className: "section" }, rows);
}

// Coerce a raw input string to the typed value for a schema field ``kind``.
// Blank (after trim) -> ``undefined`` for every kind except bool, letting the
// caller decide whether that means "drop the key / use baseline" (group child)
// or "reset to the field default" (top-level).
function _coerceSchemaInput(kind, raw) {
  if (kind === "bool") return !!raw;
  const t = (raw == null ? "" : String(raw)).trim();
  if (t === "") return undefined;
  if (kind === "int") { const n = parseInt(t, 10); return Number.isNaN(n) ? undefined : n; }
  if (kind === "float") { const n = parseFloat(t); return Number.isNaN(n) ? undefined : n; }
  if (kind === "str" || kind === "enum") return raw;
  return _inputToJsonValue(raw); // json
}

// One typed control for a single schema field. ``values`` is the current kwargs
// map at this level; ``onSet(name, value)`` writes it (``value === undefined``
// asks the parent to drop the key, i.e. fall back to the server/task baseline).
// ``groupChild`` distinguishes a nested sub-field (blank => unset) from a
// top-level field (blank => reset to default, so the key always stays present).
function SchemaField({ field, values, onSet, disabled, groupChild, path }) {
  const name = field.name;
  const key = `${path}.${name}`;
  const owned = values && Object.prototype.hasOwnProperty.call(values, name);
  const ownVal = owned ? values[name] : undefined;
  const cur = owned ? ownVal : field.default;
  // App-level fields (task / robot / connection / use_mocap, and SMPC's task_name / spot_id /
  // requires_mocap) are owned by the Run-graph window and propagate to every instance on rebuild,
  // so they render read-only here -- editing them per-instance would let the diagram's instances
  // disagree (and a rebuild would overwrite the edit anyway).
  const appLevel = !!field.app_level;
  const dis = disabled || appLevel;

  if (field.kind === "group") {
    const groupVals = (ownVal && typeof ownVal === "object" && !Array.isArray(ownVal)) ? ownVal : {};
    const setChild = (childName, childVal) => {
      const next = { ...groupVals };
      if (childVal === undefined) delete next[childName];
      else next[childName] = childVal;
      onSet(name, next); // keep the (possibly empty) map so the factory always gets it
    };
    const sub = (field.fields || []).map((f) =>
      React.createElement(SchemaField, {
        key: `${key}.${f.name}`, field: f, values: groupVals,
        onSet: setChild, disabled, groupChild: true, path: key,
      }));
    return React.createElement(
      "div", { className: "subsection", key },
      React.createElement("div", { className: "subhead", title: field.doc || "" }, name),
      sub.length ? sub : React.createElement("div", { className: "hint" }, "(no fields)")
    );
  }

  let control;
  if (field.kind === "bool") {
    control = React.createElement("input", {
      type: "checkbox", checked: !!cur, disabled: dis,
      onChange: (e) => onSet(name, e.target.checked),
    });
  } else if (field.kind === "enum") {
    const opts = (field.choices || []).map((c) =>
      React.createElement("option", { key: String(c), value: String(c) }, String(c)));
    control = React.createElement(
      "select",
      { value: cur == null ? "" : String(cur), disabled: dis, style: { flex: 1 },
        onChange: (e) => onSet(name, e.target.value) },
      (cur == null ? [React.createElement("option", { key: "__none", value: "" }, "\u2014")] : []).concat(opts)
    );
  } else {
    const isNum = field.kind === "int" || field.kind === "float";
    const dv = groupChild
      ? (owned ? (field.kind === "json" ? _jsonValueToInput(ownVal) : String(ownVal)) : "")
      : (cur == null ? "" : (field.kind === "json" ? _jsonValueToInput(cur) : String(cur)));
    control = React.createElement("input", {
      className: field.kind === "json" ? "mono" : undefined,
      type: isNum ? "number" : "text",
      step: field.kind === "int" ? "1" : (field.kind === "float" ? "any" : undefined),
      defaultValue: dv,
      placeholder: field.default == null ? "" : String(field.default),
      disabled: dis,
      style: { padding: "2px 4px", flex: 1 },
      onBlur: (e) => {
        const v = _coerceSchemaInput(field.kind, e.target.value);
        if (v === undefined) onSet(name, groupChild ? undefined : field.default);
        else onSet(name, v);
      },
    });
  }
  return React.createElement("div", { className: "kv", key },
    React.createElement("span", { className: "k", title: appLevel ? "set in the Run graph window" : (field.doc || "") },
      name,
      appLevel ? React.createElement("span", { className: "dim", style: { marginLeft: 4, fontSize: 10 } }, "\u00b7 app") : null),
    React.createElement("span", { className: "v" }, control));
}

// Schema-driven config form: renders one typed control per field (checkbox /
// number / text / dropdown / nested group), reading current values from the
// instance's kwargs and emitting a merged kwargs map on every edit. When an
// ``affects_schema`` field changes (e.g. SMPC ``optimizer_name``) it also asks
// the caller to re-request the schema so dependent sub-forms update.
function SchemaForm({ fields, values, onChange, onReschema, disabled }) {
  fields = fields || [];
  const vals = values || {};
  if (!fields.length) {
    return React.createElement("div", { className: "hint" }, "(no schema fields)");
  }
  const setTop = (name, value) => {
    const next = { ...vals };
    if (value === undefined) delete next[name];
    else next[name] = value;
    onChange(next);
    const fld = fields.find((f) => f.name === name);
    if (fld && fld.affects_schema && onReschema) onReschema(next);
  };
  const hasAppLevel = fields.some((f) => f.app_level);
  return React.createElement(
    "div", { className: "section" },
    hasAppLevel
      ? React.createElement("div", { className: "hint", key: "__applevel" },
          "\u00b7 app fields are set in the Run graph window and propagate to every instance")
      : null,
    fields.map((f) =>
      React.createElement(SchemaField, {
        key: f.name, field: f, values: vals, onSet: setTop,
        disabled, groupChild: false, path: "root",
      }))
  );
}

// Bound-input editor: list each input port; bound ports get an editable JSON
// value, unbound ports get a "+ bind" button. The connected-ports set is
// optional (caller may pre-compute) and disables binding for ports already
// driven by a real edge.
function BoundInputsEditor({ inputs, bound, connectedPorts, onChange, disabled }) {
  inputs = inputs || [];
  bound = bound || {};
  const setBound = (port, value) => {
    const next = { ...bound, [port]: value };
    onChange(next);
  };
  const unbind = (port) => {
    const next = { ...bound };
    delete next[port];
    onChange(next);
  };
  if (!inputs.length) {
    return React.createElement("div", { className: "hint" }, "(no input ports)");
  }
  return React.createElement(
    "div", { className: "section" },
    inputs.map((port) => {
      const isBound = Object.prototype.hasOwnProperty.call(bound, port);
      const isConnected = connectedPorts && connectedPorts.has(port);
      const valueEl = isBound
        ? React.createElement(
            "span", { className: "v" },
            React.createElement("input", {
              className: "mono",
              defaultValue: _jsonValueToInput(bound[port]),
              disabled,
              style: { padding: "2px 4px", flex: 1 },
              onBlur: (e) => setBound(port, _inputToJsonValue(e.target.value)),
            }),
            React.createElement("button", {
              disabled,
              title: "unbind",
              onClick: () => unbind(port),
            }, "×")
          )
        : React.createElement(
            "span", { className: "v" },
            isConnected
              ? React.createElement("span", { className: "dim" }, "(connected)")
              : React.createElement("button", {
                  disabled,
                  onClick: () => setBound(port, ""),
                }, "+ bind"),
          );
      return React.createElement("div", { key: `bi-${port}`, className: "kv" },
        React.createElement("span", { className: "k mono" }, port),
        valueEl);
    })
  );
}


// --- right-side inspector ---------------------------------------------------
function Inspector({
  selected, allProcesses, allInstances, instances,
  onAssignProcess, onAssignInstance, onUpdateSelected,
  onUpdateSelectedKwargs, onUpdateSelectedBoundInputs, onUpdateInstanceKwargs,
  selectedConnectedPorts,
  grouped, groupBy, onPickNode,
  hiddenNodeIds, hiddenInstanceIds, onToggleNodeHidden, onToggleGroupHidden, onToggleInstanceHidden, onShowAllHidden,
  selectedInstance, selectedInstanceMembers, selectedInstanceInfo, selectedInstanceSchema, onReschema,
  selectedEdge, selectedEdgeTypes, onSetEdgePolicy, onDeleteEdge, expectedRates,
  expectedOverrides, onSetExpectedHz, onSaveExpectedHz, expectedSaved,
  selectedBound, selectedBoundNode,
  locked, tapEdgeIds, tapSamples, onToggleTap, onClearTaps, onTapAll, runState,
  ctrl, onTogglePause, onStep,
  historyByEdge, scrubTime, onSetScrubTime,
}) {
  // Match the graph's per-group colors: process/instance groupings reuse the
  // same collision-free palette assignment as the node badges and boxes.
  const groupColors = (groupBy === "process" || groupBy === "instance")
    ? distinctColorMap(Object.keys(grouped))
    : null;
  // Auto-hide vs pin: pinned keeps the panel always visible; unpinned collapses
  // it to a hover grip that reveals on mouse-over. Default pinned (preserves the
  // always-visible behavior); the choice is persisted so it sticks across loads.
  const [pinned, setPinned] = useState(() => {
    try { return localStorage.getItem("inspectorPinned") !== "0"; } catch (e) { return true; }
  });
  const [hovered, setHovered] = useState(false);
  const collapsed = !pinned && !hovered;
  const togglePinned = () => {
    setPinned((prev) => {
      const next = !prev;
      try { localStorage.setItem("inspectorPinned", next ? "1" : "0"); } catch (e) { /* ignore */ }
      return next;
    });
  };
  return React.createElement(
    "div",
    {
      className: "inspector" + (collapsed ? " collapsed" : ""),
      onMouseEnter: () => setHovered(true),
      onMouseLeave: () => setHovered(false),
    },
    React.createElement("div", { className: "grip", title: "hover to reveal · pin to keep open" }, "Inspector"),
    React.createElement(
      "div",
      { className: "inspector-head" },
      React.createElement("button", {
        className: "pin-btn" + (pinned ? " pinned" : ""),
        title: pinned ? "Pinned (always show) — click to auto-hide" : "Auto-hide — click to pin",
        "aria-label": pinned ? "Unpin inspector" : "Pin inspector",
        onClick: (e) => { e.stopPropagation(); togglePinned(); },
      }, "📌"),
      React.createElement("h3", null, "Inspector")
    ),
    selected
      ? React.createElement(
          "div",
          { className: "section" },
          kv("id",         React.createElement("span", { className: "v mono" },
            React.createElement("span", { className: "lbl" }, selected.id),
            React.createElement(EyeToggle, {
              hidden: hiddenNodeIds.has(selected.id),
              onToggle: () => onToggleNodeHidden(selected.id),
              title: hiddenNodeIds.has(selected.id) ? "show this node" : "hide this node",
            }))),
          kv("target",     React.createElement("span", { className: "v mono", title: selected.data.target }, shortTarget(selected.data.target))),
          kv("pre_open",   React.createElement("span", { className: "v mono" }, shortTarget(selected.data.pre_open) || "—")),
          kv("post_close", React.createElement("span", { className: "v mono" }, shortTarget(selected.data.post_close) || "—")),
          kv("process",    React.createElement(EditableSelect, {
            value: selected.data.process || "", options: allProcesses,
            placeholder: "(none)", onChange: onAssignProcess,
          })),
          kv("instance",   React.createElement(EditableSelect, {
            value: selected.data.instance || "", options: allInstances,
            placeholder: "(none)", onChange: onAssignInstance,
          })),
          // Editable kwargs for this node call site (separate from the
          // shared-instance ctor kwargs below). ``key`` is scoped to the
          // selected node id so the inputs (which use ``defaultValue``)
          // remount when a different node is clicked -- otherwise React
          // keeps the previous node's text in the boxes.
          React.createElement(
            "div",
            { className: "subsection", key: `node-kwargs-${selected.id}` },
            React.createElement("div", { className: "subhead" }, "kwargs"),
            React.createElement(KwargsEditor, {
              kwargs: selected.data.kwargs || {},
              onChange: onUpdateSelectedKwargs,
              disabled: locked,
            })
          ),
          // Editable bound_inputs for this node (constants wired to input ports).
          React.createElement(
            "div",
            { className: "subsection", key: `node-bound-${selected.id}` },
            React.createElement("div", { className: "subhead" }, "bound inputs"),
            React.createElement(BoundInputsEditor, {
              inputs: selected.data.inputs || [],
              bound: selected.data.bound_inputs || {},
              connectedPorts: selectedConnectedPorts || new Set(),
              onChange: onUpdateSelectedBoundInputs,
              disabled: locked,
            })
          ),
          selected.data.instance && instances[selected.data.instance] &&
            React.createElement(
              "div",
              { className: "subsection", key: `inst-${selected.data.instance}` },
              React.createElement("div", { className: "subhead" }, "shared instance state"),
              kv("factory", React.createElement("span", { className: "v mono" }, instances[selected.data.instance].factory)),
              React.createElement("div", { style: { marginTop: 4 } },
                React.createElement("div", { className: "k", style: { marginBottom: 2 } }, "ctor kwargs"),
                React.createElement(KwargsEditor, {
                  kwargs: instances[selected.data.instance].kwargs || {},
                  onChange: (kw) => onUpdateInstanceKwargs(selected.data.instance, kw),
                  disabled: locked,
                })
              )
            )
        )
      : selectedInstance
        ? renderInstancePanel(selectedInstance, selectedInstanceMembers, selectedInstanceInfo, selectedInstanceSchema, onPickNode, onUpdateInstanceKwargs, onReschema, locked, hiddenNodeIds, onToggleNodeHidden, onToggleGroupHidden, hiddenInstanceIds, onToggleInstanceHidden)
        : selectedEdge
          ? renderEdgePanel(
              selectedEdge, selectedEdgeTypes, onSetEdgePolicy, onPickNode, onDeleteEdge,
              {
                locked,
                tapped: tapEdgeIds.has(selectedEdge.id),
                // Prefer the live poll sample, but fall back to the newest buffered history entry.
                // History is fetched by its own reliable per-edge effect (``/tap/history?edge=``),
                // so this guarantees flow stats keep updating for switched/re-tapped edges even if
                // the shared /tap/latest poll lags on a subscription change.
                sample: tapSamples[selectedEdge.id]
                  || (historyByEdge[selectedEdge.id] && historyByEdge[selectedEdge.id].length
                        ? historyByEdge[selectedEdge.id][historyByEdge[selectedEdge.id].length - 1]
                        : null),
                running: !!(runState && runState.running),
                history: historyByEdge[selectedEdge.id] || [],
                expectedKey: edgeKey(selectedEdge),
                expectedDerived: expectedRates ? expectedRates[selectedEdge.source] || null : null,
                expectedOverride: expectedOverrides ? expectedOverrides[edgeKey(selectedEdge)] ?? null : null,
                expectedHz: (expectedOverrides && expectedOverrides[edgeKey(selectedEdge)])
                  || (expectedRates ? expectedRates[selectedEdge.source] || null : null),
                onSetExpectedHz, onSaveExpectedHz, expectedSaved,
                scrubTime,
                onSetScrubTime,
                onStepNode: onStep,
              },
            )
          : selectedBound
            ? renderBoundPanel(selectedBound, selectedBoundNode, onPickNode)
            : React.createElement("div", { className: "hint" }, "click a node, edge, or instance container to inspect"),

    React.createElement("hr", null),
    React.createElement(
      "h4",
      { style: { display: "flex", alignItems: "center", gap: 6 } },
      `groups by ${groupBy}`,
      hiddenNodeIds && hiddenNodeIds.size > 0 && React.createElement(
        "button",
        { className: "show-all", title: "show every hidden node", onClick: onShowAllHidden },
        `show all (${hiddenNodeIds.size} hidden)`
      )
    ),
    React.createElement(
      "div",
      { className: "groups" },
      Object.entries(grouped).sort((a, b) => a[0].localeCompare(b[0])).map(([key, members]) => {
        const memberIds = members.map((m) => m.id);
        // For instance groups the eye is memorized (survives rebuilds); other
        // groupings use the plain view-only toggle. A group counts as hidden
        // when every member is hidden OR the instance is memorized-hidden.
        const isInstanceGroup = groupBy === "instance";
        const memoHidden = isInstanceGroup && hiddenInstanceIds && hiddenInstanceIds.has(key);
        const allHidden = memoHidden || (memberIds.length > 0 && memberIds.every((id) => hiddenNodeIds.has(id)));
        const onGroupToggle = isInstanceGroup && onToggleInstanceHidden
          ? () => onToggleInstanceHidden(key, memberIds)
          : () => onToggleGroupHidden(memberIds);
        return React.createElement(
          "details",
          { key, open: groupBy !== "none" && members.length > 1 },
          React.createElement(
            "summary",
            { className: allHidden ? "group-hidden" : undefined },
            React.createElement(EyeToggle, {
              hidden: allHidden,
              onToggle: onGroupToggle,
              title: allHidden ? `show all in ${key}` : `hide all in ${key}`,
              className: "eye-group",
            }),
            React.createElement("span", {
              className: "swatch",
              style: { background: (groupColors ? groupColors[key] : hashColor(key)) || "#ccc" },
            }),
            ` ${groupBy === "target" ? shortTarget(key) : key} `,
            React.createElement("span", { className: "count" }, `(${members.length})`)
          ),
          React.createElement(
            "ul",
            null,
            members.map((m) => {
              const isHidden = hiddenNodeIds.has(m.id);
              return React.createElement(
                "li",
                {
                  key: m.id,
                  onClick: () => onPickNode(m.id),
                  className: (selected && selected.id === m.id ? "active" : "") + (isHidden ? " hidden-row" : ""),
                },
                React.createElement("span", { className: "lbl" }, m.id),
                React.createElement(EyeToggle, {
                  hidden: isHidden,
                  onToggle: () => onToggleNodeHidden(m.id),
                  title: isHidden ? "show node" : "hide node",
                })
              );
            })
          )
        );
      })
    )
  );
}

function kv(label, valueEl) {
  return React.createElement(
    "div",
    { className: "kv", key: label },
    React.createElement("span", { className: "k" }, label),
    valueEl
  );
}

// Small eye button used across the inspector to hide/unhide a node or a whole
// group / instance. ``hidden`` renders the eye "off" (dimmed + red slash via CSS);
// clicking stops propagation so it never also selects the row or toggles a
// <details>. The hide state itself lives in ``App``'s ``hiddenNodeIds``.
function EyeToggle({ hidden, onToggle, title, className }) {
  return React.createElement(
    "button",
    {
      className: "eye" + (hidden ? " off" : "") + (className ? " " + className : ""),
      title: title || (hidden ? "show" : "hide"),
      "aria-label": title || (hidden ? "show" : "hide"),
      onClick: (e) => { e.stopPropagation(); e.preventDefault(); onToggle(); },
    },
    "👁"
  );
}

function openTargetInVSCode(target) {
  if (!target) return;
  fetch(`/resolve?target=${encodeURIComponent(target)}`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then(({ file, line }) => { window.location.href = `vscode://file/${file}:${line}`; })
    .catch((err) => console.error("resolve", target, err));
}

function openFileInVSCode(file, line) {
  if (!file) return;
  window.location.href = `vscode://file/${file}:${line || 1}`;
}

function EdgeTooltip({ x, y, info }) {
  const style = {
    position: "fixed",
    left: x + 12,
    top: y + 12,
    background: "rgba(33, 33, 33, 0.95)",
    color: "#fff",
    padding: "6px 8px",
    borderRadius: 6,
    fontSize: 11,
    pointerEvents: "none",
    zIndex: 2000,
    maxWidth: 360,
    boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
  };
  if (info.loading) {
    return React.createElement("div", { style }, "resolving type…");
  }
  const lines = [];
  if (info.in) {
    const ann = info.in.annotation || "(no annotation)";
    const nm = info.in.name ? `${info.in.name}: ` : "";
    lines.push(
      React.createElement("div", { key: "in", style: { fontFamily: "ui-monospace, monospace" } },
        React.createElement("span", { style: { color: "#90caf9" } }, "→ "),
        React.createElement("span", { style: { color: "#bbb" } }, info.in.port + " "),
        React.createElement("span", null, nm + ann)
      )
    );
  }
  if (info.out) {
    const ann = info.out.annotation || "(no annotation)";
    lines.push(
      React.createElement("div", { key: "out", style: { fontFamily: "ui-monospace, monospace" } },
        React.createElement("span", { style: { color: "#a5d6a7" } }, "← "),
        React.createElement("span", { style: { color: "#bbb" } }, info.out.port + " "),
        React.createElement("span", null, ann)
      )
    );
  }
  if (!lines.length) {
    return React.createElement("div", { style }, "no type info");
  }
  return React.createElement("div", { style }, lines);
}

function renderRefPills(refs) {
  if (!refs || !refs.length) return null;
  return React.createElement(
    "span",
    { className: "ref-pills" },
    refs.map((r) =>
      React.createElement(
        "span",
        {
          key: r.target,
          className: "ref-pill mono",
          title: `double-click to open ${r.target} in VS Code`,
          onDoubleClick: (e) => { e.stopPropagation(); openTargetInVSCode(r.target); },
        },
        r.name
      )
    )
  );
}

function BoundPanel({ sel, node, onPickNode }) {
  // sel = { nodeId, port }
  const value = node && node.data && node.data.bound_inputs ? node.data.bound_inputs[sel.port] : undefined;
  const inputs = (node && node.data && node.data.inputs) || [];
  const target = node && node.data && node.data.target;
  const idx = inputs.indexOf(sel.port);
  const [typeInfo, setTypeInfo] = useState({ loading: true });
  useEffect(() => {
    if (!target || idx < 0) { setTypeInfo({ name: "—" }); return; }
    setTypeInfo({ loading: true });
    fetch(`/port_type?target=${encodeURIComponent(target)}&side=in&index=${idx}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((info) => setTypeInfo(info || { name: "—" }))
      .catch((err) => setTypeInfo({ name: "—", error: String(err) }));
  }, [target, idx]);
  const typeEl = typeInfo.loading
    ? React.createElement("span", { className: "v mono", style: { opacity: 0.6 } }, "…")
    : React.createElement("span", { className: "v" },
        React.createElement("span", { className: "mono", style: { marginRight: 6 } }, typeInfo.name || "—"),
        renderRefPills(typeInfo.refs));
  return React.createElement(
    "div", { className: "section" },
    React.createElement("div", { className: "section-title", key: "title" }, "bound constant"),
    kv("node",
      React.createElement("span", { className: "v mono ref-pill",
        title: `jump to ${sel.nodeId}`,
        onClick: () => onPickNode(sel.nodeId) }, sel.nodeId)),
    kv("port", React.createElement("span", { className: "v mono" }, sel.port)),
    kv("type", typeEl),
    kv("value", renderBoundValue(value)),
  );
}
// Detects "<module.path:func(args)>" and renders a clickable function pill that
// opens the function in VS Code; otherwise pretty-prints as JSON.
function renderBoundValue(value) {
  if (typeof value === "string") {
    const m = value.match(/^<\s*([\w.]+:[\w.]+)\s*\(([^)]*)\)\s*>$/);
    if (m) {
      const target = m[1];
      const args = m[2];
      const funcName = target.split(":").pop();
      return React.createElement("span", { className: "v mono" },
        React.createElement("span", {
          className: "ref-pill",
          title: `open ${target} in VS Code`,
          onClick: () => openTargetInVSCode(target),
        }, funcName),
        React.createElement("span", null, `(${args})`));
    }
  }
  return React.createElement(
    "pre",
    { className: "v mono", style: { margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" } },
    value === undefined ? "—" : JSON.stringify(value, null, 2));
}
const renderBoundPanel = (sel, node, onPickNode) =>
  React.createElement(BoundPanel, { sel, node, onPickNode });

function renderEdgePanel(edge, types, onSetPolicy, onPickNode, onDeleteEdge, ctx) {
  ctx = ctx || {};
  const children = [];
  const policy = (edge.data && edge.data.policy) || "lossless";
  const extra = (edge.data && edge.data.extra_args) || [];
  const headerButtons = [];
  if (onDeleteEdge && !ctx.locked) {
    headerButtons.push(React.createElement("button", {
      key: "del",
      className: "btn-danger",
      title: "delete edge",
      onClick: () => onDeleteEdge(edge.id),
      style: { fontSize: 11, padding: "2px 8px", cursor: "pointer" },
    }, "delete"));
  }
  children.push(React.createElement(
    "div",
    { className: "section-title", key: "title", style: { display: "flex", justifyContent: "space-between", alignItems: "center" } },
    React.createElement("span", null, "edge", ctx.locked ? React.createElement("span", { className: "dim", style: { marginLeft: 6, fontSize: 10 } }, "(locked: running)") : null),
    React.createElement("span", { key: "btns" }, headerButtons),
  ));
  children.push(kv("from",
    React.createElement("span", { className: "v mono ref-pill",
      title: `jump to ${edge.source}`,
      onClick: () => onPickNode(edge.source) },
      `${edge.source}.${stripPrefix(edge.sourceHandle, "out:")}`)));
  children.push(kv("to",
    React.createElement("span", { className: "v mono ref-pill",
      title: `jump to ${edge.target}`,
      onClick: () => onPickNode(edge.target) },
      `${edge.target}.${stripPrefix(edge.targetHandle, "in:")}`)));
  children.push(kv("policy",
    React.createElement("span", { className: "v" },
      React.createElement("select",
        { value: policy, onChange: (e) => onSetPolicy(edge.id, e.target.value) },
        POLICIES.map((p) => React.createElement("option", { key: p, value: p }, p))
      )
    )));
  if (extra.length) {
    children.push(kv("extra_args", React.createElement("span", { className: "v mono" }, JSON.stringify(extra))));
  }

  if (types && types.loading) {
    children.push(React.createElement("div", { className: "hint", key: "loading" }, "resolving type…"));
  } else if (types) {
    const renderSide = (sideInfo, label, arrow) => {
      if (!sideInfo) return null;
      const ann = sideInfo.annotation || "(no annotation)";
      const portName = sideInfo.port || "";
      const paramName = sideInfo.name && sideInfo.name !== portName ? `${sideInfo.name}: ` : "";
      return React.createElement("div", { className: "subsection", key: label },
        React.createElement("div", { className: "subhead" }, `${arrow} ${label}`),
        kv("port", React.createElement("span", { className: "v mono" }, portName)),
        kv("type", React.createElement("span", { className: "v" },
          React.createElement("span", { className: "mono" }, paramName + ann),
          renderRefPills(sideInfo.refs)
        )),
      );
    };
    children.push(renderSide(types.out, "source output (return)", "←"));
    children.push(renderSide(types.in,  "target input (param)",  "→"));
  }

  // Flow stats (count / rate / staleness) — ALWAYS visible for a selected edge. The heart of the
  // plain-Run probe: shows whether data is actually flowing at the expected rate. ``expectedHz``
  // (per-edge override, else source node's derived driving-timer rate) drives the colour:
  // >=80% ok, 50-80% yellow (warning), <50% red (too slow / blocked).
  {
    const stats = (ctx.sample && ctx.sample.stats) || null;
    const hz = stats ? stats.hz : null;
    const exp = ctx.expectedHz || null;
    const ratio = (hz != null && exp) ? hz / exp : null;
    const rateColor = ratio == null ? undefined : ratio >= 0.8 ? "#2e7d32" : ratio >= 0.5 ? "#f9a825" : "#c62828";
    const fmtHz = (v) => (v == null ? "—" : v >= 10 ? `${v.toFixed(0)} Hz` : `${v.toFixed(1)} Hz`);
    const statRows = [];
    if (stats) {
      statRows.push(kv("messages", React.createElement("span", { className: "v mono" }, String(stats.count))));
      statRows.push(kv("rate", React.createElement("span", { className: "v mono", style: { color: rateColor, fontWeight: rateColor ? 700 : 400 } },
        fmtHz(hz) + (exp ? ` / ${fmtHz(exp)} expected` : ""))));
      statRows.push(kv("last msg", React.createElement("span", { className: "v mono", style: { color: stats.age_s > 0.5 ? "#c62828" : undefined } },
        `${stats.age_s.toFixed(2)}s ago`)));
    } else {
      statRows.push(kv("rate", React.createElement("span", { className: "dim", style: { fontSize: 11 } },
        ctx.running
          ? (exp ? `measuring… (expected ${fmtHz(exp)})` : "measuring…")
          : "run the app to measure")));
    }
    // Editable expected-frequency override. Placeholder shows the derived default; typing a value
    // stores an override (in-memory) that Save persists to the repo config file.
    const overrideVal = ctx.expectedOverride;
    const derived = ctx.expectedDerived;
    const expInput = React.createElement("span", { className: "v", style: { display: "flex", alignItems: "center", gap: 4 } },
      React.createElement("input", {
        type: "number", min: 0, step: "any",
        className: "mono",
        style: { width: 78, fontSize: 11, padding: "1px 4px" },
        value: overrideVal == null ? "" : overrideVal,
        placeholder: derived != null ? `${derived >= 10 ? derived.toFixed(0) : derived.toFixed(1)} (auto)` : "auto",
        title: "expected frequency (Hz) for this edge; blank = auto-derived from source timer",
        onChange: (e) => ctx.onSetExpectedHz && ctx.onSetExpectedHz(ctx.expectedKey, e.target.value),
      }),
      React.createElement("span", { className: "dim", style: { fontSize: 10 } }, "Hz"),
      ctx.onSaveExpectedHz
        ? React.createElement("button", {
            onClick: () => ctx.onSaveExpectedHz(),
            title: "save all expected rates to the repo config file",
            style: { fontSize: 10, padding: "1px 6px", marginLeft: "auto" },
          }, ctx.expectedSaved ? "✓ saved" : "save")
        : null,
    );
    statRows.push(kv("expected", expInput));
    children.push(React.createElement("div", { className: "subsection", key: "stats" },
      React.createElement("div", { className: "subhead" }, "flow stats"),
      ...statRows,
    ));
  }

  if (ctx.tapped) {
    const hist = ctx.history || [];
    // Pick sample to display: latest sample with ts <= scrubTime, else nearest;
    // if no history yet, fall back to live tap sample.
    let displaySample = ctx.sample;
    let isLive = true;
    if (ctx.scrubTime != null) {
      if (hist.length) {
        let pick = null;
        for (let i = hist.length - 1; i >= 0; i--) {
          if (hist[i].ts <= ctx.scrubTime) { pick = hist[i]; break; }
        }
        if (!pick) pick = hist[0];  // scrubbed before this edge's first sample.
        displaySample = pick;
        isLive = false;
      } else {
        // Mark as scrubbed even without data so user sees state.
        isLive = false;
        displaySample = null;
      }
    }
    const sampleTs = displaySample && displaySample.ts ? new Date(displaySample.ts * 1000).toLocaleTimeString() : "";
    children.push(React.createElement("div", { className: "subsection", key: "live" },
      React.createElement("div", { className: "subhead", style: { display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" } },
        React.createElement("span", null, isLive ? "live value" : (sampleTs ? "value @ " + sampleTs : "no sample at scrub time")),
        !isLive ? React.createElement("span", { className: "dim", style: { fontSize: 10 } }, `(scrubbed • ${hist.length} samples buffered)`) : null,
        ctx.running ? null : React.createElement("span", { className: "dim", style: { fontSize: 10 } }, "(run app to stream)"),
        React.createElement("span", { style: { marginLeft: "auto", display: "flex", gap: 4 } },
          ctx.running && ctx.onStepNode
            ? React.createElement("button", {
                onClick: () => ctx.onStepNode(edge.source),
                title: `step source node \"${edge.source}\" one tick`,
                style: { fontSize: 10, padding: "1px 6px" },
              }, `⏭ ${edge.source}`)
            : null,
          !isLive && ctx.onSetScrubTime
            ? React.createElement("button", {
                onClick: () => ctx.onSetScrubTime(null),
                title: "jump to live",
                style: { fontSize: 10, padding: "1px 6px" },
              }, "live")
            : null,
        ),
      ),
      React.createElement(LiveValue, { sample: displaySample }),
    ));
  }

  return React.createElement("div", { className: "section edge-panel" }, children);
}

function renderInstancePanel(inst, memberIds, info, schema, onPickNode, onUpdateInstanceKwargs, onReschema, locked, hiddenNodeIds, onToggleNodeHidden, onToggleGroupHidden, hiddenInstanceIds, onToggleInstanceHidden) {
  const children = [];
  // Fully hidden when every member node is hidden OR the instance is memorized-
  // hidden (so it stays greyed out across rebuilds even before members reload).
  const memoHidden = !!(hiddenInstanceIds && hiddenInstanceIds.has(inst.id));
  const allHidden = memoHidden || (memberIds.length > 0 && hiddenNodeIds && memberIds.every((id) => hiddenNodeIds.has(id)));
  const onInstToggle = onToggleInstanceHidden
    ? () => onToggleInstanceHidden(inst.id, memberIds)
    : (onToggleGroupHidden ? () => onToggleGroupHidden(memberIds) : null);
  children.push(React.createElement("div", { className: "section-title" + (allHidden ? " group-hidden" : ""), key: "title" },
    "shared instance: ",
    React.createElement("span", { className: "mono" }, inst.id),
    onInstToggle && React.createElement(EyeToggle, {
      hidden: allHidden,
      onToggle: onInstToggle,
      title: allHidden ? "show all nodes of this instance" : "hide all nodes of this instance",
      className: "eye-inline",
    })));
  children.push(kv("factory", React.createElement(
    "span",
    { className: "v mono ref-pill", title: "double-click to open in VS Code",
      onDoubleClick: () => openTargetInVSCode(inst.factory) },
    inst.factory || "—"
  )));
  // Editable ctor config: exactly what the factory is called with when the
  // graph is materialized, so tuning it here changes what the run builds. When
  // the server can introspect a typed schema (a node ``config_schema`` or a
  // ``CONFIG_TYPE`` dataclass, e.g. SMPC's judo controller/optimizer groups) we
  // render typed form controls; otherwise we fall back to the raw-JSON
  // KwargsEditor. ``key`` is scoped to the instance id so uncontrolled inputs
  // remount when a different container is selected.
  const hasSchema = schema && Array.isArray(schema.fields) && schema.fields.length;
  let configBody;
  if (!onUpdateInstanceKwargs) {
    configBody = React.createElement("span", { className: "v mono" }, JSON.stringify(inst.kwargs || {}));
  } else if (hasSchema) {
    configBody = React.createElement(SchemaForm, {
      fields: schema.fields,
      values: inst.kwargs || {},
      onChange: (kw) => onUpdateInstanceKwargs(inst.id, kw),
      onReschema: onReschema ? (kw) => onReschema(inst.id, kw) : null,
      disabled: locked,
    });
  } else if (schema && schema.loading) {
    configBody = React.createElement("div", { className: "hint" }, "loading schema…");
  } else {
    configBody = React.createElement(KwargsEditor, {
      kwargs: inst.kwargs || {},
      onChange: (kw) => onUpdateInstanceKwargs(inst.id, kw),
      disabled: locked,
    });
  }
  children.push(React.createElement(
    "div",
    { className: "subsection", key: `inst-kwargs-${inst.id}` },
    React.createElement("div", { className: "subhead" }, "config"),
    configBody,
    schema && schema.error
      ? React.createElement("div", { className: "hint err", key: "serr" }, `schema: ${schema.error}`)
      : null
  ));
  children.push(React.createElement("div", { className: "hint", key: "hint" },
    hasSchema
      ? "typed config for the factory call; blank sub-fields use the task/default baseline"
      : "ctor kwargs edit the factory call (JSON values); double-click container, factory, or a type pill to open in VS Code"));

  if (info && info.loading) {
    children.push(React.createElement("div", { className: "hint", key: "loading" }, "loading class state…"));
  } else if (info && info.error) {
    children.push(React.createElement("div", { className: "hint err", key: "err" }, info.error));
  } else if (info && info.attrs) {
    children.push(React.createElement("div", { className: "subhead", key: "cls" },
      `class ${info.class}`,
      info.bases && info.bases.length
        ? React.createElement(
            "span",
            { className: "mono dim", key: "bases" },
            "(",
            info.bases.map((b, i) => [
              i > 0 ? React.createElement("span", { key: `sep${i}` }, ", ") : null,
              b.refs && b.refs.length
                ? React.createElement(
                    "span",
                    {
                      key: `b${i}`,
                      className: "ref-pill",
                      title: `double-click to open ${b.refs[0].target} in VS Code`,
                      onDoubleClick: (e) => { e.stopPropagation(); openTargetInVSCode(b.refs[0].target); },
                    },
                    b.text
                  )
                : React.createElement("span", { key: `b${i}` }, b.text),
            ]),
            ")"
          )
        : null));
    if (info.doc) {
      children.push(React.createElement("div", { className: "doc", key: "doc" }, info.doc.split("\n")[0]));
    }
    children.push(React.createElement("div", { className: "subhead", key: "ahead" },
      `attributes (${info.attrs.length})`));
    children.push(React.createElement("ul", { className: "attrs", key: "alist" },
      info.attrs.map((a) => React.createElement("li", { key: a.name },
        React.createElement(
          "span",
          {
            className: "mono attr-name",
            title: `double-click to open at ${info.file}:${a.line}`,
            onDoubleClick: (e) => { e.stopPropagation(); openFileInVSCode(info.file, a.line); },
          },
          a.name
        ),
        a.annotation
          ? React.createElement("span", { className: "mono dim" }, `: ${a.annotation}`)
          : null,
        renderRefPills(a.annotation_refs),
        React.createElement("span", { className: "dim attr-source" }, a.source)
      ))
    ));
    children.push(React.createElement("div", { className: "subhead", key: "mhead" },
      `methods (${info.methods.length})`));
    children.push(React.createElement("div", { className: "methods", key: "mlist" },
      info.methods.map((m) => React.createElement(
        "span",
        {
          key: m.name,
          className: "method-pill mono",
          title: `double-click to open ${info.class}.${m.name} in VS Code`,
          onDoubleClick: () => openTargetInVSCode(`${info.target.split(":")[0]}:${info.class}.${m.name}`),
        },
        m.name
      ))
    ));
  }

  children.push(React.createElement("div", { className: "subhead", key: "members" },
    `members (${memberIds.length})`));
  children.push(React.createElement("ul", { className: "members", key: "mids" },
    memberIds.map((id) => {
      const isHidden = hiddenNodeIds && hiddenNodeIds.has(id);
      return React.createElement("li", {
        key: id,
        className: "mono" + (isHidden ? " hidden-row" : ""),
        onClick: () => onPickNode(id),
      },
      React.createElement("span", { className: "lbl" }, id),
      onToggleNodeHidden && React.createElement(EyeToggle, {
        hidden: isHidden,
        onToggle: () => onToggleNodeHidden(id),
        title: isHidden ? "show node" : "hide node",
      }));
    })
  ));

  return React.createElement("div", { className: "section instance-panel" }, children);
}

function EditableSelect({ value, options, placeholder, onChange }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  if (editing) {
    return React.createElement(
      "span",
      { className: "v" },
      React.createElement("input", {
        autoFocus: true,
        value: draft,
        placeholder: "new name",
        onChange: (e) => setDraft(e.target.value),
        onBlur: () => { setEditing(false); if (draft) onChange(draft); },
        onKeyDown: (e) => {
          if (e.key === "Enter") { setEditing(false); if (draft) onChange(draft); }
          if (e.key === "Escape") setEditing(false);
        },
      })
    );
  }
  return React.createElement(
    "span",
    { className: "v" },
    React.createElement(
      "select",
      { value, onChange: (e) => onChange(e.target.value) },
      React.createElement("option", { value: "" }, placeholder),
      options.map((o) => React.createElement("option", { key: o, value: o }, o))
    ),
    React.createElement("button", { onClick: () => { setDraft(""); setEditing(true); }, title: "new" }, "+")
  );
}

createRoot(document.getElementById("root")).render(
  React.createElement(ErrorBoundary, { label: "The editor crashed while rendering." },
    React.createElement(App)));
