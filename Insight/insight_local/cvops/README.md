# Insight CV Ops Sidecar

Separate CV Ops process for queued background training/inference jobs.

## Launch

From the `Insight` project root:

```bash
python -m insight_local.cvops --local --host 127.0.0.1 --port 8787
```

You can also set `INSIGHT_CVOPS_COLOR_SCHEME` (`default`, `marathon`, `wear_marathon`, `fire`).

This starts:
- Local FastAPI service (`HTTP + WebSocket`) on localhost.
- A dedicated Qt window with `Catalog`, `Continuous Learning`, `Range`, `Database`, `Console`, `Dashboard`, `Settings`, `Scenarios`, `Errors`, and `Gallery (Read-Only)` tabs.
- An `Errors` tab that aggregates websocket, job, and operation failures with timestamps.

## AI provider keys and where they are stored

Cloud AI features (OpenAI, Anthropic, Grok/xAI, Gemini) are optional — local
Ollama / GGUF models work with no key. When you do enter a key in
**AI Settings**, it is stored in your operating system keyring via the
[`keyring`](https://pypi.org/project/keyring/) package:

- **macOS** — Keychain
- **Windows** — Credential Locker
- **Linux** — Secret Service (GNOME Keyring / KWallet)

Keys are **not** written in plaintext to disk. The rest of AI Settings
(assistant name, voice profile, system prompt, local model paths) lives in
`state/insight_local/cvops/notes/ai_settings.json`, which contains no secrets.

If no OS keyring backend is available (e.g. a headless Linux box without a
Secret Service), the app falls back to storing keys in that same
`ai_settings.json` so it keeps working — the AI Settings tab shows a
`[NO OS KEYRING DETECTED]` warning when this fallback is active. Installing
`keyring` and restarting moves keys back into the OS store. Older installs that
saved plaintext keys in `ai_settings.json` are migrated into the keyring
automatically on first launch and stripped from the file.

`ai_settings.json` and all of `state/` are gitignored and never committed.

## API

- `GET /health`
- `GET /scenarios`
- `GET /scenarios/{scenario}/status`
- `GET /models`
- `POST /scenarios/{scenario}/model`
- `POST /scenarios/{scenario}/train`
- `POST /scenarios/{scenario}/update`
- `POST /scenarios/{scenario}/verify`
- `DELETE /scenarios/{scenario}/verify`
- `GET /datasets/{scenario}`
- `GET /datasets/{scenario}/thumb/{name}`
- `POST /datasets/{scenario}/upload`
- `DELETE /datasets/{scenario}/{name}`
- `GET /database`
- `GET /database/{slug}`
- `GET /database/{slug}/inventory`
- `POST /database/{slug}/inventory/move_by_ext`
- `POST /database/{slug}/inventory/delete_by_ext`
- `GET /database/{slug}/thumb/{name}`
- `GET /database/{slug}/label/{name}`
- `POST /database/{slug}/add`
- `DELETE /database/{slug}/{name}`
- `POST /jobs`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/training_progress`
- `WS /events`
- `GET /lineages/{lineage_id}/provenance` — PROV-JSON document for a local lineage (not available for `registry:` lineages).
- `POST /provenance/backfill` — JSON body `{"lineage_id": "<optional>"}`; omit `lineage_id` to backfill all local lineages into `provenance.db`.

## Provenance (W3C PROV)

Continuous learning lineage and snapshot lifecycle events are mirrored into a dedicated SQLite graph at `state/insight_local/cvops/provenance.db` (`prov_nodes`, `prov_edges`). Identifiers use the `urn:cvops:prov:` namespace (for example `urn:cvops:prov:entity:snapshot:snap-…`, `urn:cvops:prov:activity:drop:drop-…`, `urn:cvops:prov:entity:lineage:line-…`).

| cvops | PROV |
|-------|------|
| Model snapshot | `entity` (`prov:type` `cvops:ModelSnapshot`) |
| Drop step | `activity` |
| Lineage | `entity` collection (`cvops:LineageCollection`); `hadMember` links snapshots in order |
| Service | `agent` `urn:cvops:prov:agent:cvops-service`; `wasAssociatedWith` on drop activities |
| Snapshot register / parent link | `wasDerivedFrom` when `parent_snapshot_id` is set; optional `wasAttributedTo` if snapshot `metadata` includes `prov_agent`, `prov:agent`, `attributed_to`, or `attributedTo` (URI string) |
| New drop | `used` (prior snapshot and optional extras), `wasGeneratedBy`, `wasDerivedFrom`, `wasInformedBy`, `hadMember` |
| Fork | `specializationOf` (new lineage entity, source lineage entity) |
| Delete snapshot | `wasInvalidatedBy` on the snapshot entity (weights row is removed from the snapshot store, but provenance nodes and prior edges stay) |
| Delete lineage | `wasInvalidatedBy` on the **lineage collection** entity only (snapshots may be shared across lineages) |

Optional `source` field on drops: list arbitrary entity URIs in `prov_used_entities` to record additional `used` entities.

Environment:

- `CVOPS_PROVENANCE_BACKFILL_ON_START` — set to `1`, `true`, or `yes` to run a full backfill after the service registers routes (useful after upgrading or if `provenance.db` was deleted).

Repair provenance (CLI, no server required):

```bash
# From repo root (uses .venv)
./scripts/backfill_provenance.sh

# Or from Insight/
python -m insight_local.cvops.backfill_provenance
python -m insight_local.cvops.backfill_provenance --lineage-id line-abc123
```

Backfill only replays rows already in `lineages.db` and `snapshots.db`. It does not import train jobs or `model_registry.json` into continuous learning.

**Visualization:** `GET /ontology/graph` and the ecosystem graph (`GET /ecosystem/graph_view`) merge a W3C PROV overlay into the Cytoscape graph whenever both `lineage` and `model_snapshot` entity types are included (default unfiltered graph). Extra node types `prov_activity`, `prov_agent`, and `prov_entity` (external inputs) appear with PROV-styled edges (`prov_generates`, `prov_used`, `had_member`, `specialization_of`, etc.). Lineage node metadata includes `w3c_prov_overlay_edges` / `w3c_prov_overlay_nodes`. The Qt Continuous Learning panel and the Nice web **Lineage** column show the raw PROV-JSON for a selected lineage.

## Notes

- Jobs are durable in SQLite at `state/insight_local/cvops/jobs.db`.
- Integration stream is appended to `mlops/integration/events.jsonl`.
- Training jobs execute real YOLO training via `mlops.pipeline.train`.
- `update` jobs are incremental-by-intent: they start a new run using existing model state plus newly labeled data.
- Gallery access in CV Ops is read-only in this phase.
- Dataset library root for `/database` is `database/` at repo root.
