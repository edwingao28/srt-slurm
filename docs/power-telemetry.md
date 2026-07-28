# GPU Power Telemetry (`dcgm-power`)

> **Status:** CPU-validated; GB200 canary pending a cluster-provided DCGM
> exporter image.

The `dcgm-power` telemetry provider records raw per-GPU watts for every
allocated worker node, the topology needed to map each GPU to a `prefill`,
`decode`, or `agg` role, and the exact formal benchmark window for every
measured concurrency.

It is a **producer only**. It never integrates power into energy, and it never
branches on model, precision, or recipe. Consumers integrate watts over the
recorded window themselves.

## How it works

- One DCGM exporter task runs on each allocated worker node, launched through
  the normal SLURM/process-registry path (one `srun` per heterogeneous group).
- A collector thread inside the orchestrator process polls every exporter
  concurrently from the physical head node, so **all** sample timestamps and
  benchmark boundaries come from one clock and no node-clock skew enters
  cross-node comparison.
- Only `DCGM_FI_DEV_POWER_USAGE` is parsed. Device identity comes from the
  `gpu` and `UUID` labels; the exporter's optional `Hostname` label is ignored
  because the collector already knows which allocated node it polled.
- SA-Bench writes one measurement-window file per measured concurrency. Warmup
  omits `--save-result`, so it never writes a window.

## Configuration

```yaml
benchmark:
  type: sa-bench          # the only v1 measurement-window adapter
  client_placement: head  # keeps sample and window clocks on one host
  isl: 8192
  osl: 1024
  concurrencies: [4]

telemetry:
  enabled: true
  provider: dcgm-power
  default_frequency: 1.0            # seconds between collector cycles
  storage_subdir: power             # relative to the run log directory
  required: true                    # exit non-zero when artifacts are unpublishable
  startup_timeout_seconds: 30
  request_timeout_seconds: 2
  collector_join_timeout_seconds: 10
  dcgm_exporter:
    container_image: dcgm-exporter  # alias or path to a cluster-visible image
    port: 9401
```

`dcgm-power` needs **only** `dcgm_exporter`. Unlike `provider: scraper` it does
not require the top-level `container_image` or a `node_exporter`, because the
collector runs inside srtctl instead of a scraper container.

Validation rejects, at config load: a missing or empty exporter image, a port
outside `1..65535`, a non-positive or non-finite interval/timeout, a
`collector_join_timeout_seconds` that is not greater than
`request_timeout_seconds`, an unsafe `storage_subdir`, a benchmark that is not
`sa-bench`, a client placed off the head node, and an empty, duplicated, or
non-positive concurrency list.

Telemetry stays disabled by default and existing `provider: scraper` recipes
are unchanged.

## Artifacts

```text
<log_dir>/<storage_subdir>/
├── manifest.json
├── samples.csv
└── windows/
    └── <benchmark-result-stem>.json
```

### `samples.csv`

Exact header:

```text
schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w
```

One row per observation, and `(scrape_seq, hostname, gpu_index)` is unique.
`timestamp_unix` is the midpoint of one endpoint request on the head-node Unix
clock; `hostname` is the allocated SLURM node name; `power_w` is the raw DCGM
gauge. Rows are never interpolated, averaged, or role-attributed — role and
heterogeneous group live once in the manifest topology instead of being copied
onto every row.

### `manifest.json`

Producer identity (version, git commit, exporter image and its SHA-256 — null
when the image is a registry URI rather than a local file), the sample
interval, expected and observed device sets, the topology mapping, the
expected window list, terminal status, per-window coverage validation, and
reason codes.

`status` is a lifecycle outcome (`starting`, `running`, `complete`,
`incomplete`, `failed`); `publication_valid` is the separate publication gate
and is `null` until the manifest is terminal. A `complete` session can still be
publication-invalid.

### `windows/<result-stem>.json`

```json
{
  "schema_version": 1,
  "benchmark_type": "sa-bench",
  "result_path": "sa-bench_isl_8192_osl_1024/results_concurrency_4_gpus_8_ctx_4_gen_4.json",
  "concurrency": 4,
  "benchmark_start_time_unix": 1785168100.0,
  "benchmark_end_time_unix": 1785168120.0,
  "duration": 20.0,
  "clock_source": "head_node_unix_clock",
  "status": "completed",
  "reason": null
}
```

`duration` comes from a monotonic clock and is never derived by subtracting the
Unix fields; the Unix fields exist to align the window with power samples. The
saved SA-Bench result keeps its existing `duration` field and gains the same
two Unix boundaries, so the result and the window are boundary-identical.

`status` is `running` until the formal requests resolve, then `completed`, or
`failed` when a trustworthy end boundary exists. A window that is still
`running` after the benchmark child is reaped becomes `interrupted`, with end
and duration left null — the orchestrator never invents a completion boundary.

## Validity

A window is `power_coverage_valid` only when the window and its result are
well-formed and completed, their start/end/duration agree, every expected
device has exactly one stable identity, every expected device has a sample at
or before the formal start and at or after the formal end, and no device has a
gap greater than 3 seconds in that inclusive bracketing sequence.

A terminal manifest is `publication_valid` only when the status is `complete`,
no fatal lifecycle reason was recorded, the expected device set is non-empty
and exactly equals the observed set, every device has one stable UUID and one
semantic role, every expected window has exactly one completed artifact with
valid coverage, and no malformed, unexpected, duplicate, or symlinked artifact
file was found.

With `required: true`, all available artifacts are written first and the job
then exits non-zero whenever the terminal manifest is not publishable. With
`required: false`, telemetry invalidity never changes the benchmark exit code.

### Reason codes

Stable machine-readable strings recorded in `manifest.json`:

| Group | Codes |
|---|---|
| Exporter | `exporter_startup_timeout`, `exporter_launch_failed`, `exporter_exited` |
| Endpoint | `endpoint_timeout`, `endpoint_http_error`, `endpoint_resolution_failed` |
| Metrics | `power_metric_missing`, `duplicate_power_metric`, `gpu_index_missing`, `gpu_uuid_missing`, `invalid_power_value`, `mig_instance_unsupported` |
| Samples | `samples_csv_missing`, `samples_csv_header_mismatch`, `samples_csv_malformed`, `duplicate_sample_row`, `timestamp_non_monotonic` |
| Topology | `unexpected_device`, `expected_device_missing`, `gpu_uuid_changed`, `conflicting_worker_roles`, `conflicting_het_groups` |
| Collector | `collector_exception`, `collector_interrupted`, `collector_join_timeout`, `benchmark_child_reap_timeout`, `final_scrape_unavailable` |
| Windows | `measurement_window_missing`, `measurement_window_unexpected`, `measurement_window_duplicate`, `measurement_window_malformed`, `measurement_window_artifact_path_invalid`, `measurement_window_incomplete`, `measurement_window_result_missing`, `measurement_window_result_mismatch`, `measurement_window_result_path_invalid`, `measurement_window_clock_mismatch`, `measurement_window_not_bracketed`, `sample_gap_exceeded` |

MIG instances are not a publication target in v1; MIG-labelled series are
rejected with `mig_instance_unsupported`.

## Known limitations

**Cleanup ordering on signals and critical-process failure.** In the normal
path the orchestrator finalizes the collector and closes `samples.csv` before
`ProcessRegistry.cleanup()` terminates the exporters. Two pre-existing paths in
the shared process registry bypass that ordering:

- the `SIGTERM`/`SIGINT` handler calls `registry.cleanup()` synchronously inside
  the handler, before any unwinding reaches the orchestrator's `finally`; and
- the background process monitor calls `registry.cleanup()` as soon as a
  critical process dies.

In both cases the DCGM exporters are terminated *before* the collector is
finalized, so the final scrape sees dead endpoints. The manifest fails closed:
`exporter_exited` and `collector_interrupted` are fatal lifecycle reasons, which
force `status=incomplete` and `publication_valid=false`, so no unusable data is
ever published. The cost is readability rather than correctness — a job that was
simply cancelled can record `exporter_exited`, which reads as though the
exporter crashed on its own.

PR2 deliberately does not modify the shared process registry, since reordering
cleanup affects every provider and every job type.

## Re-validating a retained run

The artifact package is self-describing. Anyone can re-derive the verdict from
the persisted bytes without access to the live job — the manifest supplies the
expected topology, never the verdict:

```bash
uv run python -m srtctl.core.power.validate_artifacts \
  --power-dir outputs/12345/logs/power \
  --result-root outputs/12345/logs
```

It exits `0` when the package is publishable and `1` otherwise, printing each
failing reason code. Two optional flags assert an expected job shape, which is
useful for a hardware canary:

```bash
uv run python -m srtctl.core.power.validate_artifacts \
  --power-dir outputs/12345/logs/power \
  --result-root outputs/12345/logs \
  --expect-role prefill=4 --expect-role decode=4 \
  --require-distinct-het-groups
```
