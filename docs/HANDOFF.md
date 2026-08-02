# HANDOFF — Vega 7-GPU inference tuning (2026-08-02, ~08:00 UTC)

Context for the next agent picking up this work. Read this top to bottom before touching
the rig. The prior session log lives in the repo history and `benchmarks/llm/index.jsonl`
(one JSON record per experiment, including falsifications — treat it as the source of truth).

## Mission and the user's explicit targets

- Serve **Qwen3.6-35B-A3B-UD-Q4_K_XL** (21.27 GiB MoE, 256 experts top-8, arch `qwen35moe`)
  on 7× Vega 56 (gfx900) as **two pods (4+3 cards)**, every stream at 128k ctx (or slightly
  less if VRAM-bound). See `docs/serving-4plus3.md` for the blueprint.
- Per-stream floor: **≥500 PP and ≥25 TG**; more is better. 1000 PP is a standing target —
  acceptable as *combined* throughput across pods.
- **No CPU inference ever. Minimal host RAM.** The rig gets a 16 GB RAM upgrade (currently
  3.8 GB — this is the wall for anything two-process, see "Confirmed results").
- Older standing goal (still alive, deprioritized): 1000 PP + 30 TG on a **single** Vega
  with Bonsai-27B Q1_0. Best so far: 226 PP / 26 TG — prefill is the unsolved half.
- Keep publishing everything to GitHub (`dhavli/llamacpp-vega-gfx900`, branch
  `bench/vega-bonsai-baseline`). Commit style: see `git log`.

## Environment (details in memory file `vega-rig-environment`)

- Rig: `ssh root@100.95.237.97`, HiveOS, 2-core Celeron, 3.8 GB RAM, 59 GB disk at **97%**.
  7× Vega 56 on **PCIe Gen2 x1 risers (~0.5 GB/s)** — verified; ignore sysfs/lspci GPU-endpoint
  link readings (they show the on-die bridge; truth is `pp_dpm_pcie`).
- Runtime (current, b9599 + our patch):
  `/nix/store/0rlr67bjyf76wfzf1mmzgq73pk22fk5x-vega-runtime` (older b9596:
  `1fwyjlxqfl4mdbb8sas40lmhwrrvrwdn`). Models in `/root/models/`, work dir `/root/bonsai/`.
- Build box = this container: edit `flake.nix` (pin FULL 40-char revs; GitHub API is
  rate-limited), `nix build .#vega-runtime -o result-runtime`, ship with
  `nix copy --to ssh://root@100.95.237.97 --no-check-sigs ./result-runtime`.
  Patches in `patches/*.patch` auto-apply; 0001 carries ALL kernel work + env knobs.
- Clocks: a reboot reverts to the 950 MHz mining profile. Reapply:
  `for i in 1..7: echo manual > /sys/class/drm/card$i/device/power_dpm_force_performance_level; echo 7 > pp_dpm_sclk`.
  Cards 6/7 top at 1622/1750 (different BIOS tables) — fine.
- The user's `coldcard-finder` job may auto-start and take all GPUs; never kill it —
  poll for it to exit before benchmarking (see any `par*.sh` for the guard pattern).

## State of the rig RIGHT NOW

- `recover2.sh` was finishing its last matrix run (`qa-e6`); `followup.sh` (PID ~357660)
  is queued behind it via a pgrep poll. Logs: `/root/bonsai/recover2.log`,
  `/root/bonsai/followup.log`; sentinels `### RECOVERDONE`, `### FOLLOWUPDONE`.
- `followup.sh` will produce, in order:
  1. **V:** `--verbose` init on 4 cards → grep `pipeline|n_copies|retrying without` from
     `/tmp/qv.log` — diagnoses whether/why pipeline parallelism is disabled (see below).
  2. **QB:** stacked combos `nogttspill+ALLOW_GRAPHICS_QUEUE` and `+ASYNC_USE_TRANSFER_QUEUE`,
     byte-diffed against `/tmp/qa-base.txt`.
  3. **P3:** TILE_M perplexity A/B with **`-b 512`** (RAM-safe) → `/tmp/p3-{ctrl,tile}.pplout`.
     This adjudicates the +19% prefill warptile (see "Pending verdicts").
- When both sentinels are present: record everything in `benchmarks/llm/index.jsonl`,
  update README + `docs/serving-4plus3.md` env lines, push.

## Confirmed results (all output-verified; ledger has full records)

| finding | numbers |
|---|---|
| 4-card pod beats 6-card at equal serving load | 471 PP / 52.2 TG vs 425 / 48.6 (4 slots × 128k, q8_0 KV) |
| 8×128k does NOT fit 4 cards | ~32.3 GB needed vs 31.9 GB VRAM; needs 5th card |
| Dual-instance on 3.8 GB host: impossible | OOM-killed every time; **zram does NOT help** — pressure is pinned GTT (~1.2 GB/instance, anon RSS of killed procs ~150 KB). 16 GB physical RAM is a hard requirement |
| Host contention between two instances | sole surviving E1 4-card run: 321 PP / 25.4 TG vs 540 / 32.2 solo (~-35%) — even with RAM fixed, expect some contention on the 2-core host |
| **MTP self-speculation: FALSIFIED on x1 risers** | ctrl 519.5/29.7 → mtp n-max 3: 35.5/3.9 (!) despite **79% acceptance**; n-max 5: 209/16.8 at 53%. Draft calls ~77 ms each — MTP's device-to-host embedding copies amplified by Gen2 x1. Do not revisit until wider links |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | TG **29.44 → 34.87 (+18%)**, PP unchanged. Biggest single knob found |
| `RADV_PERFTEST=nogttspill` | TG 29.44 → 31.72 (+8%), PP unchanged |
| `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` | +2% alone; combo gtt+tq (30.3) < gtt alone (31.7) — tq slightly hurts when stacked |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | +4% TG alone |
| `-b 4096 -ub 256` | PP 449 vs 519 base — smaller ubatch hurts; `-ub 1024` run produced no output (check `/tmp/qa-ub1024.log` for the failure mode) |
| Perplexity runs OOM the box | 248320 vocab × 2048 batch × 4 B = 2 GB logits buffer on host. **Always `-b 512`** (508 MB). This OOM (not MTP, not the flaky card — probably) is what took the rig down overnight |
| PCIe survey | all 7 cards Gen2 x1; decode traffic ~4 KB/tok/boundary (fine), prefill ~MB/ubatch/boundary (why MoE prefill decays with card count) |
| MoE vs dense splitting | MoE prefill DECREASES with cards (641/561/506 @ pp2048 for 3/4/6); dense increases at depth. Decode optimum 4 cards, prefill optimum 3 |

Earlier falsifications (README "Falsified" section): fast-but-wrong warptiles (TM=8 etc. —
**never** trust throughput without byte-identical greedy output), VGPR diet, LDS→ALU,
f16 accumulators, `-sm row` on Vulkan (CUDA-only).

## Pending verdicts (followup.sh output)

1. **Pipeline parallelism (PP overlap) — potentially the biggest free prefill lever.**
   `qa-base` showed NO "pipeline parallelism enabled" log line at default verbosity (which
   also hides most init logs — hence the verbose rerun). Gates (from source,
   `src/llama-context.cpp` ~395-424): >1 device, `n_gpu_layers > n_layer_total` (99>41 ok),
   layer split, `offload_kqv`, NO `--override-tensor`, all backends async+events (Vulkan
   claims both). Plus a **silent fallback** at ~line 644 when the compute buffer alloc fails
   ("retrying without pipeline parallelism"). If disabled: find which gate, fix, then REDO
   the `-b`/`-ub` sweep (the current one is meaningless if PP was off). If enabled all along:
   the MoE-prefill-decay finding stands as boundary-copy cost.
2. **TILE_M candidate `256,128,64,64,32,64,2,4,4,1,64`** (+19% PP on both pods, decode
   neutral). Greedy text diverges ~140 lines in — benign-FP-reorder signature, NOT the
   garbage signature. P3 perplexity decides: ppl match to 3-4 digits vs ctrl = adopt (put in
   blueprint env); big delta = falsify and record.
3. **Stacked combos**: expect gfx+gtt ≈ 36-38 TG. If confirmed AND output byte-identical,
   blueprint decode goes 52 → ~60+ aggregate on pod A.
4. **`expert_used_count=6`** (`qa-e6` in recover2.log): read timings + eyeball
   `/tmp/qa-e6.txt` coherence. If fast and sane, it needs a REAL quality eval
   (ppl A/B at `-b 512` + a few task prompts) before entering the blueprint as an option.

## PP/TG improvement paths — tried, in-flight, and UNTRIED

### In-flight / next in queue
- Pipeline-parallelism diagnosis → possibly re-sweep `-b/-ub` after fix (PP lever, MoE+dense).
- TILE_M adjudication (PP +19% if clean).
- Stacked env combos (TG +25-30% if they stack cleanly).
- expert_used_count 6/7 (TG and PP, ~linear in expert count; quality tradeoff).

### Untried — ordered by expected value/effort (from the research pass + local analysis)
1. **HBM2 memory clock + timing straps** (decode lever, mining-proven on these exact cards):
   `amdgpu.ppfeaturemask=0xffffffff` kernel arg, then `echo "m 3 1020" > pp_od_clk_voltage`
   (Vega 56 stock 800; Samsung dies do 1020-1100 — check die vendor via `amdmemtweak --current`;
   Hynix: stay ≤945) and Eliovp `amdmemtweak` straps (full Samsung strap set in ledger notes /
   research report). Our Q1_0 matvec is HBM-LATENCY-bound (102 of 410 GB/s used, waves stall
   a full HBM latency per row) — straps cut latency, so this may move Bonsai TG well past the
   +8-16% clock scaling. **HBM2 has no ECC here: md5-verify greedy output after EVERY step.**
2. **Profile the GDN/SSM ops** — nobody has ever profiled the non-matmul half of these models.
   30/40 Qwen layers and 48/64 Bonsai layers are gated-delta-net; `ssm_scan`/`ssm_conv` Vulkan
   kernels are stock and unexamined. `GGML_VK_PERF_LOGGER=1` (+`_FREQUENCY`) gives per-op times.
   If GDN ops eat a meaningful decode share, that's a fresh, uncontested optimization surface.
3. **Verify the topk_moe fusion actually fires** on qwen35moe: the graph interleaves RESHAPE/
   cache-view nodes that break fusion matchers silently (we hit this before with SIGMOID+MUL).
   A/B with `GGML_VK_DISABLE_FUSION=1`; if no delta, the matcher never fired → relax it in the
   patch (contained decode win, all 40 layers).
4. **Bonsai Q1_0 PREFILL kernel** — the single-Vega 1000 PP goal is blocked here (226 PP).
   All ternary kernel work so far targeted the **matvec** (decode). Prefill goes through
   generic dequant+mmq. A dedicated ternary *tile* kernel (dequant Q1_0 inline in the mmq
   loop, reuse the TILE knob machinery) is the only credible route toward 500+ PP dense.
   Sizeable shader work; the patch already has the scaffolding.
5. **TILE_L / TILE_S sweeps** — only TILE_M (medium path) was ever swept. Large-batch prefill
   uses the L path: same sweep methodology (greedy byte-diff + ppl adjudication), 4-card pod.
6. **Scalar FA follow-ups for hsk=256** (our GDN-attention decode shape): the GCN occupancy
   clamp is in-tree (`ggml-vulkan.cpp:~3220`); check upstream for FA tuning merged after
   b9599 and cherry-pick into the prism pin.
7. **Post-16GB-RAM reruns**: dual-instance C2/E1/E2 (aggregate PP should ≈ sum of pods —
   this is how 1000+ combined PP happens), 5-card 8×128k pod, and `cache_prompt: true`
   serving benefits (prompt-cache reuse needs host RAM).
8. **Vulkan tensor parallelism (the decode endgame)**: pools all cards' bandwidth for the
   matvec instead of layer-serializing. Upstream tracking issue
   github.com/ggml-org/llama.cpp/issues/22648 — segfaults + missing get_tensor_2d as of
   May 2026. Check monthly; do NOT try to build it ourselves yet.
9. **ACO-level micro-experiments** (Bonsai kernels): `ACO_DEBUG=nosched` A/B; LDS row-stride
   padding to odd to dodge bank conflicts; ensure `f16vec2` (not scalar float16_t) in hot
   loops so ACO emits `v_pk_fma_f16`. Cheap to test, small expected wins.
10. **Classic speculative decode with a tiny draft** (e.g. Qwen 0.5B fully on device 0):
    MTP died on per-step D2H embedding copies; a self-contained draft model avoids those but
    still pays per-step logits sync — expect marginal at best on x1; only try if bored.
11. **Hardware paths** (user-side): Gen3-capable board/risers would double+ boundary
    bandwidth (the hardest wall for MoE multi-GPU prefill); reflash/retire the flaky
    `07:00.0` card (hang suspect); 16 GB RAM (committed).

### Explicit dead ends — do not retry (see ledger/README for evidence)
MMVQ forcing (int8 dot is emulated on gfx900), MTP on x1 risers, zram as dual-instance fix,
`-sm row` on Vulkan, warptile geometry violations, VGPR-occupancy diet, LDS→ALU ternary
expansion, f16 ternary accumulators, alternative Vulkan runtimes (MLC/kompute — nothing
beats llama.cpp Vulkan on GCN5), "disable -fa on Vega" (false), CPU/hybrid inference (user
forbids; also 2-core Celeron).

## Traps that cost us hours — read before running anything

- **Throughput without byte-identical greedy output is not a result.** `llama-bench` never
  checks output; invalid warptiles run fast and compute garbage. Always
  `llama-completion --temp 0 --ignore-eos -f prompt8k.txt -n 128..256`, `cmp` the outputs.
  Late divergence (>100 identical lines) = FP reorder → adjudicate with perplexity, `-b 512`.
- **Perplexity = 2 GB host logits buffer at default batch** (248k vocab). `-b 512` always.
- **Speculative flags**: `llama-batched-bench` never has them; `llama-completion` lacks them;
  `llama-cli` HAS them but falls into conversation mode (`-no-cnv` needed; it once dumped
  2.8 GB to stdout). For spec tests use **llama-server + /completion** (also gives
  `draft_n_accepted` in timings).
- **pgrep/pkill self-kill**: a remote `bash -c 'pkill -f X ...'` whose command line contains
  X kills your own ssh. Read numeric PIDs first, `kill <pid>` in a second command.
- **Never edit a running bash script** — bash re-reads it mid-execution. Deploy `parN+1.sh`,
  kill old by numeric PID, relaunch.
- **ssh channels linger** when a remote background job is started even with setsid/nohup —
  wrap in `timeout 25 ssh ... </dev/null` and verify state with a separate probe. Long
  local timeouts get backgrounded by the harness; just re-probe.
- **GTT spill is invisible in VRAM readings** (evicted buffers stop counting). Judge fit by
  throughput vs llama-bench, never by "free VRAM". `RADV_PERFTEST=nogttspill` is now default.
- **Disk is at 97%** — 2.2 GB free. Don't create big files; /tmp is on the root disk.
- The 5629-token benchmark prompt is `/root/bonsai/prompt8k.txt` (synthetic keyword soup —
  fine for A/B, absolute ppl numbers are meaningless).
- Locale warnings (`LC_ALL: cannot change locale`) are noise; `LC_ALL=C` placates.

## Bookkeeping

- Tasks: #14 tracks collecting recover2/followup results. #13 (b9599 rebase) done.
- Memory files: `vega-rig-environment`, `bonsai-vega-perf-baseline`,
  `vega-gpu-clocks-underclocked`, `vega-moe-multigpu` — update `vega-moe-multigpu` with the
  MTP falsification + env-knob wins, and create one for the serving blueprint when finalized.
- Everything ships to GitHub on every meaningful result — the user wants the work public.
