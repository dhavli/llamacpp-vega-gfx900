# HANDOFF — Vega 7-GPU inference tuning (2026-08-02, ~08:30 UTC)

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
  `/nix/store/scb4cmx0h15sfbrapkjyx0r5jrzv8gpi-vega-runtime` (pre-server-fix b9599:
  `0rlr67bjyf76wfzf1mmzgq73pk22fk5x`; older b9596:
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

- `recover2.sh`, `followup.sh`, and the corrected `ppl-tile-ab.sh` have finished; sentinels
  `### RECOVERDONE`, `### FOLLOWUPDONE`, and `### P4DONE` are present. No benchmark should be
  active. Logs: `/root/bonsai/{recover2,followup,ppl-tile-ab}.log`.
- Pipeline parallelism is enabled. The env stacks are non-additive. The Qwen TILE_M candidate
  is falsified by perplexity. These results are recorded in `benchmarks/llm/index.jsonl`.
- The expert-count quality A/B and corrected task run are complete. Top-6 passed the narrow
  experimental gate (PPL +0.30%, three complete tasks quality-equivalent).
- A server-only 8.6× decode regression was found and fixed behind
  `LLAMA_SERVER_FULL_OUTPUT_RESERVE=1`: 3.77 → 29.70 TG, identical text. The new runtime is
  deployed. The actual 4×128k pod-A server shape also loads and runs four requests, but
  per-stream PP/TG falls below the explicit floor under concurrency; see below.

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
| `LLAMA_SERVER_FULL_OUTPUT_RESERVE=1` | server 3.77 → 29.70 TG (7.9×), short PP 22.8 → 134.2; restores parity with completion (32.47 TG) |
| Real server 4×128k concurrency | all four requests complete, but streams get only 130-492 PP / 3.4-13.0 TG; serialize to one active request/pod for the ≥500/25 SLA |
| GDN/SSM per-op profile | direct GATED_DELTA_NET + SSM_CONV = 1.04 ms/token, only 2.9% of wall; kernel work here is low value |
| Vulkan/top-k MoE fusion is active | disabling fusion: 519.69/29.44 → 507.66/24.87, byte-identical; perf log names TOPK_MOE_EARLY_SOFTMAX_NORM |
| `-b 4096 -ub 256` | PP 449 vs 519 base — smaller ubatch hurts; `-ub 1024` run produced no output (check `/tmp/qa-ub1024.log` for the failure mode) |
| Perplexity runs OOM the box | 248320 vocab × batch × 4 B logits buffer. Default 2048 (2 GB) and even `-b 512` (508 MB) OOM-kill this host. **Always use `--no-mmap -b 128`**; P4 completed with ~0.9 GB minimum available RAM. |
| PCIe survey | all 7 cards Gen2 x1; decode traffic ~4 KB/tok/boundary (fine), prefill ~MB/ubatch/boundary (why MoE prefill decays with card count) |
| MoE vs dense splitting | MoE prefill DECREASES with cards (641/561/506 @ pp2048 for 3/4/6); dense increases at depth. Decode optimum 4 cards, prefill optimum 3 |

Earlier falsifications (README "Falsified" section): fast-but-wrong warptiles (TM=8 etc. —
**never** trust throughput without byte-identical greedy output), VGPR diet, LDS→ALU,
f16 accumulators, `-sm row` on Vulkan (CUDA-only).

## Followup verdicts

1. **Pipeline parallelism: CONFIRMED ENABLED.** The line appears twice with `--verbose`; it
   was merely hidden at normal verbosity. No buffer-allocation fallback occurred, so the
   MoE-prefill-decay finding and prior batch/ubatch sweep remain valid.
2. **TILE_M candidate `256,128,64,64,32,64,2,4,4,1,64`: FALSIFIED for Qwen Q4_K.**
   RAM-safe P4 (`--no-mmap -b 128`) measured PPL 3,583,950.87 vs control 131.0328. The
   ~140-line greedy prefix was misleading. Never use this override for Qwen; it remains
   independently verified for the Bonsai Q1_0 path only. `-b 512` still OOM-killed this
   3.8 GB host; use `--no-mmap -b 128` for all further Qwen perplexity runs.
3. **Stacked combos: confirmed non-additive.** gfx+gtt = 32.08 TG; adding transfer queue =
   33.78, both byte-identical but below graphics queue alone at 34.87. Use `nogttspill` as a
   conservative fit/fail policy, not as the peak-throughput configuration.
4. **`expert_used_count=6`: EXPERIMENTAL PASS.** 559.1 PP / 31.13 TG versus 519.69 / 29.44
   control (+7.6% / +5.7%). PPL is 131.422 vs 131.033 (+0.30%, inside error); both configs
   correctly answered the rate, Python mutable-default, and HTTP PUT tasks. The prime task
   truncated in both and was excluded. Expose top-6 as optional, keep top-8 as default.
5. **llama-server graph reservation: FIXED.** Server forces `n_outputs_max=n_parallel`, which
   made one-slot Qwen decode 3.77 TG while identical `llama-completion` reached 32.47.
   `LLAMA_SERVER_FULL_OUTPUT_RESERVE=1` restores 29.70 TG and identical text. Disabling
   checkpoints/cache did not move TG. Export the env for every serving pod.

## PP/TG improvement paths — tried, in-flight, and UNTRIED

### In-flight / next in queue
- Production-shape A/B of graphics queue alone versus conservative nogttspill stacks; the
  real 4×128k A/B is now complete: graphics changes total wall time only 53.29 → 52.60 s.
  Next serving gate is the post-16GB two-pod run with one admitted request per pod.

### Untried — ordered by expected value/effort (from the research pass + local analysis)
1. **HBM2 memory clock + timing straps** (decode lever, mining-proven on these exact cards):
   `amdgpu.ppfeaturemask=0xffffffff` kernel arg, then `echo "m 3 1020" > pp_od_clk_voltage`
   (Vega 56 stock 800; Samsung dies do 1020-1100 — check die vendor via `amdmemtweak --current`;
   Hynix: stay ≤945) and Eliovp `amdmemtweak` straps (full Samsung strap set in ledger notes /
   research report). Our Q1_0 matvec is HBM-LATENCY-bound (102 of 410 GB/s used, waves stall
   a full HBM latency per row) — straps cut latency, so this may move Bonsai TG well past the
   +8-16% clock scaling. **HBM2 has no ECC here: md5-verify greedy output after EVERY step.**
2. **GDN/SSM ops: PROFILED, LOW VALUE.** Direct GATED_DELTA_NET + SSM_CONV total 1.04 ms/token
   across all three devices (4.0% of GPU-op time, 2.9% wall). The profiler requires pipeline
   to be disabled via an unmatched tensor override; both normal and concurrent logger modes
   assert during warmup otherwise. Do not prioritize these kernels.
3. **topk_moe fusion: CONFIRMED ACTIVE.** `GGML_VK_DISABLE_FUSION=1` drops decode 29.44 →
   24.87 and PP 519.69 → 507.66 with byte-identical output. The perf log explicitly names
   `TOPK_MOE_EARLY_SOFTMAX_NORM`, so the suspected view-node matcher miss is falsified.
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
  Late divergence (>100 identical lines) is only a reason to adjudicate, not evidence of
  correctness: the falsified TILE_M candidate matched ~140 lines before PPL exploded.
- **Perplexity = 2 GB host logits buffer at default batch** (248k vocab), and `-b 512` also
  OOM-kills this host. Use `--no-mmap -b 128`; see `benchmarks/llm/ppl-tile-ab.sh`.
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
