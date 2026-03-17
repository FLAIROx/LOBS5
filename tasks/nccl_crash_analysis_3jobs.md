# NCCL Deadlock Analysis: 3 Muon Optimizer Jobs

**Date**: 2026-03-10
**Analyst**: Claude Code (Opus 4.6)

---

## 1. Job Summary

| Field          | j2696686                                      | j2715891                                      | j2688717                                      |
|:---------------|:----------------------------------------------|:----------------------------------------------|:----------------------------------------------|
| **Nodes**      | nid[010113,010122,010138-010139,010167,010170-010171,010177] | SAME as j2696686 | nid[010770-010777] |
| **Start**      | 2026-03-10 11:54:59                           | 2026-03-10 21:12:02                           | 2026-03-10 00:06:18                           |
| **End**        | 2026-03-10 20:19:34                           | 2026-03-10 22:04:10                           | 2026-03-10 00:58:30                           |
| **Elapsed**    | 08:24:35                                      | 00:52:08                                      | 00:52:12                                      |
| **Crash step** | 23,437                                        | 23,307 (resumed from 21,968)                  | 1,277                                         |
| **Training**   | ~7h 52min training + 15min stall + 17min init | ~30min training + 15min stall + 7min init     | ~30min training + 15min stall + 7min init     |
| **Exit code**  | FAILED (126:0)                                | FAILED (126:0)                                | FAILED (126:0)                                |
| **Kill cause**  | Step watchdog timeout (900s)                  | Step watchdog timeout (900s)                  | Step watchdog timeout (900s)                  |

---

## 2. Node-to-NID Mapping

### j2696686 & j2715891 (same nodes)

| Node Index | NID       | Role        |
|:-----------|:----------|:------------|
| node0      | nid010113 | Coordinator |
| node1      | nid010122 |             |
| node2      | nid010138 |             |
| node3      | nid010139 |             |
| node4      | nid010167 |             |
| node5      | nid010170 |             |
| node6      | nid010171 |             |
| node7      | nid010177 |             |

### j2688717 (different nodes)

| Node Index | NID       | Role        |
|:-----------|:----------|:------------|
| node0      | nid010770 | Coordinator |
| node1      | nid010771 |             |
| node2      | nid010772 |             |
| node3      | nid010773 |             |
| node4      | nid010774 |             |
| node5      | nid010775 |             |
| node6      | nid010776 |             |
| node7      | nid010777 |             |

---

## 3. Per-Node Error Analysis

### j2696686 (crashed at step 23,437 after 8h)

| Node | NID       | Last tqdm step | Watchdog msg | Other errors |
|:-----|:----------|:---------------|:-------------|:-------------|
| 0    | nid010113 | 23437          | FATAL: watchdog 900s at batch 23437 | None |
| 1    | nid010122 | 23437          | FATAL: watchdog 900s at batch 23437 | None |
| 2    | nid010138 | 23437          | FATAL: watchdog 900s at batch 23437 | None |
| 3    | nid010139 | 23437          | FATAL: watchdog 900s at batch 23437 | None |
| 4    | nid010167 | **23436**      | **JAX coordination service FATAL** | "another task died" + GRPC UNAVAILABLE |
| 5    | nid010170 | 23437          | FATAL: watchdog 900s at batch 23437 | None |
| 6    | nid010171 | 23437          | FATAL: watchdog 900s at batch 23437 | None |
| 7    | nid010177 | 23437          | FATAL: watchdog 900s at batch 23437 | None |

**Key finding**: Node 4 (nid010167) was **1 step behind** all other nodes (23436 vs 23437). It did NOT get the watchdog timeout -- instead it got a JAX coordination service error saying "another task died." This means node 4 was the **lagging node** that caused the NCCL deadlock. All other 7 nodes entered step 23437's collective, waited 900s for node 4, then the watchdog killed them. Node 4 was still stuck completing step 23436 (or the transition to 23437) when the coordinator (node 0) was killed by the watchdog, which broke the coordination service RPC.

### j2715891 (crashed at step 23,307 after 52min, same nodes)

| Node | NID       | Last tqdm step | Watchdog msg |
|:-----|:----------|:---------------|:-------------|
| 0    | nid010113 | 23307          | FATAL: watchdog 900s at batch 23307 |
| 1    | nid010122 | 23307          | FATAL: watchdog 900s at batch 23307 |
| 2    | nid010138 | 23307          | FATAL: watchdog 900s at batch 23307 |
| 3    | nid010139 | **23308**      | FATAL: watchdog 900s at batch **23308** |
| 4    | nid010167 | **23308**      | FATAL: watchdog 900s at batch **23308** |
| 5    | nid010170 | 23307          | FATAL: watchdog 900s at batch 23307 |
| 6    | nid010171 | 23307          | FATAL: watchdog 900s at batch 23307 |
| 7    | nid010177 | **23308**      | FATAL: watchdog 900s at batch **23308** |

**Key finding**: 5 nodes (0,1,2,5,6) stalled at batch 23307, while 3 nodes (3,4,7) reached batch 23308. The deadlock split is 5:3 -- a subset of nodes advanced one step further than the rest. No single node is uniquely behind. No NCCL WARN, no FATAL coordinator errors -- just the watchdog on all nodes.

### j2688717 (crashed at step 1,277 after 30min, different nodes)

| Node | NID       | Last tqdm step | Watchdog msg |
|:-----|:----------|:---------------|:-------------|
| 0    | nid010770 | 1277           | FATAL: watchdog 900s at batch 1277 |
| 1    | nid010771 | 1277           | FATAL: watchdog 900s at batch 1277 |
| 2    | nid010772 | 1277           | FATAL: watchdog 900s at batch 1277 |
| 3    | nid010773 | **1278**       | FATAL: watchdog 900s at batch **1278** |
| 4    | nid010774 | 1277           | FATAL: watchdog 900s at batch 1277 |
| 5    | nid010775 | **1278**       | FATAL: watchdog 900s at batch **1278** |
| 6    | nid010776 | 1277           | FATAL: watchdog 900s at batch 1277 |
| 7    | nid010777 | 1277           | FATAL: watchdog 900s at batch 1277 |

**Key finding**: 6 nodes stalled at batch 1277, 2 nodes (3,5) at batch 1278. Again a split deadlock, no single culprit node. Completely different hardware (nid010770-010777) than the other two jobs.

---

## 4. Cross-Job Comparison

### Is there a common bad node?

**NO.** The three jobs used two completely different sets of nodes:

```
j2696686 & j2715891: nid010113, nid010122, nid010138, nid010139, nid010167, nid010170, nid010171, nid010177
j2688717:            nid010770, nid010771, nid010772, nid010773, nid010774, nid010775, nid010776, nid010777
```

- j2688717 crashed on **completely different hardware** than j2696686/j2715891
- No node overlap between the two sets
- Therefore, no common bad node exists across all three crashes

### Within j2696686 & j2715891 (same nodes)?

| Analysis | j2696686 | j2715891 |
|:---------|:---------|:---------|
| Lagging node(s) | node4 (nid010167) -- 1 step behind | nodes 0,1,2,5,6 behind; nodes 3,4,7 ahead |
| nid010167 status | **Lagging (behind)** | **Ahead (23308)** |

Node 4 (nid010167) was the lagging node in j2696686 but was among the **ahead** nodes in j2715891. This contradicts a bad-node hypothesis for nid010167.

---

## 5. Speed Spike Analysis

All three jobs show periodic speed spikes (1.3-1.5s/it vs steady-state 1.21s/it). These are evenly distributed across ALL nodes -- no single node is uniquely slow.

| Job | Total spikes >1.5s/it per node | Distribution |
|:----|:-------------------------------|:-------------|
| j2696686 | ~15-20 per node | Even across all 8 nodes |
| j2715891 | 16-20 per node | Even (node0 highest at 20, node4 lowest at 16) |
| j2688717 | 19-22 per node | Even across all 8 nodes |

The spikes are consistent with periodic collective synchronization jitter, NOT hardware-specific issues.

---

## 6. Checkpoint Save Correlation

j2696686 saved mid-epoch checkpoints every ~1470 steps (~30 minutes):
```
Steps: 1342, 2816, 4292, 5766, 7241, 8714, 10189, 11663, 13139, 14613, 16082, 17554, 19023, 20495, 21967
```

The crash at step 23437 is exactly 1470 steps after the last save at step 21967 -- right when the next checkpoint save was due. However, checkpoint saves do NOT cause speed spikes at the save step itself (1.20-1.22s/it at save steps).

For j2715891 and j2688717, NO checkpoint saves occurred before the crash (too early in the run).

---

## 7. Root Cause Assessment

### Conclusion: Software-level NCCL deadlock, NOT bad hardware

Evidence:
1. **Different hardware, same failure mode**: j2688717 on nid010770-777 crashed identically to j2696686/j2715891 on nid010113-177
2. **No consistent culprit node**: The "lagging" node differs between crashes (node4 in j2696686, no clear leader in the other two)
3. **No hardware error signatures**: Zero ECC errors, zero Xid errors, zero CUDA illegal address errors, zero SIGABRT across all 24 node logs
4. **Clean NCCL -- no NCCL WARN or ERROR**: All NCCL messages are INFO-level only
5. **Pure watchdog timeouts**: All crashes are the step watchdog detecting a 900s stall during a collective operation
6. **Speed spikes evenly distributed**: No single node shows anomalous latency

### Most likely cause: Muon optimizer collective deadlock

The Muon optimizer requires additional all-reduce/all-gather operations (for the Newton-Schulz orthogonalization step) beyond the standard gradient all-reduce. This increases the number of NCCL collective calls per step and the probability of a deadlock, especially at 8-node (32 GPU) scale.

The deadlock pattern (some nodes 1 step ahead of others, all stuck in collectives) is consistent with a **race condition in collective ordering** rather than hardware failure.

### Contributing factors:
- **360M model on 32 GPUs**: Large model with many collective operations per step
- **Muon optimizer**: Additional collective operations for weight orthogonalization
- **Non-contiguous nodes**: The nid010113-177 set spans a wide range (not contiguous), increasing inter-node network latency variance
- **NCCL terminate_on_error=true with 600s timeout**: These flags are set, but the failure manifests as a deadlock (all processes alive but stuck), not an error that NCCL can detect

---

## 8. Recommendations

1. **No nodes to add to --exclude list** -- this is not a bad node issue
2. **Investigate Muon optimizer collective pattern** -- the additional all-reduce for Newton-Schulz may need explicit barrier synchronization
3. **Consider reducing NCCL timeout to detect deadlocks faster** -- current 600s NCCL timeout + 900s watchdog = 25min of wasted GPU-hours per crash
4. **Test with AdamW on same nodes** -- if AdamW doesn't deadlock on nid010113-177, that confirms Muon's collectives as the root cause
5. **Check if Muon's all-reduce is inside or outside the JAX-compiled function** -- if outside, it may not benefit from XLA's collective ordering guarantees
