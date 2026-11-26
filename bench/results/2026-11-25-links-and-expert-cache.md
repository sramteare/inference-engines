# GPU links and DeepSeek expert cache (2026-11-25)

Two RTX PRO 6000 Blackwell Max-Q cards on PCIe 5.0 x16, with 2 channels of DDR5-6400 (246 GiB). These numbers back the plan doc "Basalt: two cards as one pool". GB/s means 1e9 bytes per second.

## Links: `bench/gpulinks.cu --mib 1024`

Qwen on GPU 0 was serving a request during this run. That is a likely reason, not checked, for why "0 reads 1" is half of "1 reads 0".

```
buffer: 1024 MiB per GPU; local-read and host-copy numbers are only meaningful when the buffer is bigger than the GPU L2 (~128 MB) and the CPU L3 (~64 MB), so use --mib 512 or more for those
p2p-access: canAccessPeer 0->1 yes, 1->0 yes (enabled both ways)
gpu0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 188 SMs
copy 0->1                  median    52.13 GB/s   best    52.56 GB/s
copy 1->0                  median    52.01 GB/s   best    53.38 GB/s
copy 0<->1 both at once    median   100.94 GB/s   best   102.54 GB/s
read remote 0 reads 1      median    24.58 GB/s   best    26.16 GB/s
read remote 1 reads 0      median    50.17 GB/s   best    51.45 GB/s
read local 0               median  1583.76 GB/s   best  1617.16 GB/s
read local 1               median  1633.42 GB/s   best  1638.40 GB/s
h2d gpu0                   median    56.98 GB/s   best    57.00 GB/s
h2d gpu1                   median    56.98 GB/s   best    57.03 GB/s
h2d both at once           median    94.20 GB/s   best    96.03 GB/s
d2h gpu0                   median    55.49 GB/s   best    55.58 GB/s
d2h gpu1                   median    56.41 GB/s   best    56.50 GB/s
done
```

## Expert-cache refill: `bench/gpulinks.cu --blob 11 --total 4096 --ring 1024`

These are copies of 11 MiB each, the size of one expert. The refill line is for DeepSeek's 61.19 GiB set of fixed experts on a card. This run is after the fix that sets each card's id in blob mode. Before it, the 4-stream, both-at-once and peer lines ran with the wrong card selected, but they gave the same numbers. GPU 1 was 26% busy with other work when the run started.

```
blob mode: 11 MiB blobs x 372 copies = 4092 MiB per run, ring 1024 MiB
blob h2d gpu0              median    56.66 GB/s   best    56.67 GB/s   refill 61.19 GiB: 1.2 s
blob h2d gpu1              median    56.64 GB/s   best    56.66 GB/s   refill 61.19 GiB: 1.2 s
blob h2d gpu1 4 streams    median    56.54 GB/s   best    56.57 GB/s   refill 61.19 GiB: 1.2 s
blob h2d both at once      median    95.35 GB/s   best    96.02 GB/s   refill 61.19 GiB/card at half total rate: 1.4 s
blob peer 0->1             median    55.40 GB/s   best    55.54 GB/s   refill 61.19 GiB: 1.2 s
blob peer 1->0             median    55.31 GB/s   best    55.51 GB/s   refill 61.19 GiB: 1.2 s
done
```

## Hit rate with two sessions taking turns: `bench/hitrate.py`

This run used DeepSeek V4.1 Flash (UD-Q2_K_XL) on GPU 1 alone, Basalt 0.5.6. Session A writes Python and session B writes Chinese prose, with 1,000 tokens per request.

The hit rate is the share of routed experts found in VRAM, as Basalt counts it per request. pcie% is the share of all routed experts the GPU read over PCIe or from another GPU. Those are not in the hit count.

Command: `bench/hitrate.py 8001 ~/.cache/basalt/<hash>/engine.log AAABBBABABAB 1000`

Normal server:

```
step sess turn  out  hit%   pcie%  lookups
   0  A     0  1000   93.9    3.0  216681
   1  A     1  1000   94.8    2.6  209855
   2  A     2  1000   95.0    2.4  224985
   3  B     0  1000   95.2    1.5  245617
   4  B     1  1000   97.4    0.6  249140
   5  B     2  1000   96.3    1.1  227574
   6  A     3  1000   93.0    3.6  212773
   7  B     3  1000   95.4    1.4  247383
   8  A     4  1000   92.1    4.6  199496
   9  B     4  1000   94.7    2.0  241076
  10  A     5  1000   92.6    4.1  204556
  11  B     5  1000   94.5    1.9  244151
```

The same plan with `"diag-tier-frozen": true`: the 28 adaptive slots per layer are off, and so is fetching over PCIe.

```
step sess turn  out  hit%   pcie%  lookups
   0  A     0  1000   73.4    0.0  220822
   1  A     1  1000   74.0    0.0  218464
   2  A     2  1000   72.2    0.0  222655
   3  B     0  1000   43.3    0.0  234099
   4  B     1  1000   43.2    0.0  229179
   5  B     2  1000   40.3    0.0  212541
   6  A     3  1000   71.9    0.0  221239
   7  B     3  1000   47.9    0.0  230751
   8  A     4  1000   70.8    0.0  209095
   9  B     4  1000   45.3    0.0  228003
  10  A     5  1000   73.0    0.0  213341
  11  B     5  1000   44.9    0.0  230683
```

The 132 fixed experts per layer are ranked once and suit code. The adaptive slots lift both sessions to 92–97%, and the first request after a switch is already at 95%. The counts are per request, so they don't show how fast the cache adapts within one.
