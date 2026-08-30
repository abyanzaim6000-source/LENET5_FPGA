Baseline (C1 convolution, no pragmas) — recorded numbers
Metric	Value
Latency	61,299 cycles
Latency (time)	613,000 ns ≈ 0.613 ms
Initiation Interval (II)	13
Trip Count	4,704 (= 28×28×6 output pixels)
DSP	10 (4% of chip)
FF	5,303 (4%)
LUT	5,639 (10%)
BRAM	0 used
Timing slack	-1.12 (negative → see below)
Pragmas used	none


## C1 Convolution — HLS Optimization Log

| Experiment | Latency (cycles) | II | DSP | FF | LUT | Notes |
|---|---|---|---|---|---|---|
| Baseline (no pragmas) | 61,299 | 13 | 10 | 5,303 | 5,639 | input & weights both single-port memory |
| ARRAY_PARTITION on weights only | 61,299 | 13 | 10 | 10,752 | 5,972 | No improvement — weights was not the bottleneck; input array (784 elements, still 2 ports) is now the limiting factor |

Comparing baseline vs after weights partitioning
Metric	Baseline	After partitioning weights	Change
Latency	61,299 cycles	61,299 cycles	No change
II	13	13	No change
DSP	10	10	No change
FF	5,303	10,752	+103%
LUT	5,639	5,972	+6%