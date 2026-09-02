## [Today's date] — HLS optimization + first Vivado integration (C1 convolution)

Took the C1 convolution layer through the full HLS optimization pipeline the paper describes: 
started from a plain, unoptimized baseline (II=13, only 2 DSP-equivalent units), diagnosed the 
real memory-port bottleneck (input array, not weights — an assumption that turned out wrong and 
cost some time to discover), built a line-buffer streaming architecture matching the paper's Fig 
2(b)/(c), and reached full II=1 pipelining with 150 fully-parallel MAC units.

Then built a genuine 8-PE systolic variant, matching the paper's exact PE count — first attempt 
accidentally re-created full parallelism due to a PIPELINE pragma placement mistake, second attempt 
correctly time-multiplexed down to just 2 physical multiply/add units, at the cost of II rising to 4 
and Fmax dropping from 137.82MHz to 104.80MHz. This tradeoff is the central resource-vs-throughput 
finding the paper's Section IV-B describes, and now I have real numbers to back it, not just the 
paper's claims.

Converted the IP to AXI interfaces (m_axi + s_axilite) for system integration — hit a serious pitfall 
here: fully-partitioned arrays cannot use m_axi interfaces, and my first attempt generated 150+ 
redundant AXI masters, taking ~20 minutes to synthesize. Fixed by copying AXI-facing arrays into 
local on-chip buffers first, then doing all fast/parallel computation on the local copies — this 
is apparently the standard HLS pattern for this situation.

Successfully built and validated a Zynq block design in Vivado: PS + AXI interconnects + the 
conv_c1_systolic IP, including working through an address-overlap validation error (fixed via 
Address Editor's Auto Assign Address applied across the whole design at once, rather than 
incrementally).

**Not yet done**: PE_COUNT=1/2/4 sweep, bitstream generation, applying this same process to C3, 
generalizing to a reusable/parameterized core, and adding a DMA block to more closely match the 
paper's exact architecture.

**Biggest lessons**: (1) always verify which array is the real bottleneck via the II-violation 
warning rather than assuming; (2) PIPELINE II=1 forces complete unrolling of everything inside it, 
which can silently defeat resource-sharing goals; (3) AXI interfaces and array partitioning are 
fundamentally incompatible on the same array — burst-copy to local buffers is the fix.