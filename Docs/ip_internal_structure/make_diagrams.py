"""Render IP internal-structure diagrams for the INT8 fixed-point C1 and C3 HLS IPs.

Every number drawn here is taken from one of two places, never estimated:
  * the HLS source (loop bounds, PE_COUNT, pragmas, array shapes):
      hls/conv_c1/src/conv_c1_int8_fixedpoint.{h,cpp}
      hls/conv_c3/src/conv_c3_int8_fixedpoint.{h,cpp}
  * the Vitis HLS csynth / Bind Op Report figures for the matching solutions
    (conv_c1_int8_fixedpoint_proj, conv_c3_int8_fixedpoint_proj), as recorded in
    Results/hls_results.md. The raw csynth.rpt files themselves are not in the repo
    (all *proj*/ directories are gitignored).

Run:  python3 Docs/ip_internal_structure/make_diagrams.py
Outputs c1_internal_structure.png, c3_internal_structure.png and
IP_Internal_Structure.pdf next to this script.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Category colours (fill, edge)
C_IF = ("#E8EEF7", "#3B5B8C")     # interfaces / external memory
C_BUF = ("#EAF4EA", "#3E7B3E")    # on-chip buffers / registers
C_PE = ("#FFF1DC", "#B7701A")     # MAC processing elements
C_ACC = ("#F3E9F7", "#7A4A8F")    # accumulate / ReLU / requantize
C_LUT = ("#FDEBEB", "#A33A3A")    # LUT decode
C_NOTE = ("#F6F6F6", "#8A8A8A")   # annotation panels

TXT = "#1E1E1E"


def box(ax, x, y, w, h, title, lines=(), color=C_BUF, dsp=None, title_size=11,
        body_size=8.8, dashed=False, align="center"):
    fill, edge = color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fill, ec=edge, lw=1.6, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=TXT)
    if lines:
        tx = x + w / 2 if align == "center" else x + 2.0
        ax.text(tx, y + h - 1.2 - title_size * 0.30, "\n".join(lines), ha=align,
                va="top", fontsize=body_size, color=TXT, linespacing=1.35)
    if dsp is not None:
        badge = f"{dsp} DSP"
        ax.text(x + w - 0.6, y + 0.6, badge, ha="right", va="bottom", fontsize=8.2,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=edge, ec=edge))


def arrow(ax, p0, p1, color="#444444", lw=1.5, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}"))


def new_canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 10), dpi=100)          # 1600 x 1000 px
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(80, 97.5, title, ha="center", va="top", fontsize=17, fontweight="bold", color=TXT)
    ax.text(80, 93.6, subtitle, ha="center", va="top", fontsize=10.5, color="#444444")
    return fig, ax


def pe_array(ax, x, y, w, h, n, pe_title, pe_line, dsp_each, header, header_lines):
    """Draw a dashed container with n stacked PE boxes; return list of (left, right, ymid)."""
    box(ax, x, y, w, h, header, header_lines, color=C_PE, dashed=True, title_size=11.5)
    top = y + h - 9.0
    gap = 0.9
    pe_h = (top - (y + 1.2) - gap * (n - 1)) / n
    ports = []
    for p in range(n):
        py = top - (p + 1) * pe_h - p * gap
        px, pw = x + 2.0, w - 4.0
        fill, edge = C_PE
        ax.add_patch(FancyBboxPatch((px, py), pw, pe_h, boxstyle="round,pad=0.1,rounding_size=0.5",
                                    fc="white", ec=edge, lw=1.2))
        ax.text(px + 1.0, py + pe_h / 2, pe_title.format(p=p), ha="left", va="center",
                fontsize=8.8, fontweight="bold", color=TXT)
        ax.text(px + pw * 0.40, py + pe_h / 2, pe_line.format(p=p), ha="left", va="center",
                fontsize=7.9, color=TXT)
        ax.text(px + pw - 0.6, py + pe_h / 2, f"{dsp_each} DSP", ha="right", va="center",
                fontsize=7.6, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=edge, ec=edge))
        ports.append((px, px + pw, py + pe_h / 2))
    return ports


def summary_table(ax, x, y, w, h, title, rows):
    box(ax, x, y, w, h, title, color=C_NOTE, title_size=11)
    row_h = (h - 5.0) / len(rows)
    for i, (k, v) in enumerate(rows):
        ry = y + h - 5.2 - i * row_h - row_h / 2
        ax.text(x + 1.5, ry, k, ha="left", va="center", fontsize=9, color="#333333")
        ax.text(x + w - 1.5, ry, v, ha="right", va="center", fontsize=9, fontweight="bold",
                color=TXT)


def legend(ax, x, y):
    items = [("Interface / external memory", C_IF), ("On-chip buffer / registers", C_BUF),
             ("Index decode (LUT)", C_LUT), ("MAC processing element", C_PE),
             ("Accumulate / requantize", C_ACC)]
    for i, (label, (fill, edge)) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x + i * 31, y), 2.2, 1.6, boxstyle="round,pad=0.1",
                                    fc=fill, ec=edge, lw=1.2))
        ax.text(x + i * 31 + 3.0, y + 0.8, label, ha="left", va="center", fontsize=8.6,
                color="#333333")


def requant_chain(ax, x, top, w, n_pe, rescale_core, acc_note):
    """Accumulate -> ReLU -> fixed-point rescale -> round/saturate, stacked top to bottom.
    Returns (left-mid of accumulator box, bottom-mid of last box, right-mid of last box)."""
    h_acc, h_relu, h_mul, h_rnd, g = 10.0, 6.6, 11.0, 10.0, 2.6
    y_acc = top - h_acc
    box(ax, x, y_acc, w, h_acc, "Partial-sum reduction",
        [f"acc = bias[o] + Σ partial_sum[0..{n_pe - 1}]", acc_note,
         "adds on fabric (0 DSP in Bind Op)"], color=C_ACC, dsp=0)
    y_relu = y_acc - g - h_relu
    box(ax, x, y_relu, w, h_relu, "ReLU on int32 acc",
        ["acc_relu = (acc < 0) ? 0 : acc"], color=C_ACC, dsp=0)
    y_mul = y_relu - g - h_mul
    box(ax, x, y_mul, w, h_mul, "Fixed-point rescale multiply",
        ["prod = acc_relu × requant_mult (Q31)", rescale_core,
         "only multiplier outside the PE array"], color=C_ACC, dsp=3)
    y_rnd = y_mul - g - h_rnd
    box(ax, x, y_rnd, w, h_rnd, "Round + shift + saturate",
        ["(prod + 2^(S-1)) >> requant_shift", "clip to [-128, 127] → ap_int<8>",
         "integer add/shift on fabric"], color=C_ACC, dsp=0)
    cx = x + w / 2
    arrow(ax, (cx, y_acc), (cx, y_relu + h_relu))
    arrow(ax, (cx, y_relu), (cx, y_mul + h_mul))
    arrow(ax, (cx, y_mul), (cx, y_rnd + h_rnd))
    return (x, y_acc + h_acc / 2), (cx, y_rnd), (x + w, y_rnd + h_rnd / 2)


# ---------------------------------------------------------------------------
# C1 -- conv_c1_int8_fixedpoint (conv_c1_int8_fixedpoint_proj/solution1)
# ---------------------------------------------------------------------------
def draw_c1(path):
    fig, ax = new_canvas(
        "C1 IP internal structure — conv_c1_int8_fixedpoint (8-PE systolic MAC array)",
        "Solution conv_c1_int8_fixedpoint_proj/solution1  •  xc7z020clg400-1 (PYNQ-Z2)  "
        "•  10 ns clock  •  28×28×1 → 5×5 conv, 'same' pad "
        "→ 28×28×6, int8 × int8 → int32")
    top = 88.5

    # Column A: AXI interfaces + burst-copy local buffers
    xa, wa = 2.0, 27.0
    box(ax, xa, top - 11, wa, 11, "s_axilite  (bundle=control)",
        ["base addrs of input/weights/bias/output", "requant_mult  ap_int<32>  (Q31 M)",
         "requant_shift ap_int<8>   (S)"], color=C_IF)
    box(ax, xa, top - 26, wa, 12, "m_axi gmem0 / 1 / 2",
        ["input   28×28×1  int8", "weights 5×5×1×6  int8",
         "bias    6  int32  (pre-quantized)", "burst-copy into local buffers"], color=C_IF)
    box(ax, xa, top - 44, wa, 15, "Local on-chip buffers",
        ["local_input  [28][28][1]  int8", "local_weights [5][5][1][6]",
         "  ARRAY_PARTITION complete → regs", "local_bias [6]  int32",
         "local_output [28][28][6]  int8"], color=C_BUF)
    arrow(ax, (xa + wa / 2, top - 26), (xa + wa / 2, top - 29))

    # Column B: line buffer + window
    xb, wb = 34.0, 27.0
    box(ax, xb, top - 17, wb, 17, "Line buffer",
        ["line_buf [K=5][IN_W+2·PAD=32][1]", "ARRAY_PARTITION complete dim=0",
         "shift rows up (UNROLL), load new row", "fed row-by-row from local_input",
         "zero-fill for 'same' padding (PAD=2)"],
        color=C_BUF)
    box(ax, xb, top - 33, wb, 12, "Sliding window",
        ["window [5][5][1]  int8", "ARRAY_PARTITION complete dim=0",
         "reloaded per out_c from line_buf"], color=C_BUF)
    arrow(ax, (xa + wa, top - 36.5), (xb, top - 8.5), rad=-0.15)
    arrow(ax, (xb + wb / 2, top - 17), (xb + wb / 2, top - 21))

    # Column C: PE array
    xc, wc = 66.0, 45.0
    ports = pe_array(
        ax, xc, 25.0, wc, top - 25.0, 8, "PE {p}", "partial_sum[{p}] += win × w",
        1, "PE array  —  PE_COUNT = 8  (8 DSP total)",
        ["8 × mac_muladd_8s_8s_32s_32_4_1  (1 DSP / PE)",
         "spatially unrolled (#pragma HLS UNROLL on p)"])
    arrow(ax, (xb + wb, top - 27), (xc, top - 30))
    ax.text(xb + wb + 0.4, top - 25.0, "window", fontsize=7.8, color="#555555")
    arrow(ax, (xa + wa, top - 38), (xc, top - 44), rad=0.12)
    ax.text(xa + wa + 7, top - 45.3, "local_weights (regs)", fontsize=7.8, color="#555555")

    # Column D: reduction + requantize chain
    xd, wd = 116.0, 42.0
    acc_in, chain_bottom, chain_right = requant_chain(
        ax, xd, top, wd, 8, "mul_31ns_32s_63_2_1  (3-DSP integer core)",
        "8 partial sums, ap_int<32>")
    for (_, right, ym) in ports:
        arrow(ax, (right, ym), acc_in, lw=0.9, color="#8A6A3A")

    # Output buffer + AXI out, under the requant chain
    box(ax, xd, 25.0, wd / 2 - 1, 10, "local_output",
        ["[28][28][6] int8", "written per (r,c,o)"], color=C_BUF)
    box(ax, xd + wd / 2 + 1, 25.0, wd / 2 - 1, 10, "m_axi gmem3",
        ["output 28×28×6", "burst-copy to DDR"], color=C_IF)
    arrow(ax, chain_bottom, (xd + wd / 4, 35.0))
    arrow(ax, (xd + wd / 2 - 1, 30.0), (xd + wd / 2 + 1, 30.0))

    # Bottom band: loop nest + resources
    box(ax, 2.0, 4.5, 90, 17.5, "Loop nest & pipelining (from source + csynth)",
        ["for out_r in 0..27  {  shift line_buf  }",
         "  for out_c in 0..27  {  load window  }",
         "    for o in 0..5  (OUT_C = 6)",
         "      for m in 0..3  (MACS_PER_PE = ⌈25/8⌉ = 4)   #pragma HLS PIPELINE II=1",
         "        for p in 0..7  #pragma HLS UNROLL  → mac_idx = m·8+p  (if < 25)",
         "25 MACs/pixel (5×5×1) over 32 PE slots • decode via / and % by K·IN_C, K"],
        color=C_NOTE, body_size=8.9, align="left")
    summary_table(ax, 96.0, 4.5, 62, 17.5, "csynth results (recorded in Results/hls_results.md)", [
        ("Latency", "124,991 cycles"),
        ("Timing @10 ns / Est. Fmax", "slack 0.00 ns / 136.99 MHz"),
        ("DSP", "11 (5%) = 8 MAC + 3 rescale"),
        ("BRAM_18K / FF / LUT", "5 (1%) / 9,889 (9%) / 16,124 (30%)"),
    ])
    legend(ax, 4.0, 1.0)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C3 -- conv_c3_int8_fixedpoint (conv_c3_int8_fixedpoint_proj/solution1)
# ---------------------------------------------------------------------------
def draw_c3(path):
    fig, ax = new_canvas(
        "C3 IP internal structure — conv_c3_int8_fixedpoint (6-PE partial-sum split + LUT index decode)",
        "Solution conv_c3_int8_fixedpoint_proj/solution1  •  xc7z020clg400-1 (PYNQ-Z2)  "
        "•  10 ns clock  •  14×14×6 → 5×5 conv, 'valid' "
        "→ 10×10×16, int8 × int8 → int32")
    top = 88.5

    # Column A: LUT decode (drives the addresses)
    xa, wa = 2.0, 27.0
    box(ax, xa, top - 12, wa, 12, "mac_idx generator",
        ["mac_idx = m · 6 + p", "valid if mac_idx < 150", "(TOTAL_MACS = 6·5·5)"],
        color=C_LUT)
    box(ax, xa, top - 34, wa, 18, "LUT-based index decode",
        ["i_lut [150]   → i  (0..5)", "kr_lut[150]   → kr (0..4)",
         "kc_lut[150]   → kc (0..4)", "static const, ARRAY_PARTITION", "complete dim=0",
         "replaces / and % by K=5 (no dividers)"], color=C_LUT)
    arrow(ax, (xa + wa / 2, top - 12), (xa + wa / 2, top - 16))
    box(ax, xa, top - 47, wa, 9, "Address generation",
        ["input  [r+kr][c+kc][i]", "weights[kr][kc][i][o]"], color=C_LUT)
    arrow(ax, (xa + wa / 2, top - 34), (xa + wa / 2, top - 38))

    # Column B: plain ap_memory arrays + scalars
    xb, wb = 34.0, 27.0
    box(ax, xb, top - 10, wb, 10, "Scalar inputs",
        ["requant_mult  ap_int<32>  (Q31 M)", "requant_shift ap_int<8>   (S)",
         "(no INTERFACE pragmas in source)"], color=C_IF)
    box(ax, xb, top - 26, wb, 13, "input  (ap_memory array)",
        ["[14][14][6]  ap_int<8>", "read as input[r+kr][c+kc][i]",
         "no line buffer / no local copy", "BRAM 0 in report"], color=C_IF)
    box(ax, xb, top - 42, wb, 13, "weights  (ap_memory array)",
        ["[5][5][6][16]  ap_int<8>", "read as weights[kr][kc][i][o]",
         "bias [16]  ap_int<32>", "(pre-quantized)"], color=C_IF)
    arrow(ax, (xa + wa, top - 41.5), (xb, top - 19.5), rad=0.15)
    arrow(ax, (xa + wa, top - 43.5), (xb, top - 35.5), rad=0.1)
    ax.text(xa + 1.0, top - 50.5, "addresses → input / weights", fontsize=7.8, color="#555555")

    # Column C: PE array
    xc, wc = 66.0, 45.0
    ports = pe_array(
        ax, xc, 25.0, wc, top - 25.0, 6, "PE {p}", "partial_sum[{p}] += x × w",
        1, "PE array  —  PE_COUNT = 6  (6 DSP total)",
        ["6 × mac_muladd_8s_8s_20s  (1 DSP / PE)",
         "spatially unrolled (#pragma HLS UNROLL on p)"])
    arrow(ax, (xb + wb, top - 19.5), (xc, top - 20))
    arrow(ax, (xb + wb, top - 35.5), (xc, top - 44), rad=0.1)
    ax.text(xb + 1.0, top - 46.0, "6 (x, w) operand pairs per m-iteration\n→ one pair to each PE",
            fontsize=7.8, color="#555555")

    # Column D: reduction + requantize chain
    xd, wd = 116.0, 42.0
    acc_in, chain_bottom, _ = requant_chain(
        ax, xd, top, wd, 6, "mul_31ns_32s_63  (3-DSP integer core)",
        "6 partial sums, ap_int<32>")
    for (_, right, ym) in ports:
        arrow(ax, (right, ym), acc_in, lw=0.9, color="#8A6A3A")

    box(ax, xd, 25.0, wd, 10, "output  (ap_memory array)",
        ["[10][10][16]  ap_int<8>", "written as output[r][c][o]"], color=C_IF)
    arrow(ax, chain_bottom, (xd + wd / 2, 35.0))

    box(ax, 2.0, 4.5, 90, 17.5, "Loop nest & pipelining (from source + csynth)",
        ["for o in 0..15  (OUT_C = 16)",
         "  for r in 0..9 ,  for c in 0..9   (OUT_H = OUT_W = 10)",
         "    for m in 0..24  (MACS_PER_PE = 150/6 = 25)   #pragma HLS PIPELINE II=1",
         "      for p in 0..5  #pragma HLS UNROLL",
         "MAC pipeline VITIS_LOOP_61_5: trip 25, II achieved 3 / target 1",
         "(int32 accumulator recurrence; 'loop constraints NOT satisfied')"],
        color=C_NOTE, body_size=8.9, align="left")
    summary_table(ax, 96.0, 4.5, 62, 17.5, "csynth results (recorded in Results/hls_results.md)", [
        ("Latency", "150,401 cycles"),
        ("Timing @10 ns / Est. Fmax", "slack 0.00 ns / 136.99 MHz"),
        ("DSP", "9 (4%) = 6 MAC + 3 rescale"),
        ("BRAM_18K / FF / LUT", "0 / 1,523 (1%) / 4,919 (9%)"),
    ])
    legend(ax, 4.0, 1.0)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


CAPTIONS = [
    ("c1_internal_structure.png",
     "Figure 1 — C1 (conv_c1_int8_fixedpoint). PE_COUNT = 8 spatially-unrolled int8 MAC "
     "PEs (8× mac_muladd_8s_8s_32s_32_4_1, 1 DSP each) fed from a fully partitioned 5×32 "
     "line buffer and 5×5 window; fixed-point requantize via one mul_31ns_32s_63_2_1 (3 DSP). "
     "Totals: DSP 11 (5%), BRAM_18K 5 (1%), FF 9,889 (9%), LUT 16,124 (30%); latency 124,991 "
     "cycles; slack 0.00 ns at 10 ns, est. Fmax 136.99 MHz; xc7z020clg400-1."),
    ("c3_internal_structure.png",
     "Figure 2 — C3 (conv_c3_int8_fixedpoint). PE_COUNT = 6 partial-sum split "
     "(6× mac_muladd_8s_8s_20s, 1 DSP each) with division-free (i, kr, kc) decode from three "
     "150-entry fully partitioned LUTs; fixed-point requantize via one mul_31ns_32s_63 (3 DSP). "
     "MAC loop VITIS_LOOP_61_5: trip 25, II 3 (target 1). Totals: DSP 9 (4%), BRAM 0, FF 1,523 "
     "(1%), LUT 4,919 (9%); latency 150,401 cycles; slack 0.00 ns, est. Fmax 136.99 MHz."),
]


def build_pdf(pdf_path):
    import textwrap
    with PdfPages(pdf_path) as pdf:
        for fname, caption in CAPTIONS:
            fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            img = mpimg.imread(os.path.join(OUT_DIR, fname))
            ax = fig.add_axes([0.03, 0.17, 0.94, 0.80])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.5, 0.135, "\n".join(textwrap.wrap(caption, 150)), ha="center", va="top",
                     fontsize=9.5, linespacing=1.4)
            fig.text(0.5, 0.03, "Source: HLS source in hls/conv_c1/src, hls/conv_c3/src; "
                     "csynth / Bind Op Report figures as recorded in Results/hls_results.md.",
                     ha="center", fontsize=7.5, color="#666666")
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    draw_c1(os.path.join(OUT_DIR, "c1_internal_structure.png"))
    draw_c3(os.path.join(OUT_DIR, "c3_internal_structure.png"))
    build_pdf(os.path.join(OUT_DIR, "IP_Internal_Structure.pdf"))
    print("wrote diagrams to", OUT_DIR)
