This is our final year college project under c2s on the topic Lenet5 FPGA Accelerator.
LeNet-5 FPGA Accelerator (INT8) on PYNQ-Z2

Hardware-accelerated LeNet-5 CNN inference on the PYNQ-Z2 (Zynq-7020) FPGA board, built with Vitis HLS and Vivado. The design uses INT8 fixed-point arithmetic to reduce resource usage and memory bandwidth while keeping accuracy close to the floating-point model.


LeNet-5 is a classic convolutional neural network for handwritten digit recognition. This project implements its inference path as a custom HLS IP core (`lenet5_top_int8`) and integrates it into a Zynq block design, where the ARM processing system (PS) controls the accelerator in the programmable logic (PL).

## Features

- INT8 fixed-point inference of the full LeNet-5 pipeline
- HLS-generated IP core (`lenet5_top_int8`) with AXI interfaces
- Zynq PS-PL integration, with AXI interconnect to the PS high-performance port (`S_AXI_HP0`)
- Python/PYNQ-based host flow for loading the bitstream and running inference
- Custom Verilog AXI arbiter as a redesign of the 7:1 `axi_mem_intercon` block *(in progress)*

Lenet5_FPGA/
├── hls/            # HLS C++ source and testbench for lenet5_top_int8
├── vivado/         # Block design scripts, constraints
├── verilog/        # Custom RTL (AXI arbiter)
├── software/       # Training, quantization, preprocessing scripts
├── notebooks/      # PYNQ Jupyter notebooks for running on the board
