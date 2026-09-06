# EXPLORED, REVERTED variant -- see lenet5_top_dataflow.cpp and
# Results/hls_results.md. Kept so the DATAFLOW exploration stays
# independently reproducible; not part of the primary build (run_hls.tcl).
open_project lenet5_top_dataflow_proj
set_top lenet5_top_dataflow
add_files ../conv_c1/src/conv_c1_systolic.cpp
add_files ../conv_c1/src/conv_c1.h
add_files ../pool/src/pool_s2.cpp
add_files ../pool/src/pool_s2.h
add_files ../conv_c3/src/conv_c3_partialsum.cpp
add_files ../conv_c3/src/conv_c3.h
add_files ../pool_s4/src/pool_s4.cpp
add_files ../pool_s4/src/pool_s4.h
add_files ../dense_c5/src/dense_c5_partialsum.cpp
add_files ../dense_c5/src/dense_c5.h
add_files ../dense_f6/src/dense_f6.cpp
add_files ../dense_f6/src/dense_f6.h
add_files ../dense_output/src/dense_output.cpp
add_files ../dense_output/src/dense_output.h
add_files src/lenet5_top_dataflow.cpp
add_files src/lenet5_top.h
add_files -tb tb/lenet5_top_dataflow_tb.cpp
add_files -tb tb/lenet5_test_data.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
