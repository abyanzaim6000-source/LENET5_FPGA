open_project dense_output_proj
set_top dense_output
add_files src/dense_output.cpp
add_files src/dense_output.h
add_files -tb tb/dense_output_tb.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
