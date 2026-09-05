open_project dense_f6_proj
set_top dense_f6
add_files src/dense_f6.cpp
add_files src/dense_f6.h
add_files -tb tb/dense_f6_tb.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
