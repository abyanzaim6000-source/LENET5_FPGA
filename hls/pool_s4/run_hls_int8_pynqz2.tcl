open_project pool_s4_int8_proj
set_top pool_s4_int8
add_files src/pool_s4_int8.cpp
add_files src/pool_s4_int8.h
add_files -tb tb/pool_s4_int8_tb.cpp
add_files -tb tb/pool_s4_int8_test_data.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
