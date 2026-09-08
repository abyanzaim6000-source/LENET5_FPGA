open_project conv_c1_int8_proj
set_top conv_c1_int8
add_files src/conv_c1_int8.cpp
add_files src/conv_c1_int8.h
add_files -tb tb/conv_c1_int8_tb.cpp
add_files -tb tb/conv_c1_int8_test_data.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
