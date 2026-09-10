open_project dense_output_int8_proj
set_top dense_output_int8
add_files src/dense_output_int8.cpp
add_files src/dense_output_int8.h
add_files -tb tb/dense_output_int8_tb.cpp
add_files -tb tb/dense_output_int8_test_data.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
