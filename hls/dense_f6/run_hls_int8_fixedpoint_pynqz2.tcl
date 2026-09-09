open_project dense_f6_int8_fixedpoint_proj
set_top dense_f6_int8_fixedpoint
add_files src/dense_f6_int8_fixedpoint.cpp
add_files src/dense_f6_int8_fixedpoint.h
add_files -tb tb/dense_f6_int8_fixedpoint_tb.cpp
add_files -tb tb/dense_f6_int8_fixedpoint_test_data.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
