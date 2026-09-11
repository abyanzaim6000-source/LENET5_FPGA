open_project lenet5_top_int8_proj
set_top lenet5_top_int8
add_files ../conv_c1/src/conv_c1_int8_fixedpoint.cpp
add_files ../conv_c1/src/conv_c1_int8_fixedpoint.h
add_files ../pool/src/pool_s2_int8.cpp
add_files ../pool/src/pool_s2_int8.h
add_files ../conv_c3/src/conv_c3_int8_fixedpoint.cpp
add_files ../conv_c3/src/conv_c3_int8_fixedpoint.h
add_files ../pool_s4/src/pool_s4_int8.cpp
add_files ../pool_s4/src/pool_s4_int8.h
add_files ../dense_c5/src/dense_c5_int8_fixedpoint.cpp
add_files ../dense_c5/src/dense_c5_int8_fixedpoint.h
add_files ../dense_f6/src/dense_f6_int8_fixedpoint.cpp
add_files ../dense_f6/src/dense_f6_int8_fixedpoint.h
add_files ../dense_output/src/dense_output_int8.cpp
add_files ../dense_output/src/dense_output_int8.h
add_files src/lenet5_top_int8.cpp
add_files src/lenet5_top_int8.h
add_files -tb tb/lenet5_top_int8_tb.cpp
add_files -tb tb/lenet5_top_int8_test_data.h
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
