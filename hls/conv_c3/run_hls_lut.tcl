open_project conv_c3_partialsum_lut_proj
set_top conv_c3_partialsum_lut
add_files src/conv_c3_partialsum_lut.cpp
add_files src/conv_c3.h
add_files -tb tb/conv_c3_partialsum_lut_tb.cpp
add_files -tb src/conv_c3_partialsum.cpp
open_solution "solution1" -flow_target vivado
set_part {xc7z020-clg484-1}
create_clock -period 10 -name default
csim_design
csynth_design
exit
