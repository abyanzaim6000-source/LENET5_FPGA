# PYNQ-Z2 project for the INT8 combined lenet5_top_int8 IP. Sibling of
# Lenet5_fpga_top_pynqz2 (the float32 build) -- a separate project
# directory, not a mutation of that one, same "old build stays untouched"
# convention this whole project follows. Same board file (tul.com.tw
# pynq-z2 part0) and same actual PYNQ-Z2 chip/package (xc7z020clg400-1,
# matching every INT8 HLS build's own set_part) used identically to the
# float32 PYNQ-Z2 project.
#
# Project/BD names kept SHORT (L5_int8_pynqz2 / lenet5_int8, not the
# longer Lenet5_fpga_top_int8_pynqz2 / lenet5_int8_system originally
# tried) -- the longer names pushed one HLS-exported RAM .dat filename's
# full generated path past Windows' 260-char MAX_PATH during
# generate_target (272 chars measured, vs. 237 for the float32 build's
# equivalent path: "lenet5_top_int8" is 5 chars longer than "lenet5_top"
# and appears TWICE per generated filename, plus the longer project/BD
# names each appear twice in the nested .gen/ip path). This is a genuine
# OS path-length ceiling, not a script bug -- confirmed by measuring both
# paths directly before renaming.
#
# Run from the L5_int8_pynqz2 directory:
#   vivado -mode batch -source create_project_pynqz2.tcl -nolog -nojournal

set proj_name "L5_int8_pynqz2"
set proj_dir  "C:/Users/DELL/LENET5_FPGA/L5_int8_pynqz2"
set board_vlnv "tul.com.tw:pynq-z2:part0:1.0"
set part_name  "xc7z020clg400-1"

set result "UNKNOWN"
set errmsg ""

if {[catch {

    set available [get_board_parts -quiet -filter "NAME == $board_vlnv"]
    if {[llength $available] == 0} {
        error "Board part $board_vlnv not found -- install the PYNQ-Z2 board_files first"
    }

    create_project $proj_name $proj_dir -part $part_name -force
    set_property board_part $board_vlnv [current_project]

    puts "Project part: [get_property PART [current_project]]"
    puts "Project board_part: [get_property board_part [current_project]]"

    close_project
    set result "SUCCESS"

} caterr]} {
    set result "FAILURE"
    set errmsg $caterr
}

puts "===BUILD_RESULT: $result==="
if {$result eq "FAILURE"} {
    puts "===BUILD_ERROR: $errmsg==="
}

exit [expr {$result eq "SUCCESS" ? 0 : 1}]
