# Creates a fresh Vivado project targeting the PYNQ-Z2 board (tul.com.tw
# board file, part xc7z020clg400-1 -- the actual PYNQ-Z2 chip/package,
# distinct from both the xc7z020clg484-1 the HLS export was synthesized
# against and the xc7z020iclg484-1L the original Lenet5_fpga_top project
# targets). Kept as a sibling project directory rather than mutating the
# existing project in place, so the validated xc7z020iclg484-1L build stays
# intact and reproducible.
#
# Run from the Lenet5_fpga_top_pynqz2 directory:
#   vivado -mode batch -source create_project_pynqz2.tcl -nolog -nojournal

set proj_name "Lenet5_fpga_top_pynqz2"
set proj_dir  "C:/Users/DELL/LENET5_FPGA/Lenet5_fpga_top_pynqz2"
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
