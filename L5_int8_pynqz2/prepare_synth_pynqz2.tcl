# INT8 counterpart of ../Lenet5_fpga_top_pynqz2/prepare_synth_pynqz2.tcl --
# same stage-1 split (wrapper generation kept separate from the expensive
# synth+impl stage) for the same reason: generate_target on a board_part
# -enabled project has been flaky (Windows Tcl interpreter/TLS-slot
# exhaustion inside the single batch process, "Could not create slave
# interpreter '::ipgen_iptclns'"), so it's isolated here as its own
# short-lived, cheaply-retriable invocation.
#
# Run via `vivado -mode batch -source prepare_synth_pynqz2.tcl`, retrying
# the whole process (fresh Tcl interpreter budget each time) until it
# exits 0.

open_project L5_int8_pynqz2.xpr

set result "UNKNOWN"
set errmsg ""

if {[catch {

    set bd_files [get_files -quiet -filter {NAME =~ "*/lenet5_int8.bd"}]
    if {[llength $bd_files] == 0} {
        error "lenet5_int8.bd not found in project sources"
    }
    set bd_file [lindex $bd_files 0]
    set bd_dir [file dirname $bd_file]
    puts "Using block design: $bd_file"

    open_bd_design $bd_file

    make_wrapper -files [get_files $bd_file] -top -force

    set gen_bd_dir [string map {.srcs .gen} $bd_dir]
    set wrapper_candidates [glob -nocomplain -directory [file join $bd_dir hdl] "lenet5_int8_wrapper.*"]
    if {[llength $wrapper_candidates] == 0} {
        set wrapper_candidates [glob -nocomplain -directory [file join $gen_bd_dir hdl] "lenet5_int8_wrapper.*"]
    }
    if {[llength $wrapper_candidates] == 0} {
        error "make_wrapper did not produce lenet5_int8_wrapper.* under $bd_dir/hdl or $gen_bd_dir/hdl"
    }
    set wrapper_file [lindex $wrapper_candidates 0]
    puts "Generated wrapper: $wrapper_file"

    add_files -norecurse $wrapper_file
    update_compile_order -fileset sources_1

    set wrapper_name [file rootname [file tail $wrapper_file]]
    set_property top $wrapper_name [current_fileset]
    update_compile_order -fileset sources_1
    puts "Top set to: $wrapper_name"

    generate_target all [get_files $bd_file]

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
