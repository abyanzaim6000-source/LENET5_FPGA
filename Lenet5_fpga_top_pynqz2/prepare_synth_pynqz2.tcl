# Stage 1 of the PYNQ-Z2 bitstream build, split out of
# build_bitstream_pynqz2.tcl so the crash-prone generate_target call can be
# retried cheaply (~1 min) without re-running the expensive synth+impl
# stage (stage 2, build_bitstream_pynqz2.tcl) to find out it needs a retry.
#
# generate_target on this board_part-enabled project has hit
# "ERROR: [Common 17-232] Could not create slave interpreter
# '::ipgen_iptclns'" (Windows Tcl interpreter/TLS-slot exhaustion inside
# the single batch process) 2 times out of 2 attempts, dying partway
# through the ~15 IPs in this BD (a different IP each time). Skipping
# generate_target entirely avoided the crash but left the wrapper file not
# recognized as a synthesis source by launch_runs's auto-derived non-project
# run ("ERROR: [Synth 8-439] module 'lenet5_system_wrapper' not found"),
# so it's still needed -- just retried here as its own short-lived
# invocation until it succeeds.
#
# Run via `vivado -mode batch -source prepare_synth_pynqz2.tcl`, retrying
# the whole process (fresh Tcl interpreter budget each time) until it
# exits 0.

open_project Lenet5_fpga_top_pynqz2.xpr

set result "UNKNOWN"
set errmsg ""

if {[catch {

    set bd_files [get_files -quiet -filter {NAME =~ "*/lenet5_system.bd"}]
    if {[llength $bd_files] == 0} {
        error "lenet5_system.bd not found in project sources"
    }
    set bd_file [lindex $bd_files 0]
    set bd_dir [file dirname $bd_file]
    puts "Using block design: $bd_file"

    open_bd_design $bd_file

    make_wrapper -files [get_files $bd_file] -top -force

    set gen_bd_dir [string map {.srcs .gen} $bd_dir]
    set wrapper_candidates [glob -nocomplain -directory [file join $bd_dir hdl] "lenet5_system_wrapper.*"]
    if {[llength $wrapper_candidates] == 0} {
        set wrapper_candidates [glob -nocomplain -directory [file join $gen_bd_dir hdl] "lenet5_system_wrapper.*"]
    }
    if {[llength $wrapper_candidates] == 0} {
        error "make_wrapper did not produce lenet5_system_wrapper.* under $bd_dir/hdl or $gen_bd_dir/hdl"
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
