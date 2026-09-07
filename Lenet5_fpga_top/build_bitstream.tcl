# Generates the HDL wrapper for lenet5_system, then runs synthesis and
# implementation (through write_bitstream) in one unattended batch pass.
# Run via `vivado -mode batch -source build_bitstream.tcl`, launched as a
# detached process the same way the lenet5_top HLS export was -- this is
# expected to take considerably longer than anything run so far.

open_project Lenet5_fpga_top.xpr

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

    # ---- Wrapper generation ----
    make_wrapper -files [get_files $bd_file] -top -force

    # Managed-IP projects write generated wrapper/output products under the
    # .gen tree (mirroring .srcs's bd/<name> structure), not next to the
    # .bd file itself under .srcs -- check both to be safe.
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

    # ---- Synthesis ----
    reset_run synth_1
    launch_runs synth_1 -jobs 1
    wait_on_run synth_1
    set synth_status [get_property STATUS [get_runs synth_1]]
    set synth_progress [get_property PROGRESS [get_runs synth_1]]
    puts "synth_1: STATUS=$synth_status PROGRESS=$synth_progress"
    if {$synth_progress ne "100%"} {
        error "synth_1 did not complete successfully: STATUS=$synth_status PROGRESS=$synth_progress"
    }

    # ---- Implementation through bitstream ----
    reset_run impl_1
    launch_runs impl_1 -to_step write_bitstream -jobs 1
    wait_on_run impl_1
    set impl_status [get_property STATUS [get_runs impl_1]]
    set impl_progress [get_property PROGRESS [get_runs impl_1]]
    puts "impl_1: STATUS=$impl_status PROGRESS=$impl_progress"
    if {$impl_progress ne "100%"} {
        error "impl_1 did not complete successfully: STATUS=$impl_status PROGRESS=$impl_progress"
    }

    # get_files only sees files registered as project sources, not the
    # bitstream Vivado writes into the run's output directory -- checking
    # for it there directly is the only reliable way to confirm it exists.
    set bit_file [file join [get_property DIRECTORY [get_runs impl_1]] "${wrapper_name}.bit"]
    puts "Bitstream file: $bit_file"
    if {![file exists $bit_file]} {
        error "impl_1 reported 100% but expected bitstream file not found: $bit_file"
    }

    # ---- Reports off the routed design ----
    open_run impl_1 -name impl_1_routed
    file mkdir "reports"
    report_timing_summary -file "reports/timing_summary.rpt"
    report_utilization -file "reports/utilization.rpt"
    report_drc -file "reports/drc.rpt"

    set wns [get_property STATS.WNS [get_runs impl_1]]
    set whs [get_property STATS.WHS [get_runs impl_1]]
    puts "===TIMING: WNS=$wns WHS=$whs==="

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
