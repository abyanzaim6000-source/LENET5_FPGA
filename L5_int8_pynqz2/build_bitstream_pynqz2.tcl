# INT8 counterpart of ../Lenet5_fpga_top_pynqz2/build_bitstream_pynqz2.tcl.
# Stage 2 of the PYNQ-Z2 bitstream build -- run only after
# prepare_synth_pynqz2.tcl has completed successfully. Single-job
# synthesis and implementation (-jobs 1), same as the float32 build, per
# the RAM constraints hit on this machine previously.
#
# Run via `vivado -mode batch -source build_bitstream_pynqz2.tcl` from
# this directory.

open_project L5_int8_pynqz2.xpr

set result "UNKNOWN"
set errmsg ""

if {[catch {

    set wrapper_name [get_property top [current_fileset]]
    puts "Project top: $wrapper_name"
    if {$wrapper_name ne "lenet5_int8_wrapper"} {
        error "Project top is '$wrapper_name', expected lenet5_int8_wrapper -- run prepare_synth_pynqz2.tcl first"
    }

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
