# PYNQ-Z2 counterpart of Lenet5_fpga_top/build_lenet5_bd.tcl -- same
# lenet5_system block design (PS7 + lenet5_top HLS IP, AXI interconnect
# fan-in to S_AXI_HP0), built against the Lenet5_fpga_top_pynqz2 project
# (part xc7z020clg400-1, board_part tul.com.tw:pynq-z2:part0:1.0) instead of
# the original xc7z020iclg484-1L project. The lenet5_top HLS IP itself is
# reused unmodified from the existing hls/lenet5_top export: its
# component.xml declares supportedFamilies={zynq} (family-level, not
# part/package-specific), so no HLS re-export is needed for the package
# change.
#
# One addition vs. the original script: PS7 is brought up with
# apply_board_preset enabled so Vivado configures the PS7 (DDR3 timing for
# the board's MT41J256M16 chip, clock inputs, etc.) from the board file's
# preset.xml instead of Vivado's generic Zynq-7020 defaults.
#
# Run via `vivado -mode batch -source build_lenet5_bd_pynqz2.tcl` from this
# directory.

set bd_dir "C:/Users/DELL/LENET5_FPGA/Lenet5_fpga_top_pynqz2/Lenet5_fpga_top_pynqz2.srcs/sources_1/bd/lenet5_system"
if {[file exists $bd_dir]} {
    puts "Deleting pre-existing $bd_dir to rebuild lenet5_system cleanly"
    file delete -force $bd_dir
}

open_project Lenet5_fpga_top_pynqz2.xpr

foreach f [get_files -quiet -filter {NAME =~ "*lenet5_system.bd"}] {
    puts "Removing stale project source reference: $f"
    catch {remove_files -quiet $f}
}

set result "UNKNOWN"
set errmsg ""

if {[catch {

    set ip_repo "C:/Users/DELL/LENET5_FPGA/hls/lenet5_top/lenet5_top_proj/solution1/impl/ip"
    set existing_repos [get_property ip_repo_paths [current_project]]
    if {[lsearch -exact $existing_repos $ip_repo] < 0} {
        lappend existing_repos $ip_repo
    }
    set_property ip_repo_paths $existing_repos [current_project]
    update_ip_catalog -rebuild

    set ip_defs [get_ipdefs -filter {NAME == lenet5_top}]
    if {[llength $ip_defs] == 0} {
        error "lenet5_top IP not found in catalog after adding repo $ip_repo"
    }
    set lenet5_vlnv [lindex $ip_defs 0]
    puts "Using lenet5_top VLNV: $lenet5_vlnv"

    create_bd_design "lenet5_system"
    current_bd_design [get_bd_designs lenet5_system]

    # ---- Zynq7 Processing System, brought up from the PYNQ-Z2 board preset ----
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
        [get_bd_cells processing_system7_0]

    # Board preset configures DDR/clocks/FIXED_IO but not the HP ports --
    # enable S_AXI_HP0 explicitly same as the original build script, so the
    # axi4 automation below has an actual slave interface to target.
    set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

    # ---- lenet5_top HLS IP ----
    create_bd_cell -type ip -vlnv $lenet5_vlnv lenet5_top_0

    # ---- Control interface: PS7 M_AXI_GP0 -> lenet5_top_0/s_axi_control ----
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config { Master {/processing_system7_0/M_AXI_GP0} Slave {/lenet5_top_0/s_axi_control} intc_ip {New AXI Interconnect} Clk_xbar {Auto} Clk_master {Auto} Clk_slave {Auto} } \
        [get_bd_intf_pins lenet5_top_0/s_axi_control]

    # ---- All m_axi master interfaces on the IP, fanned into ONE shared
    # interconnect into PS7 S_AXI_HP0 (manual wiring -- automation's axi4
    # rule doesn't work for this IP's multiple m_axi bundles; see the
    # original build_lenet5_bd.tcl for the full explanation). ----
    set masters [get_bd_intf_pins -of_objects [get_bd_cells lenet5_top_0] -filter {MODE == Master}]
    puts "Found [llength $masters] master AXI interfaces on lenet5_top_0:"
    foreach m $masters { puts "  $m" }
    if {[llength $masters] == 0} {
        error "No master AXI interfaces found on lenet5_top_0 -- IP packaging mismatch?"
    }
    set n [llength $masters]

    create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_intercon
    set_property -dict [list CONFIG.NUM_SI $n CONFIG.NUM_MI {1}] [get_bd_cells axi_mem_intercon]

    for {set i 0} {$i < $n} {incr i} {
        set m [lindex $masters $i]
        set sidx [format "S%02d_AXI" $i]
        puts "Connecting $m -> axi_mem_intercon/$sidx"
        connect_bd_intf_net [get_bd_intf_pins $m] [get_bd_intf_pins axi_mem_intercon/$sidx]
    }
    connect_bd_intf_net [get_bd_intf_pins axi_mem_intercon/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

    # ---- Clocks ----
    set clk_pins [get_bd_pins -of_objects [get_bd_cells axi_mem_intercon] -filter {NAME =~ "*ACLK*"}]
    lappend clk_pins [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
    puts "Clock pins to tie to FCLK_CLK0: $clk_pins"
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] $clk_pins

    # ---- Reset ----
    create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_mem_intercon
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins rst_mem_intercon/slowest_sync_clk]
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_mem_intercon/ext_reset_in]
    set rst_pins [get_bd_pins -of_objects [get_bd_cells axi_mem_intercon] -filter {NAME =~ "*ARESETN*"}]
    puts "Reset pins to tie to rst_mem_intercon/peripheral_aresetn: $rst_pins"
    connect_bd_net [get_bd_pins rst_mem_intercon/peripheral_aresetn] $rst_pins

    # ---- Explicit, single, full address assignment pass ----
    assign_bd_address

    puts "===ADDRESS MAP AFTER assign_bd_address==="
    foreach seg [get_bd_addr_segs] {
        puts "SEG: $seg -> [get_property OFFSET $seg] / [get_property RANGE $seg]"
    }

    # ---- Validate ----
    validate_bd_design

    save_bd_design
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
