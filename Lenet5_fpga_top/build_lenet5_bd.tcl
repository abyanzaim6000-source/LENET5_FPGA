# Builds the lenet5_top block design in one unattended batch pass:
# PS7 + lenet5_top HLS IP, connection automation for every interface,
# explicit assign_bd_address, then validate_bd_design. Deliberately done as
# ONE script execution (not incremental GUI passes) to avoid the
# address-overlap issue seen from partial/repeated manual automation.

# Delete any partial lenet5_system design left over from a previous
# (e.g. failed) run BEFORE the project is even opened -- once open_project
# has scanned it in as a registered source, cleaning it up mid-session is
# much fiddlier than just not letting it exist in the first place.
set bd_dir "C:/Users/DELL/LENET5_FPGA/Lenet5_fpga_top/Lenet5_fpga_top.srcs/sources_1/bd/lenet5_system"
if {[file exists $bd_dir]} {
    puts "Deleting pre-existing $bd_dir to rebuild lenet5_system cleanly"
    file delete -force $bd_dir
}

open_project Lenet5_fpga_top.xpr

# The project file itself can still reference lenet5_system.bd as a
# registered source even after the physical file above was deleted (stale
# fileset metadata from the earlier failed run) -- drop that reference too,
# or create_bd_design below fails with "A design with the name
# 'lenet5_system' already exists" despite the file being gone.
foreach f [get_files -quiet -filter {NAME =~ "*lenet5_system.bd"}] {
    puts "Removing stale project source reference: $f"
    catch {remove_files -quiet $f}
}

set result "UNKNOWN"
set errmsg ""

if {[catch {

    # Add to whatever ip_repo_paths the project already has (e.g. the
    # conv_c1_systolic repo used by system_1) rather than overwriting it --
    # a plain overwrite would break system_1's IP resolution if it's ever
    # reopened.
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

    # ---- Zynq7 Processing System ----
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
    apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
        -config {make_external "FIXED_IO, DDR"} \
        [get_bd_cells processing_system7_0]

    # PS7 doesn't expose S_AXI_HP0 by default (no board preset in this
    # project to enable it implicitly) -- turn it on explicitly so the axi4
    # automation below has an actual slave interface to target. Without
    # this, apply_bd_automation fails with "No valid slave interface could
    # be found" because the pin doesn't exist on the cell yet.
    set_property -dict [list CONFIG.PCW_USE_S_AXI_HP0 {1}] [get_bd_cells processing_system7_0]

    # ---- lenet5_top HLS IP ----
    create_bd_cell -type ip -vlnv $lenet5_vlnv lenet5_top_0

    # ---- Control interface: PS7 M_AXI_GP0 -> lenet5_top_0/s_axi_control ----
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config { Master {/processing_system7_0/M_AXI_GP0} Slave {/lenet5_top_0/s_axi_control} intc_ip {New AXI Interconnect} Clk_xbar {Auto} Clk_master {Auto} Clk_slave {Auto} } \
        [get_bd_intf_pins lenet5_top_0/s_axi_control]

    # ---- All m_axi master interfaces on the IP, fanned into ONE shared
    # interconnect into PS7 S_AXI_HP0. NOTE: apply_bd_automation's axi4 rule
    # cannot be used for these pins -- confirmed by direct testing that
    # S_AXI_HP0 and its ACLK exist and are perfectly valid targets, yet the
    # rule engine still fails with:
    #   ERROR: [Ip 78-87] Error found in procedure get_rule_options.
    #   ERROR: [Ip 78-92] Failed to extract configurable options
    #   ERROR: [xilinx.com:hls:lenet5_top:1.0-1] /lenet5_top_0 No valid
    #   slave interface could be found to connect to </lenet5_top_0/m_axi_...>
    # This is specific to this HLS IP's own exported automation metadata for
    # its multiple independently-named m_axi bundles (s_axi_control, the
    # IP's one slave interface, connects via automation above just fine) --
    # not a PS7/config problem. Worked around by building the fan-in
    # interconnect explicitly (manual connect_bd_intf_net/connect_bd_net)
    # instead of relying on automation for just these pins. ----
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

    # Clock: FCLK_CLK0 already exists (enabled implicitly by the control
    # interface's automation above) -- drive every clock pin on the new
    # interconnect plus S_AXI_HP0's ACLK from it. Name-based filtering
    # (not TYPE ==) since axi_interconnect's clock/reset pins didn't
    # reliably carry the expected TYPE property in this IP version.
    set clk_pins [get_bd_pins -of_objects [get_bd_cells axi_mem_intercon] -filter {NAME =~ "*ACLK*"}]
    lappend clk_pins [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]
    puts "Clock pins to tie to FCLK_CLK0: $clk_pins"
    connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] $clk_pins

    # Reset: a dedicated proc_sys_reset fed by FCLK_RESET0_N, matching the
    # rst_ps7_0_50M pattern already used for the control-path interconnect.
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
