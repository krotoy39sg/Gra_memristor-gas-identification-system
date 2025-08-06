transcript on
if ![file isdirectory verilog_libs] {
	file mkdir verilog_libs
}

vlib verilog_libs/altera_ver
vmap altera_ver ./verilog_libs/altera_ver
vlog -vlog01compat -work altera_ver {g:/softdata/quartus ii 13.0/quartus/eda/sim_lib/altera_primitives.v}

vlib verilog_libs/lpm_ver
vmap lpm_ver ./verilog_libs/lpm_ver
vlog -vlog01compat -work lpm_ver {g:/softdata/quartus ii 13.0/quartus/eda/sim_lib/220model.v}

vlib verilog_libs/sgate_ver
vmap sgate_ver ./verilog_libs/sgate_ver
vlog -vlog01compat -work sgate_ver {g:/softdata/quartus ii 13.0/quartus/eda/sim_lib/sgate.v}

vlib verilog_libs/altera_mf_ver
vmap altera_mf_ver ./verilog_libs/altera_mf_ver
vlog -vlog01compat -work altera_mf_ver {g:/softdata/quartus ii 13.0/quartus/eda/sim_lib/altera_mf.v}

vlib verilog_libs/altera_lnsim_ver
vmap altera_lnsim_ver ./verilog_libs/altera_lnsim_ver
vlog -sv -work altera_lnsim_ver {g:/softdata/quartus ii 13.0/quartus/eda/sim_lib/altera_lnsim.sv}

vlib verilog_libs/cycloneive_ver
vmap cycloneive_ver ./verilog_libs/cycloneive_ver
vlog -vlog01compat -work cycloneive_ver {g:/softdata/quartus ii 13.0/quartus/eda/sim_lib/cycloneive_atoms.v}

if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/Sig_dly.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/DAC_pulse.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/ADC_loop.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/wr_ctrl.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/UART_param.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/DAC_driver.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/ADC_set.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/ADC_driver.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl/ip_core {G:/verilog/work/20240925_wr_ctrl/ip_core/Fifo.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/Uart_tx.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/UART_send.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/Uart_rx.v}
vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/UART_recv.v}

vlog -vlog01compat -work work +incdir+G:/verilog/work/20240925_wr_ctrl {G:/verilog/work/20240925_wr_ctrl/wr_ctrl.vt}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cycloneive_ver -L rtl_work -L work -voptargs="+acc"  wr_ctrl_tb

add wave *
view structure
view signals
run 1 ns
