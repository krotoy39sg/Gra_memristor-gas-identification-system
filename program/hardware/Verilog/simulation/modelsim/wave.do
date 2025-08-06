onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/idle
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/led
add wave -noupdate -divider RECV
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/rx
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/rec_done
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/rec_data
add wave -noupdate -radix binary /wr_ctrl_tb/u_wr_ctrl/ad_chan
add wave -noupdate -radix unsigned /wr_ctrl_tb/u_wr_ctrl/ad_dly
add wave -noupdate -divider DAC1
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_chan
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_amp
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_wid
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_start
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_sck
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_sdi
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_ldn
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da1_done
add wave -noupdate -divider DAC2
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da2_chan
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da2_amp
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da2_start
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da2_sck
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da2_sdi
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da2_ldn
add wave -noupdate -divider DAC
add wave -noupdate -radix binary /wr_ctrl_tb/u_wr_ctrl/da_csn
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da_sck
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da_sdi
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/da_ldn
add wave -noupdate -divider ADC
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/U_ADC_loop/U_ADC_set/start
add wave -noupdate -radix unsigned /wr_ctrl_tb/u_wr_ctrl/U_ADC_loop/U_ADC_set/chan
add wave -noupdate -radix unsigned /wr_ctrl_tb/u_wr_ctrl/U_ADC_loop/U_ADC_set/dly
add wave -noupdate -radix unsigned /wr_ctrl_tb/u_wr_ctrl/ad_nb
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/ad_csn
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/ad_sck
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/ad_sdo
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/ad_sdi
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/ad_eoc
add wave -noupdate -radix unsigned /wr_ctrl_tb/u_wr_ctrl/ad_data
add wave -noupdate -divider SEND
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/fifo_data
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/fifo_rd
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/fifo_wr
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/fifo_empty
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/fifo_q
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/send_idle
add wave -noupdate /wr_ctrl_tb/u_wr_ctrl/tx
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {4084521589 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits us
update
WaveRestoreZoom {0 ps} {21000001050 ps}
