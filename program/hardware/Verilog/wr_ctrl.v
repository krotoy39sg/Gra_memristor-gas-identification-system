/* ---------------------------------- 读写控制 ---------------------------------- */
module wr_ctrl(
    input                   clk             ,
    input                   rst_n           ,
    input                   rx              ,
    output                  tx              ,
    output      [1:0]       da_csn          ,
    output                  da_sck          ,
    output                  da_sdi          ,
    output      [1:0]       da_ldn          ,
    input                   ad_sdo          ,
    input                   ad_eoc          ,
    output                  ad_csn          ,
    output                  ad_sck          ,
    output                  ad_sdi          ,
    output                  trig            ,
    output  reg [8:0]       sw              ,
    output  reg [3:0]       led
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 空闲状态
    reg                     idle            ;
    // 串口接收
    wire        [111:0]     rec_data        ;
    wire                    rec_done        ;
    // 接收指令
    wire        [8:0]       sw_line         ;
    wire        [4:0]       ad_nb           ;
    wire        [10:0]      ad_chan         ;
    wire        [15:0]      ad_dly          ;
    wire                    da1_chan        ;
    wire        [11:0]      da1_amp         ;
    wire        [31:0]      da1_wid         ;
    wire                    da2_chan        ;
    wire        [11:0]      da2_amp         ;
    // 配置栅压DAC
    wire                    da2_start       ;
    wire                    da2_sck         ;
    wire                    da2_sdi         ;
    wire                    da2_ldn         ;
    // 配置输入DAC
    wire                    da1_start       ;
    wire                    da1_sck         ;
    wire                    da1_sdi         ;
    wire                    da1_ldn         ;
    wire                    da1_done        ;
    // 配置ADC
    wire                    ad_start        ;
    wire        [15:0]      ad_data         ;
    wire                    ad_wren         ;
    // FIFO缓冲
    wire        [15:0]      fifo_data       ;
    wire                    fifo_rd         ;
    wire                    fifo_wr         ;
    wire                    fifo_empty      ;
    wire        [15:0]      fifo_q          ;
    // 发送结果
    wire                    send_idle       ;

    /* ----------------------------------- 空闲状态 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if(rec_done)
                idle <= 1'b0;
            else if(da1_done)
                idle <= 1'b1;
        end
    end

    /* ---------------------------------- 串口接收 ---------------------------------- */
    UART_recv U_UART_recv(
        .clk(clk),
        .rst_n(rst_n),
        .en(idle),
        .rx(rx),
        .data(rec_data),
        .wren(rec_done)
    );

    /* ---------------------------------- 接收指令 ---------------------------------- */
    assign da2_chan = rec_data[108];
    assign da2_amp = rec_data[107:96];
    assign sw_line = rec_data[88:80];
    assign ad_nb = rec_data[79:75];
    assign ad_chan = rec_data[74:64];
    assign ad_dly = rec_data[63:48];
    assign da1_chan = rec_data[44];
    assign da1_amp = rec_data[43:32];
    assign da1_wid = rec_data[31:0];

    /* ---------------------------------- 配置开关 --------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            sw <= 9'b1_0000_0000;
        end
        else begin
            if(idle)
                sw <= 9'b1_0000_0000;
            else 
                sw <= sw_line;
        end
    end

    /* ---------------------------------- 配置栅压DAC --------------------------------- */
    DAC_pulse U_DAC_bias(
        .clk(clk),
        .rst_n(rst_n),
        .start(da2_start),
        .chan(da2_chan),
        .amp(da2_amp),
        .wid(da1_wid+1'b1),
        .cs_n(da_csn[1]),
        .sck(da2_sck),
        .sdi(da2_sdi),
        .ld_n(da2_ldn),
        .done()
    );
    assign da2_start = rec_done;

    /* ---------------------------------- 延时输入 --------------------------------- */
    Sig_dly U_Sig_dly(
        .clk(clk),
        .rst_n(rst_n),
        .sig_in(da2_start),
        .dly(16'b1),
        .sig_out(da1_start)
    );

    /* ---------------------------------- 配置输入DAC --------------------------------- */
    DAC_pulse U_DAC_pulse(
        .clk(clk),
        .rst_n(rst_n),
        .start(da1_start),
        .chan(da1_chan),
        .amp(da1_amp),
        .wid(da1_wid),
        .cs_n(da_csn[0]),
        .sck(da1_sck),
        .sdi(da1_sdi),
        .ld_n(da1_ldn),
        .done(da1_done)
    );

    /* ---------------------------------- DAC数据选择 --------------------------------- */
    assign da_sck = da_csn[0] ? da2_sck : da1_sck; 
    assign da_sdi = da_csn[0] ? da2_sdi : da1_sdi; 
    assign da_ldn = {da2_ldn, da1_ldn}; 

    /* ---------------------------------- 配置ADC --------------------------------- */
    ADC_loop U_ADC_loop(
        .clk(clk),
        .rst_n(rst_n),
        .start(ad_start),
        .num_b(ad_nb),
        .chan(ad_chan),
        .dly(ad_dly),
        .sdo(ad_sdo),
        .eoc(ad_eoc),
        .cs_n(ad_csn),
        .sck(ad_sck),
        .sdi(ad_sdi),
        .data(ad_data),
        .wren(ad_wren)
    );
    assign ad_start = da1_start;

    /* ---------------------------------- FIFO缓冲 --------------------------------- */
    Fifo U_Fifo(
        .aclr(~rst_n),
        .clock(clk),
        .data(fifo_data),
        .rdreq(fifo_rd),
        .wrreq(fifo_wr),
        .empty(fifo_empty),
        .q(fifo_q)
    );
    
    assign fifo_wr = ad_wren;
    assign fifo_rd = (~fifo_empty) & send_idle;
    assign fifo_data = ad_data;

    /* ---------------------------------- 发送结果 --------------------------------- */
    UART_send U_UART_send(
       .clk(clk),
       .rst_n(rst_n),
       .start(fifo_rd),
       .data(fifo_q),
       .idle(send_idle),
       .tx(tx)
    ); 

    /* -------------------------------- LED显示当前状态 ------------------------------- */
    always@(*)begin
        if(idle)
            led = 4'b1111;
        else
            led = 4'b0000;
    end

    /* -------------------------------- 示波器触发 ------------------------------- */
    assign trig = ~ idle;

endmodule
