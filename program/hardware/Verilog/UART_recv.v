/* -------------------------------- 多Byte接收模块 ------------------------------- */
/*模块调用
    UART_recv U_UART_recv(
        .clk(clk),
        .rst_n(rst_n),
        .en(rec_en),
        .rx(rx),
        .data(rec_data),
        .wren(rec_wren)
    );
*/
`include   "UART_param.v"

module UART_recv(
    input                               clk             ,
    input                               rst_n           ,
    input                               en              ,
    input                               rx              ,
    output  reg [`UARTR_DATA_W-1:0]     data            ,
    output  reg                         wren
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 串口接收
    wire        [7:0]                   rx_data         ;
    wire                                rx_done         ;

    /* --------------------------------- 串口接收 --------------------------------- */
    Uart_rx U_Uart_rx(
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .rx(rx),
        .data(rx_data),
        .done(rx_done)
    );

    /* --------------------------------- Byte计数 --------------------------------- */
    reg [`UARTR_BYTE_W-1:0] cnt_num;
    wire en_num;
    wire co_num;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_num <= 1'b0;
        end
        else if(en_num)begin
            if(co_num)
                cnt_num <= 1'b0;
            else
                cnt_num <= cnt_num + 1'b1;
        end
    end
    assign en_num = rx_done;
    assign co_num = en_num && (cnt_num == `UARTR_BYTE_N - 1'b1);

    /* ---------------------------------- 保存数据 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            data <= 1'b0;
        end
        else begin
            if(en_num)
                data[{`UARTR_BYTE_N-cnt_num-1, 3'b000} +: 8] <= rx_data;
        end
    end
    /* ---------------------------------- 写入使能 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            wren <= 1'b0;
        end
        else begin
            wren <= co_num;
        end
    end

endmodule

