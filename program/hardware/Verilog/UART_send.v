/***************多Byte发送模块***************/
/*模块调用
    UART_send U_UART_send(
       .clk(clk),
       .rst_n(rst_n),
       .start(send_start),
       .data(data_out),
       .idle(idle),
       .tx(tx)
    ); 
*/
`include   "UART_param.v"

module UART_send(
    clk,
    rst_n,
    start,
    data,
    idle,
    tx
    );

    //输入输出
    input                               clk;
    input                               rst_n;
    input                               start;
    input       [`UARTS_DATA_W-1:0]     data;
    output reg                          idle;
    output                              tx;

    /***************调用串口发送***************/
    reg tx_start;
    wire tx_done;
    wire [7:0] tx_data;
    Uart_tx U_Uart_tx(
       .clk(clk),
       .rst_n(rst_n),
       .start(tx_start),
       .data(tx_data),
       .tx(tx),
       .done(tx_done)
    ); 
    /***************Byte计数***************/
    reg [`UARTS_BYTE_W-1:0] cnt_num;
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
    assign en_num = tx_done;       
    assign co_num = (en_num) && (cnt_num == `UARTS_BYTE_N - 1'b1);
    /***************输入数据缓冲***************/
    reg [`UARTS_DATA_W-1:0] data_reg;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            data_reg <= 1'b0;
        end
        else begin
            if(start & idle)
                data_reg <= data;
        end
    end
    /***************输出发送内容***************/
    assign tx_data = data_reg[{`UARTS_BYTE_N-cnt_num-1, 3'b000} +: 8];
    /***************启动tx***************/
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            tx_start <= 1'b0;
        end
        else begin
            if((cnt_num == 1'b0) && (start))
                tx_start <= 1'b1;
            else if((cnt_num == `UARTS_BYTE_N - 1'b1) && (tx_done))
                tx_start <= 1'b0;
        end
    end  
    /***************工作状态***************/
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if(start)
                idle <= 1'b0;
            else if(co_num)
                idle <= 1'b1;
        end
    end

endmodule

