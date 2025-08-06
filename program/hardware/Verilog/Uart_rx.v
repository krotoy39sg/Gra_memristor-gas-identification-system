/***************串口接收模块***************/
/*模块调用
    Uart_rx U_Uart_rx(
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .rx(rx),
        .data(data),
        .done(done)
    );
*/
/*tb仿真
task Rx_rec;
    input [7:0] data;
    begin
        #104_140 rx = 0;
        #104_140 rx = data[0]; #104_140 rx = data[1];
        #104_140 rx = data[2]; #104_140 rx = data[3];
        #104_140 rx = data[4]; #104_140 rx = data[5];
        #104_140 rx = data[6]; #104_140 rx = data[7];
        #104_140 rx = 1;
    end
endtask
Rx_rec(8'b1010_1010);
*/
`include   "UART_param.v"

module Uart_rx(
    clk,
    rst_n,
    en,
    rx,
    data,
    done
    );

    //输入输出
    input                clk;
    input                rst_n;
    input                en;
    input                rx;
    output     [7:0]     data;
    output               done;
    reg        [7:0]     data;
    wire                 done;

    /***************接收端rx下降沿检测***************/
    reg rx_buf0;
    reg rx_buf1;
    reg rx_buf2;
    wire rx_fall;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            rx_buf0 <= 1'b0;
            rx_buf1 <= 1'b0;
            rx_buf2 <= 1'b0;
        end
        else begin
            if(en) 
                rx_buf0 <= rx;
            rx_buf1 <= rx_buf0;
            rx_buf2 <= rx_buf1;
        end
    end
    assign rx_fall = (~rx_buf1) & (rx_buf2);
    /***************空闲位***************/
    reg idle;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if((idle == 1'b1) && (rx_fall == 1'b1))
                idle <= 1'b0;
            else if((idle == 1'b0) && (done == 1'b1))
                idle <= 1'b1;
        end
    end
    /***************波特率分频div***************/
    reg [12:0] cnt_div;
    wire en_div;
    wire co_div;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_div <= 1'b0;
        end
        else if(done == 1'b1)
            cnt_div <= 1'b0;
        else if(en_div)begin
            if(co_div)
                cnt_div <= 1'b0;
            else
                cnt_div <= cnt_div + 1'b1;
        end
    end
    assign en_div = (idle == 1'b0);       
    assign co_div = (en_div) & (cnt_div == `UART_CNT_BPS - 1'b1);
    /***************数据位置idx***************/
    reg [3:0] cnt_idx;
    wire en_idx;
    wire co_idx;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_idx <= 1'b0;
        end
        else if(done == 1'b1)
            cnt_idx <= 1'b0;
        else if(en_idx)begin
            if(co_idx)
                cnt_idx <= 1'b0;
            else
                cnt_idx <= cnt_idx + 1'b1;
        end
    end
    assign en_idx = co_div;       
    assign co_idx = (en_idx) && (cnt_idx == 4'd9);
    /***************接收端rx***************/
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            data <= 8'b0;
        end
        else begin
            if(cnt_div == `UART_HALF_BPS - 1'b1)
                case(cnt_idx)
                    4'd1: data[0] <= rx_buf1;
                    4'd2: data[1] <= rx_buf1;
                    4'd3: data[2] <= rx_buf1;
                    4'd4: data[3] <= rx_buf1;
                    4'd5: data[4] <= rx_buf1;
                    4'd6: data[5] <= rx_buf1;
                    4'd7: data[6] <= rx_buf1;
                    4'd8: data[7] <= rx_buf1;
                endcase
        end
    end
    /***************结束标志***************/
    assign done = (cnt_idx == 4'd9) && (cnt_div == `UART_HALF_BPS - 1'b1);

endmodule

