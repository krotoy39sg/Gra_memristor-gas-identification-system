/***************串口发送模块***************/
/*模块调用
    Uart_tx U_Uart_tx(
       .clk(clk),
       .rst_n(rst_n),
       .start(start),
       .data(data),
       .tx(tx),
       .done(done)
    ); 
*/
`include   "UART_param.v"

module Uart_tx(
    clk,
    rst_n,
    start,
    data,
    tx,
    done
    );

    //输入输出
    input                clk;
    input                rst_n;
    input                start;
    input      [7:0]     data;
    output               tx;
    output               done;
    reg                  tx;
    wire                 done;

    /***************空闲位***************/
    reg idle;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if((idle == 1'b1) && (start == 1'b1))
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
        else if(en_div)begin
            if(co_div)
                cnt_div <= 1'b0;
            else
                cnt_div <= cnt_div + 1'b1;
        end
    end
    assign en_div = (idle == 1'b0);       
    assign co_div = (en_div) && (cnt_div == `UART_CNT_BPS - 1'b1);
    /***************数据位置idx***************/
    reg [3:0] cnt_idx;
    wire en_idx;
    wire co_idx;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_idx <= 1'b0;
        end
        else if(en_idx)begin
            if(co_idx)
                cnt_idx <= 1'b0;
            else
                cnt_idx <= cnt_idx + 1'b1;
        end
    end
    assign en_idx = co_div;       
    assign co_idx = (en_idx) && (cnt_idx == 4'd9);
    /***************数据锁存***************/
    reg [7:0] data_buf;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            data_buf <= 8'b0;
        end
        else begin
            if((idle == 1'b0) && (cnt_idx == 4'd0))
                data_buf <= data;
        end
    end
    /***************发送端tx***************/
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            tx <= 1'b1;
        end
        else if(idle == 1'b1)
            tx <= 1'b1;
        else begin
            case(cnt_idx)
                4'd0: tx <= 1'b0;
                4'd1: tx <= data_buf[0];
                4'd2: tx <= data_buf[1];
                4'd3: tx <= data_buf[2];
                4'd4: tx <= data_buf[3];
                4'd5: tx <= data_buf[4];
                4'd6: tx <= data_buf[5];
                4'd7: tx <= data_buf[6];
                4'd8: tx <= data_buf[7];
                4'd9: tx <= 1'b1;
                default: tx <= 1'b1;
            endcase
        end
    end
    /***************结束标志***************/
    assign done = co_idx;

endmodule

