/* ---------------------------------- 信号延时模块 --------------------------------- */
module Sig_dly(
    input                   clk             ,
    input                   rst_n           ,
    input                   sig_in          ,
    input      [15:0]       dly             ,
    output                  sig_out   
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 空闲状态
    reg                     idle           ;
    // 1us分频
    reg         [5:0]       cnt_div         ;
    wire                    en_div          ;
    wire                    co_div          ;
    // 延迟时间
    reg         [15:0]      cnt_dly         ;
    wire                    en_dly          ;
    wire                    co_dly          ;
    parameter   CNT_1US     =   6'd50       ;
    /* ----------------------------------- 空闲状态 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if(sig_in)
                idle <= 1'b0;
            else if(sig_out)
                idle <= 1'b1;
        end
    end

    /* ---------------------------------- 1us分频 --------------------------------- */
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
    assign en_div = ~ idle;
    assign co_div = en_div & (cnt_div == CNT_1US - 1'b1);

    /* ----------------------------------- 延迟时间 ----------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_dly <= 1'b0;
        end
        else if(en_dly)begin
            if(co_dly)
                cnt_dly <= 1'b0;
            else
                cnt_dly <= cnt_dly + 1'b1;
        end
    end
    assign en_dly = co_div;  
    assign co_dly = en_dly & (cnt_dly == dly - 1'b1);

    /* ---------------------------------- 输出信号 ---------------------------------- */
    assign sig_out = co_dly;

endmodule

