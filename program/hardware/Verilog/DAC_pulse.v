/* ---------------------------------- DAC脉冲配置 --------------------------------- */
//32位脉宽,单位us,0 ~ 4_294_967_296us
/***************设置通道,幅值,脉宽***************/
/*模块调用
    DAC_pulse U_DAC_pulse(
        .clk(clk),
        .rst_n(rst_n),
        .start(da_start),
        .chan(da_chan),
        .amp(da_amp),
        .wid(da_wid),
        .cs_n(da_csn),
        .sck(da_sck),
        .sdi(da_sdi),
        .ld_n(da_ldn),
        .done(da_done)
    );
*/
module DAC_pulse(
    input                   clk             ,
    input                   rst_n           ,
    input                   start           ,
    input                   chan            ,
    input       [11:0]      amp             ,
    input       [31:0]      wid             ,
    output                  cs_n            ,
    output                  sck             ,
    output                  sdi             ,
    output                  ld_n            ,   
    output                  done   
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 状态机
    reg         [1:0]       state_c         ;
    reg         [1:0]       state_n         ;
    wire                    idle_rise       ;
    wire                    rise_wait       ;
    wire                    wait_fall       ;
    wire                    fall_idle       ;
    // DAC驱动
    reg                     da_start        ;
    reg         [15:0]      da_data         ;
    wire                    da_done         ;
    // 1us分频
    reg         [5:0]       cnt_div         ;
    wire                    en_div          ;
    wire                    co_div          ;
    // 脉宽
    reg         [31:0]      cnt_wid         ;
    wire                    en_wid          ;
    wire                    co_wid          ;
    
    /* ----------------------------------- 状态机 ---------------------------------- */
    parameter CNT_1US = 6'd50;
    parameter IDLE = 2'd0;
    parameter RISE = 2'd1;
    parameter WAIT = 2'd2;
    parameter FALL = 2'd3;
    //状态更新
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)
            state_c <= IDLE;
        else
            state_c <= state_n;
    end
    //状态转移
    always@(*)begin
        case(state_c)
            IDLE:begin
                if(idle_rise)
                    state_n = RISE;
                else 
                    state_n = state_c;
            end
            RISE:begin
                if(rise_wait)
                    state_n = WAIT;
                else 
                    state_n = state_c;
            end
            WAIT:begin
                if(wait_fall)
                    state_n = FALL;
                else
                    state_n = state_c;
            end
            FALL:begin
                if(fall_idle)
                    state_n = IDLE;
                else
                    state_n = state_c;
            end
            default:begin
                state_n = IDLE;
            end
        endcase
    end

    /* ---------------------------------- DAC驱动 --------------------------------- */
    DAC_driver U_DAC_driver(
        .clk(clk),
        .rst_n(rst_n),
        .start(da_start),
        .data(da_data),
        .cs_n(cs_n),
        .sck(sck),
        .sdi(sdi),
        .ld_n(ld_n),
        .done(da_done)
    );
    
    /* ---------------------------------- DAC启动 --------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            da_start <= 1'b0;
        end
        else begin
            da_start <= idle_rise | wait_fall;
        end
    end

    /* ---------------------------------- 数据缓冲 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            da_data <= 16'b0001_000000000000;
        end
        else begin
            if(idle_rise)
                da_data <= {chan, 3'b001, amp};
            else if(wait_fall)
                da_data <= {chan, 15'b001_000000000000};
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
    assign en_div = (state_c == RISE) | (state_c == WAIT);
    assign co_div = en_div & (cnt_div == CNT_1US - 1'b1);

    /* ----------------------------------- 脉宽 ----------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_wid <= 1'b0;
        end
        else if(en_wid)begin
            if(co_wid)
                cnt_wid <= 1'b0;
            else
                cnt_wid <= cnt_wid + 1'b1;
        end
    end
    assign en_wid = co_div;  
    assign co_wid = en_wid & (cnt_wid == wid - 1'b1);

    /* --------------------------------- 状态转移条件 --------------------------------- */
    assign idle_rise = (state_c == IDLE) && (start);
    assign rise_wait = (state_c == RISE) && (da_done);
    assign wait_fall = (state_c == WAIT) && (co_wid);
    assign fall_idle = (state_c == FALL) && (da_done);

    /* ---------------------------------- 结束信号 ---------------------------------- */
    assign done = fall_idle;

endmodule

