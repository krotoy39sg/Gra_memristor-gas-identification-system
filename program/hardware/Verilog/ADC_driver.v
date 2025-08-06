/* ------------------------------ ADC_TLC2543驱动 ----------------------------- */
//工作周期15us
/*tb仿真
always@(negedge ad_sck)begin
    ad_sdo = ~ad_sdo;
end
always@(posedge ad_csn)begin
    #1_000 ad_eoc = 0;
    #10_000 ad_eoc = 1;
    ad_sdo = ~ad_sdo;
end
*/
module ADC_driver(
    input                   clk             ,
    input                   rst_n           ,
    input                   start           ,
    input       [7:0]       din             ,
    input                   sdo             ,
    input                   eoc             ,
    output reg              cs_n            ,
    output reg              sck             ,
    output reg              sdi             ,
    output reg  [11:0]      dout            ,
    output reg              done
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 状态机
    reg         [1:0]       state_c         ;
    reg         [1:0]       state_n         ;
    wire                    idle_conf       ;
    wire                    conf_wait       ;
    wire                    wait_idle       ;
    // 数据缓冲
    reg         [7:0]       din_buf         ;
    // 时间步计数
    reg         [7:0]       cnt_step        ;
    wire                    en_step         ;
    wire                    co_step         ;
    // EOC上升沿检测
    reg                     eoc_buf0        ;
    reg                     eoc_buf1        ;
    reg                     eoc_buf2        ;
    wire                    eoc_rise        ;

    /* ----------------------------------- 状态机 ---------------------------------- */
    parameter IDLE = 2'b00;
    parameter CONF = 2'b01;
    parameter WAIT = 2'b10;
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
                if(idle_conf)
                    state_n = CONF;
                else 
                    state_n = state_c;
            end
            CONF:begin
                if(conf_wait)
                    state_n = WAIT;
                else 
                    state_n = state_c;
            end
            WAIT:begin
                if(wait_idle)
                    state_n = IDLE;
                else 
                    state_n = state_c;
            end
            default:begin
                state_n = IDLE;
            end
        endcase
    end

    /* ---------------------------------- 数据缓冲 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            din_buf <= 8'b0;
        end
        else begin
            if(start & (state_c == IDLE))
                din_buf <= din;
        end
    end

    /* ----------------------------------- 时间步计数 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_step <= 1'b0;
        end
        else if(en_step)begin
            if(co_step)
                cnt_step <= 1'b0;
            else
                cnt_step <= cnt_step + 1'b1;
        end
    end
    assign en_step = (state_c == CONF);
    assign co_step = (en_step) & (cnt_step == 8'd250);

    /* ---------------------------------- SPI信号 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cs_n <= 1'b1;
            sck  <= 1'b0;
            sdi  <= 1'b0;
        end
        else begin
            if(state_c == CONF)begin
                case(cnt_step)
                    8'd0  : cs_n <= 1'b0;
                    8'd75 : sdi <= din_buf[7];
                    8'd82 : begin dout[11] <= sdo;      sck <= 1'b1; end
                    8'd89 : sck <= 1'b0;
                    8'd92 : sdi <= din_buf[6];
                    8'd96 : begin dout[10] <= sdo;      sck <= 1'b1; end
                    8'd103: sck <= 1'b0;
                    8'd106: sdi <= din_buf[5];
                    8'd110: begin dout[9] <= sdo;       sck <= 1'b1; end
                    8'd117: sck <= 1'b0;
                    8'd120: sdi <= din_buf[4];
                    8'd124: begin dout[8] <= sdo;       sck <= 1'b1; end
                    8'd131: sck <= 1'b0;
                    8'd134: sdi <= din_buf[3];
                    8'd138: begin dout[7] <= sdo;       sck <= 1'b1; end
                    8'd145: sck <= 1'b0;
                    8'd148: sdi <= din_buf[2];
                    8'd152: begin dout[6] <= sdo;       sck <= 1'b1; end
                    8'd159: sck <= 1'b0;
                    8'd162: sdi <= din_buf[1];
                    8'd166: begin dout[5] <= sdo;       sck <= 1'b1; end
                    8'd173: sck <= 1'b0;
                    8'd176: sdi <= din_buf[0];
                    8'd180: begin dout[4] <= sdo;       sck <= 1'b1; end
                    8'd187: sck <= 1'b0;
                    8'd194: begin dout[3] <= sdo;       sck <= 1'b1; end
                    8'd201: sck <= 1'b0;
                    8'd208: begin dout[2] <= sdo;       sck <= 1'b1; end
                    8'd215: sck <= 1'b0;
                    8'd222: begin dout[1] <= sdo;       sck <= 1'b1; end
                    8'd229: sck <= 1'b0;
                    8'd236: begin dout[0] <= sdo;       sck <= 1'b1; end
                    8'd243: sck <= 1'b0;
                    8'd250: cs_n <= 1'b1;
                endcase
            end
        end
    end

    /* -------------------------------- EOC上升沿检测 -------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            eoc_buf0 <= 1'b0;
            eoc_buf1 <= 1'b0;
            eoc_buf2 <= 1'b0;
        end
        else begin
            eoc_buf0 <= eoc;
            eoc_buf1 <= eoc_buf0;
            eoc_buf2 <= eoc_buf1;
        end
    end
    assign eoc_rise = (eoc_buf1) & (~ eoc_buf2);
    
    /* ---------------------------------- 转移条件 ---------------------------------- */
    assign idle_conf = (state_c == IDLE) & (start);
    assign conf_wait = (state_c == CONF) & (co_step);
    assign wait_idle = (state_c == WAIT) & (eoc_rise);

    /* ---------------------------------- 结束标志 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            done <= 1'b0;
        end
        else begin
            done <= wait_idle;
        end
    end

endmodule

