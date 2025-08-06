/* ---------------------------------- ADC配置 --------------------------------- */
//16位脉宽,单位us,0 ~ 65_535us,采样延时
//5次采样, 75us
/***************设置通道,延时***************/
/*模块调用
    ADC_set U_ADC_set(
        .clk(clk),
        .rst_n(rst_n),
        .start(ad_start),
        .chan(ad_chan),
        .dly(ad_dly),
        .sdo(ad_sdo),
        .eoc(ad_eoc),
        .cs_n(ad_csn),
        .sck(ad_sck),
        .sdi(ad_sdi),
        .data(ad_data),
        .done(ad_done)
    );
*/
module ADC_set(
    input                   clk             ,
    input                   rst_n           ,
    input                   start           ,
    input      [4:0]        num_b           ,
    input      [3:0]        chan            ,
    input      [15:0]       dly             ,
    input                   sdo             ,
    input                   eoc             ,
    output                  cs_n            ,
    output                  sck             ,
    output                  sdi             ,
    output     [11:0]       data            ,
    output reg              done
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 状态机
    reg         [1:0]       state_c         ;
    reg         [1:0]       state_n         ;
    wire                    idle_dely       ;
    wire                    dely_read       ;
    wire                    read_wait       ;
    wire                    wait_read       ;
    wire                    wait_idle       ;
    // ADC驱动
    wire                    ad_start        ;
    wire        [7:0]       ad_din          ;
    wire        [11:0]      ad_dout         ;
    wire                    ad_done         ;
    // 1us分频
    reg         [5:0]       cnt_div         ;
    wire                    en_div          ;
    wire                    co_div          ;
    // 延时采样
    reg         [15:0]      cnt_dly         ;
    wire                    en_dly          ;
    wire                    co_dly          ;
    // 采样次数
    reg         [7:0]       cnt_num         ;
    wire                    en_num          ;
    wire                    co_num          ;
    // 采样累加
    reg         [18:0]      cnt_dout        ;

    /* --------------------------------- 状态机 --------------------------------- */
    parameter CNT_1US = 6'd50;
    parameter IDLE = 2'd0;
    parameter DELY = 2'd1;
    parameter READ = 2'd2;
    parameter WAIT = 2'd3;
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
                if(idle_dely)
                    state_n = DELY;
                else 
                    state_n = state_c;
            end
            DELY:begin
                if(dely_read)
                    state_n = READ;
                else 
                    state_n = state_c;
            end
            READ:begin
                if(read_wait)
                    state_n = WAIT;
                else 
                    state_n = state_c;
            end
            WAIT:begin
                if(wait_read)
                    state_n = READ;
                else if(wait_idle)
                    state_n = IDLE;
                else 
                    state_n = state_c;
            end
            default:begin
                state_n = IDLE;
            end
        endcase
    end

    /* ---------------------------------- ADC驱动 --------------------------------- */
    ADC_driver U_ADC_driver(
        .clk(clk),
        .rst_n(rst_n),
        .start(ad_start),
        .din(ad_din),
        .sdo(sdo),
        .eoc(eoc),
        .cs_n(cs_n),
        .sck(sck),
        .sdi(sdi),
        .dout(ad_dout),
        .done(ad_done)
    ); 
    assign ad_start = dely_read | wait_read;
    assign ad_din = {chan, 4'b0000};

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
    assign en_div = (state_c == DELY);
    assign co_div = en_div & (cnt_div == CNT_1US - 1'b1);

    /* ---------------------------------- 延时采样 ---------------------------------- */
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

    /* ------------------------------ 多次采样 ------------------------------ */
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
    assign en_num = ad_done;
    assign co_num = en_num & (cnt_num == 1 << num_b);

    /* --------------------------------- 去除第一次结果，取均值 -------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_dout <= 1'b0; 
        end
        else begin
            if(ad_done)begin
                if(cnt_num == 1'b1)
                    cnt_dout <= ad_dout;
                else
                    cnt_dout <= cnt_dout + ad_dout;
            end
        end
    end
    assign data = cnt_dout[11+num_b-:12];

    /* --------------------------------- 状态转移条件 --------------------------------- */
    assign idle_dely = (state_c == IDLE) & start;
    assign dely_read = (state_c == DELY) & co_dly;
    assign read_wait = (state_c == READ);
    assign wait_read = (state_c == WAIT) & ad_done & (~co_num);
    assign wait_idle = (state_c == WAIT) & co_num;

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
