/* ---------------------------------- ADC循环采集 --------------------------------- */
module ADC_loop(
    input                   clk             ,
    input                   rst_n           ,
    input                   start           ,
    input      [4:0]        num_b           ,
    input      [10:0]       chan            ,
    input      [15:0]       dly             ,
    input                   sdo             ,
    input                   eoc             ,
    output                  cs_n            ,
    output                  sck             ,
    output                  sdi             ,
    output     [15:0]       data            ,
    output                  wren
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 空闲状态
    reg                     idle            ;
    reg                     ad_idle         ;
    // ADC配置
    reg                     ad_start        ;
    reg         [3:0]       ad_chan         ;
    reg         [15:0]      ad_dly          ;
    // ADC通道
    reg         [3:0]       cnt_chan        ;
    wire                    en_chan         ;
    wire                    co_chan         ;

    /* ---------------------------------- ADC配置 --------------------------------- */
    ADC_set U_ADC_set(
        .clk(clk),
        .rst_n(rst_n),
        .start(ad_start),
        .num_b(num_b),
        .chan(cnt_chan),
        .dly(ad_dly),
        .sdo(sdo),
        .eoc(eoc),
        .cs_n(cs_n),
        .sck(sck),
        .sdi(sdi),
        .data(data[11:0]),
        .done(wren)
    );
    assign data[15:12] = cnt_chan;

    /* ----------------------------------- 空闲状态 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if(start)
                idle <= 1'b0;
            else if(co_chan)
                idle <= 1'b1;
        end
    end

    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            ad_idle <= 1'b1;
        end
        else begin
            if(ad_start)
                ad_idle <= 1'b0;
            else if(wren)
                ad_idle <= 1'b1;
        end
    end

    /* ------------------------------ ADC启动 ------------------------------ */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            ad_start <= 1'b0;
        end
        else begin
            if(~idle & ad_idle & chan[cnt_chan])
                ad_start <= 1'b1;
            else
                ad_start <= 1'b0;
        end
    end

    /* ------------------------------ ADC通道 ------------------------------ */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cnt_chan <= 1'b0;
        end
        else if(en_chan)begin
            if(co_chan)
                cnt_chan <= 1'b0;
            else
                cnt_chan <= cnt_chan + 1'b1;
        end
    end
    assign en_chan = (~ idle) & ((wren & chan[cnt_chan]) | (~chan[cnt_chan]));
    assign co_chan = (en_chan) & (cnt_chan == 4'd11 - 1'b1);

    /* ------------------------------ ADC延时 ------------------------------ */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            ad_dly <= 1'b0;
        end
        else begin
            if(start | co_chan)
                ad_dly <= dly;
            else if(wren)
                ad_dly <= 16'd10;
        end
    end

endmodule
