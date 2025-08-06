/* ------------------------------- DAC_MCP4822驱动 ------------------------------ */
//工作周期 1.54us
module DAC_driver(
    input                   clk             ,
    input                   rst_n           ,
    input                   start           ,
    input       [15:0]      data            ,
    output  reg             cs_n             ,
    output  reg             sck             ,
    output  reg             sdi             ,
    output  reg             ld_n            ,   
    output                  done   
    );

    /* ---------------------------------- 中间信号 ---------------------------------- */
    // 空闲状态
    reg                     idle            ;
    // 数据缓冲
    reg         [15:0]      data_buf        ;
    // 时间步计数
    reg         [6:0]       cnt_step        ;
    wire                    en_step         ;
    wire                    co_step         ;

    /* ---------------------------------- 空闲状态 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            idle <= 1'b1;
        end
        else begin
            if(start)
                idle <= 1'b0;
            else if(done)
                idle <= 1'b1;
        end
    end

    /* ---------------------------------- 数据缓冲 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            data_buf <= 16'b0;
        end
        else begin
            if(start & idle)
                data_buf <= data;
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
    assign en_step = ~ idle;
    assign co_step = (en_step) & (cnt_step == 7'd76);

    /* ---------------------------------- SPI信号 ---------------------------------- */
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            cs_n <= 1'b1;
            sck <= 1'b0;
            sdi <= 1'b0;
            ld_n <= 1'b1;
        end
        else begin
            if(~ idle)begin
                case(cnt_step)
                    7'd0 : cs_n <= 1'b0;
                    7'd2 : sdi <= data_buf[15];
                    7'd3 : sck <= 1'b1;
                    7'd5 : sck <= 1'b0;
                    7'd6 : sdi <= data_buf[14];
                    7'd7 : sck <= 1'b1;
                    7'd9 : sck <= 1'b0;
                    7'd10: sdi <= data_buf[13];
                    7'd11: sck <= 1'b1;
                    7'd13: sck <= 1'b0;
                    7'd14: sdi <= data_buf[12];
                    7'd15: sck <= 1'b1;
                    7'd17: sck <= 1'b0;
                    7'd18: sdi <= data_buf[11];
                    7'd19: sck <= 1'b1;
                    7'd21: sck <= 1'b0;
                    7'd22: sdi <= data_buf[10];
                    7'd23: sck <= 1'b1;
                    7'd25: sck <= 1'b0;
                    7'd26: sdi <= data_buf[9];
                    7'd27: sck <= 1'b1;
                    7'd29: sck <= 1'b0;
                    7'd30: sdi <= data_buf[8];
                    7'd31: sck <= 1'b1;
                    7'd33: sck <= 1'b0;
                    7'd34: sdi <= data_buf[7];
                    7'd35: sck <= 1'b1;
                    7'd37: sck <= 1'b0;
                    7'd38: sdi <= data_buf[6];
                    7'd39: sck <= 1'b1;
                    7'd41: sck <= 1'b0;
                    7'd42: sdi <= data_buf[5];
                    7'd43: sck <= 1'b1;
                    7'd45: sck <= 1'b0;
                    7'd46: sdi <= data_buf[4];
                    7'd47: sck <= 1'b1;
                    7'd49: sck <= 1'b0;
                    7'd50: sdi <= data_buf[3];
                    7'd51: sck <= 1'b1;
                    7'd53: sck <= 1'b0;
                    7'd54: sdi <= data_buf[2];
                    7'd55: sck <= 1'b1;
                    7'd57: sck <= 1'b0;
                    7'd58: sdi <= data_buf[1];
                    7'd59: sck <= 1'b1;
                    7'd61: sck <= 1'b0;
                    7'd62: sdi <= data_buf[0];
                    7'd63: sck <= 1'b1;
                    7'd65: sck <= 1'b0;
                    7'd67: cs_n <= 1'b1;
                    7'd70: ld_n <= 1'b0;
                    7'd76: ld_n <= 1'b1;
                endcase
            end
        end
    end

    /* ---------------------------------- 结束信号 ---------------------------------- */
    assign done = co_step;

endmodule

