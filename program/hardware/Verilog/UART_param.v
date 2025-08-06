/* ----------------------------- UART 参数 ----------------------------- */
// 接收端
`define     UARTR_BYTE_N        14      // BYTE数目
`define     UARTR_BYTE_W        4       // BYTE数目位宽
`define     UARTR_DATA_W        112     // 数据位宽

// 发送端
`define     UARTS_BYTE_N        2       // BYTE数目
`define     UARTS_BYTE_W        2       // BYTE数目位宽
`define     UARTS_DATA_W        16      // 数据位宽

// 波特率
`define     UART_CNT_BPS        434     // 波特率计数
`define     UART_HALF_BPS       217     // 波特率半计数
// `define     UARTR_CNT_BPS       44      // 波特率计数
// `define     UARTR_HALF_BPS      22      // 波特率半计数

// 波特率计算
// 9600 bps     ->  1/9600/20e-9    ->  5208    (8 bit / 1 ms)      # 104_140
// 115200 bps   ->  1/115200/20e-9  ->  434     (8 bit / 90 us)     # 8_680