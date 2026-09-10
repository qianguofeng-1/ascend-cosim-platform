/* ============================================================================
 * logx.h - 轻量运维日志模块(记录到专用日志文件)
 *
 * 用途：
 *   1) 记录引擎运行的关键事件(请求到达/配置解码/建模/仿真/出错等)；
 *   2) 记录“函数调用关系”：(日志文件里通过 "--> 进入函数 / <-- 离开函数"
 *      对 + 缩进，还原一次请求处理时的函数调用树)，方便运维定位问题；
 *   3) 全部写入一个独立的日志文件(默认 cosim_engine.log)，不影响业务输出。
 *
 * 运行期开关(通过环境变量控制，无需改代码)：
 *   COSIM_LOG_FILE    日志文件路径(缺省: 当前目录 cosim_engine.log)
 *   COSIM_LOG_LEVEL   日志级别 1..4(缺省 3=INFO)
 *                       1=ERROR 仅错误  2=WARN  +警告
 *                       3=INFO  +主流程调用链(推荐运维)  4=DEBUG 含内部细节
 *
 * 说明: 本模块为单线程 HTTP 服务设计, 不做线程同步。
 * ==========================================================================*/
#ifndef LOGX_H
#define LOGX_H

/* ---------------- 日志级别 ---------------- */
enum {
    LOG_LEVEL_ERROR = 1,   /* 仅记录错误                      */
    LOG_LEVEL_WARN  = 2,   /* 记录警告及以下                  */
    LOG_LEVEL_INFO  = 3,   /* 记录主流程信息(默认, 运维建议值) */
    LOG_LEVEL_DEBUG = 4    /* 记录全部调试细节                 */
};

/* ---------------- 接口 ---------------- */

/* 打开日志文件(追加写)。
 * path 为 NULL 时依次取环境变量 COSIM_LOG_FILE、默认文件名 "cosim_engine.log"。
 * 打开失败时静默降级(后续日志全部丢弃), 绝不影响主流程。 */
void logxOpen(const char *path);

/* 关闭日志文件(进程退出前调用) */
void logxClose(void);

/* 设置日志级别; 传入 <=0 表示读取环境变量 COSIM_LOG_LEVEL(缺省 INFO)。
 * 应在 logxOpen 之后、处理第一个请求之前调用一次。 */
void logxSetLevel(int level);

/* 查询当前日志级别 */
int logxGetLevel(void);

/* 判断某级别当前是否会被输出(供调用方避免昂贵的日志参数构造) */
int logxEnabled(int level);

/* 记录一条普通日志, fmt 同 printf */
void logxLog(int level, const char *fmt, ...);

/* 函数调用关系追踪:
 *   logxTraceEnter - 进入函数时调用: 输出 "--> 函数名 附加信息", 缩进+1
 *   logxTraceLeave - 离开函数前调用: 缩进-1, 输出 "<-- 函数名 附加信息"
 * 二者成对使用可在日志中还原调用树。 */
void logxTraceEnter(int level, const char *functionName, const char *fmt, ...);
void logxTraceLeave(int level, const char *functionName, const char *fmt, ...);

/* 冲刷缓冲到磁盘(请求处理结束后调用, 保证日志及时落盘) */
void logxFlush(void);

#endif /* LOGX_H */
