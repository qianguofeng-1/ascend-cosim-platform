/* ============================================================================
 * logx.c - 轻量运维日志模块实现
 * 见 logx.h 头部的设计说明。
 * ==========================================================================*/
#include "logx.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>   /* GetLocalTime: 毫秒级时间戳 */
#endif

/* ---------------- 模块内部状态 ---------------- */
static FILE    *g_logStream = NULL; /* 日志文件句柄(NULL=日志不可用)        */
static int      g_logLevel  = LOG_LEVEL_INFO; /* 当前日志级别(默认 INFO)   */
static int      g_traceDepth = 0;   /* 函数调用树缩进深度                  */

/* ---------------- 内部小工具 ---------------- */

/* 取本地时间字符串 yyyy-mm-dd hh:mm:ss.mmm, 写入 out(须 >=32 字节) */
static void timestampNow(char *out, size_t capacity) {
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime(&st);
    snprintf(out, capacity, "%04u-%02u-%02u %02u:%02u:%02u.%03u",
             st.wYear, st.wMonth, st.wDay,
             st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
#else
    /* 非 Windows 环境退化为秒级时间(本项目面向 Windows, 极少走到这里) */
    time_t now = time(NULL);
    struct tm *localTimeInfo = localtime(&now);
    if (localTimeInfo)
        snprintf(out, capacity, "%04d-%02d-%02d %02d:%02d:%02d.000",
                 localTimeInfo->tm_year + 1900, localTimeInfo->tm_mon + 1,
                 localTimeInfo->tm_mday,
                 localTimeInfo->tm_hour, localTimeInfo->tm_min, localTimeInfo->tm_sec);
    else
        snprintf(out, capacity, "????-??-?? ??:??:??.???");
#endif
}

/* 级别 -> 显示名(定宽, 便于对齐) */
static const char *levelName(int level) {
    switch (level) {
        case LOG_LEVEL_ERROR: return "ERROR";
        case LOG_LEVEL_WARN:  return "WARN ";
        case LOG_LEVEL_DEBUG: return "DEBUG";
        default:              return "INFO ";
    }
}

/* 带时间戳与缩进写一行日志的核心实现 */
static void writeLine(int level, const char *line) {
    char timestamp[40];
    int  indentSpaces;
    int  spaceColumn;

    if (!g_logStream)
        return;
    timestampNow(timestamp, sizeof(timestamp));

    indentSpaces = g_traceDepth * 2;               /* 每层函数缩进 2 空格 */
    fprintf(g_logStream, "[%s][%s] ", timestamp, levelName(level));
    for (spaceColumn = 0; spaceColumn < indentSpaces; spaceColumn++)
        fputc(' ', g_logStream);
    fputs(line, g_logStream);
    fputc('\n', g_logStream);
}

/* ---------------- 对外接口 ---------------- */

void logxOpen(const char *path) {
    const char *targetPath = path;
    char envPath[1024];

    logxClose();

    /* 未显式指定路径时: 环境变量 COSIM_LOG_FILE -> 默认文件名 */
    if (!targetPath) {
        const char *env = getenv("COSIM_LOG_FILE");
        if (env && env[0]) {
            strncpy(envPath, env, sizeof(envPath) - 1);
            envPath[sizeof(envPath) - 1] = '\0';
            targetPath = envPath;
        }
    }
    if (!targetPath)
        targetPath = "cosim_engine.log";

    g_logStream = fopen(targetPath, "a");
    if (!g_logStream)
        return;   /* 打不开日志文件时静默禁用日志, 不影响主流程 */

    logxLog(LOG_LEVEL_INFO, "========== 日志文件已打开: %s ==========", targetPath);
}

void logxClose(void) {
    if (g_logStream) {
        logxLog(LOG_LEVEL_INFO, "========== 日志文件关闭 ==========");
        fflush(g_logStream);
        fclose(g_logStream);
        g_logStream = NULL;
    }
    g_traceDepth = 0;
}

void logxSetLevel(int level) {
    if (level >= LOG_LEVEL_ERROR && level <= LOG_LEVEL_DEBUG) {
        g_logLevel = level;
        return;
    }
    /* 传入非法值(如 0)时读取环境变量 COSIM_LOG_LEVEL, 缺省 INFO */
    {
        const char *env = getenv("COSIM_LOG_LEVEL");
        int envLevel = env ? atoi(env) : 0;
        if (envLevel >= LOG_LEVEL_ERROR && envLevel <= LOG_LEVEL_DEBUG)
            g_logLevel = envLevel;
        else
            g_logLevel = LOG_LEVEL_INFO;
    }
}

int logxGetLevel(void) {
    return g_logLevel;
}

int logxEnabled(int level) {
    return g_logStream != NULL && level <= g_logLevel;
}

void logxLog(int level, const char *fmt, ...) {
    char msg[2048];
    va_list args;

    if (!logxEnabled(level))
        return;

    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    writeLine(level, msg);
}

/* 组装 "标记 函数名[: 附加信息]" 形式的日志文本并输出 */
static void writeTraceLine(int level, const char *marker,
                           const char *functionName, const char *fmt, va_list args) {
    char msg[2048];

    if (fmt && fmt[0]) {
        vsnprintf(msg, sizeof(msg), fmt, args);
        snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "  [%s]", functionName);
    } else {
        snprintf(msg, sizeof(msg), "%s", functionName);
    }

    /* 在“父级缩进”处输出, 便于体现嵌套关系 */
    {
        int savedDepth = g_traceDepth;
        char full[4096];
        int charsWritten = snprintf(full, sizeof(full), "%s %s", marker, msg);
        (void)charsWritten;
        g_traceDepth = savedDepth > 0 ? savedDepth - 1 : 0;
        writeLine(level, full);
        g_traceDepth = savedDepth;
    }
}

void logxTraceEnter(int level, const char *functionName, const char *fmt, ...) {
    va_list args;

    if (!g_logStream)
        return;

    g_traceDepth++;                        /* 进入函数: 缩进加深 1 层 */
    if (level > g_logLevel)                /* 级别不足: 只更新缩进不输出 */
        return;

    va_start(args, fmt);
    writeTraceLine(level, "-->", functionName, fmt, args);
    va_end(args);
}

void logxTraceLeave(int level, const char *functionName, const char *fmt, ...) {
    va_list args;

    if (!g_logStream)
        return;

    if (g_traceDepth > 0)
        g_traceDepth--;                    /* 离开函数: 缩进恢复 1 层 */
    if (level > g_logLevel)
        return;

    va_start(args, fmt);
    writeTraceLine(level, "<--", functionName, fmt, args);
    va_end(args);
}

void logxFlush(void) {
    if (g_logStream)
        fflush(g_logStream);
}
