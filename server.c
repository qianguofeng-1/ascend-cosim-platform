/* ============================================================================
 * server.c - 单文件 exe 入口: Win32 + Winsock 内嵌 HTTP 服务
 *
 * 运行方式: 双击 exe -> 自动寻找空闲端口(8098 起) -> 自动打开默认浏览器
 * 访问前端页面; 前端(index.html 内嵌于 webdata.h)通过 POST /api/run
 * 调用 C 引擎完成仿真。
 *
 * 运维日志: 启动时初始化 logx 模块, 默认写入当前目录 cosim_engine.log;
 * 可用环境变量 COSIM_LOG_FILE / COSIM_LOG_LEVEL 覆盖(详见 logx.h)。
 * ==========================================================================*/
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>   /* clock(): 请求耗时统计 */

#include "engine.h"
#include "webdata.h"   /* WEB_INDEX[] / WEB_INDEX_LEN 由构建脚本生成 */

/* 请求序号(用于日志中区分每次请求) */
static long g_requestSequenceNumber = 0;

/* ============================================================================
 * 底层收发小工具
 * ==========================================================================*/

/* 尽可能把 buf[0..len) 全部发送出去(处理部分发送) */
static void sendAllBytes(SOCKET clientSocket, const char *buffer, int length) {
    int offset = 0;
    while (offset < length) {
        int sent = send(clientSocket, buffer + offset, length - offset, 0);
        if (sent == SOCKET_ERROR || sent <= 0)
            break;
        offset += sent;
    }
}

/* 写一条完整的 HTTP 响应 */
static void writeHttpResponse(SOCKET clientSocket, const char *contentType,
                              int statusCode, const char *body, int bodyLength,
                              const char *extraHeaders) {
    char header[512];
    const char *reasonPhrase =
        (statusCode == 200) ? "OK"
        : (statusCode == 204) ? "No Content" : "Error";
    int headerLength;

    headerLength = sprintf(header,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n%s\r\n",
        statusCode, reasonPhrase, contentType, bodyLength,
        extraHeaders ? extraHeaders : "");
    sendAllBytes(clientSocket, header, headerLength);
    sendAllBytes(clientSocket, body, bodyLength);
}

/* 写一条 JSON 响应(200) */
static void writeJsonResponse(SOCKET clientSocket, const char *jsonText) {
    writeHttpResponse(clientSocket, "application/json; charset=utf-8", 200,
                      jsonText, (int)strlen(jsonText), NULL);
}

/* 在请求头区查找指定头部(大小写不敏感), 返回其值(指向原缓冲区, 跳过前导空白) */
static const char *findRequestHeader(const char *headerData, int headerLength,
                                     const char *headerName) {
    const char *lineStart = headerData;
    const char *end = headerData + headerLength;
    int nameLength = (int)strlen(headerName);

    while (lineStart + nameLength + 2 <= end) {
        if (!_strnicmp(lineStart, headerName, (size_t)nameLength) &&
            lineStart[nameLength] == ':') {
            const char *value = lineStart + nameLength + 1;
            while (value < end && (*value == ' ' || *value == '\t'))
                value++;
            return value;
        }
        {
            const char *lineEnd = strstr(lineStart, "\r\n");
            if (!lineEnd || lineEnd >= end)
                break;
            lineStart = lineEnd + 2;
        }
    }
    return NULL;
}

/* ============================================================================
 * 处理单条 HTTP 连接(一次请求, 读完即关)
 * ==========================================================================*/
static void handleConnection(SOCKET clientSocket, const char *request, int requestLength) {
    const char *headerBodySep;          /* 请求头/体分隔 "\r\n\r\n" 位置 */
    int headerLength;
    char requestLine[1024];
    char method[16], path[512], version[32];
    const char *contentLengthText;
    int bodyLength;
    const char *bodyStart;
    long requestNo = ++g_requestSequenceNumber;

    /* 没有请求头结束标记则直接拒绝 */
    headerBodySep = strstr(request, "\r\n\r\n");
    if (!headerBodySep) {
        writeHttpResponse(clientSocket, "text/plain", 400, "bad request", 11, NULL);
        return;
    }
    headerLength = (int)(headerBodySep - request);

    /* 拷贝首行并解析 method / path / version */
    {
        int copyLength = headerLength < 1023 ? headerLength : 1023;
        memcpy(requestLine, request, (size_t)copyLength);
        requestLine[copyLength] = '\0';
    }
    if (sscanf(requestLine, "%15s %511s %31s", method, path, version) < 2) {
        writeHttpResponse(clientSocket, "text/plain", 400, "bad request", 11, NULL);
        return;
    }

    /* 从 Content-Length 头取请求体长度 */
    contentLengthText = findRequestHeader(request, headerLength, "Content-Length");
    bodyLength = contentLengthText ? atoi(contentLengthText) : 0;
    if (bodyLength < 0)
        bodyLength = 0;
    bodyStart = headerBodySep + 4;

    logxLog(LOG_LEVEL_INFO, "HTTP[%ld] %s %s 请求体=%d字节", requestNo, method, path,
            bodyLength);

    /* CORS 预检(跨地址回退访问时浏览器会发送) */
    if (!strcmp(method, "OPTIONS")) {
        writeHttpResponse(clientSocket, "text/plain", 204, "", 0,
            "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nAccess-Control-Max-Age: 600");
        return;
    }
    /* 首页 */
    if (!strcmp(method, "GET") && (!strcmp(path, "/") || !strcmp(path, "/index.html"))) {
        writeHttpResponse(clientSocket, "text/html; charset=utf-8", 200,
                          (const char *)WEB_INDEX, (int)WEB_INDEX_LEN, NULL);
        return;
    }
    /* 版本信息 */
    if (!strcmp(method, "GET") && !strcmp(path, "/api/info")) {
        writeJsonResponse(clientSocket,
                          "{\"name\":\"软硬协同调度仿真平台\",\"ver\":\"1.0.0\",\"engine\":\"C cycle-level\",\"ok\":1}");
        return;
    }
    /* 浏览器图标: 空响应即可 */
    if (!strcmp(method, "GET") && !strcmp(path, "/favicon.ico")) {
        writeHttpResponse(clientSocket, "image/x-icon", 204, "", 0, NULL);
        return;
    }
    /* ---- 仿真主入口 ---- */
    if (!strcmp(method, "POST") && !strcmp(path, "/api/run")) {
        char *jsonCopy;
        JsonValue *requestJson;
        StringBuffer output;
        clock_t startClock = clock();

        if (bodyLength <= 0 || bodyLength > (int)(request + requestLength - bodyStart)) {
            writeJsonResponse(clientSocket, "{\"ok\":0,\"err\":\"空请求体\"}");
            return;
        }

        /* 拷贝请求体为 NUL 结尾字符串后解析 JSON */
        jsonCopy = (char *)malloc((size_t)bodyLength + 1);
        memcpy(jsonCopy, bodyStart, (size_t)bodyLength);
        jsonCopy[bodyLength] = '\0';
        requestJson = jsonParse(jsonCopy);
        free(jsonCopy);
        if (!requestJson) {
            StringBuffer errorOutput;
            strbufInit(&errorOutput);
            strbufAppend(&errorOutput, "{\"ok\":0,\"err\":");
            strbufAppendEscapedString(&errorOutput, "JSON 解析失败");
            strbufAppend(&errorOutput, "}");
            writeJsonResponse(clientSocket, errorOutput.data);
            strbufFree(&errorOutput);
            logxLog(LOG_LEVEL_ERROR, "HTTP[%ld] /api/run JSON 解析失败: %s",
                    requestNo, jsonLastError());
            return;
        }

        strbufInit(&output);
        runEngineRequest(requestJson, &output);
        jsonValueFree(requestJson);
        writeJsonResponse(clientSocket, output.data);

        logxLog(LOG_LEVEL_INFO, "HTTP[%ld] /api/run 响应 %d 字节(耗时 %.1fms)",
                requestNo, output.length,
                1000.0 * (double)(clock() - startClock) / CLOCKS_PER_SEC);
        strbufFree(&output);
        return;
    }

    writeHttpResponse(clientSocket, "text/plain", 404, "not found", 9, NULL);
}

/* ============================================================================
 * 内存查找子串(供 serveLoop 解析请求边界用)
 * ==========================================================================*/
static const char *memmemFind(const void *haystack, size_t haystackLength,
                              const void *needle, size_t needleLength) {
    const char *base = (const char *)haystack;
    size_t position;
    if (needleLength == 0)
        return base;
    if (haystackLength < needleLength)
        return NULL;
    for (position = 0; position + needleLength <= haystackLength; position++) {
        if (!memcmp(base + position, needle, needleLength))
            return base + position;
    }
    return NULL;
}

/* ============================================================================
 * 接收循环: 不断 accept, 读完整请求(头+体), 逐条处理
 * ==========================================================================*/
static void serveLoop(SOCKET listenSocket) {
    for (;;) {
        SOCKET clientSocket = accept(listenSocket, NULL, NULL);
        if (clientSocket == INVALID_SOCKET)
            break;

        /* 读请求: 先收头, 再按 Content-Length 收满 body */
        {
            static char receiveBuffer[2 * 1024 * 1024];
            int received = 0;
            DWORD receiveTimeoutMs = 1000;
            setsockopt(clientSocket, SOL_SOCKET, SO_RCVTIMEO,
                       (const char *)&receiveTimeoutMs, sizeof(receiveTimeoutMs));

            while (received < (int)sizeof(receiveBuffer)) {
                int got = recv(clientSocket, receiveBuffer + received,
                               (int)sizeof(receiveBuffer) - received - 1, 0);
                const char *separator;
                if (got <= 0)
                    break;
                received += got;
                receiveBuffer[received] = '\0';

                separator = memmemFind(receiveBuffer, (size_t)received, "\r\n\r\n", 4);
                if (separator) {
                    /* 头已收满: 继续收 body 到 Content-Length */
                    int bodyNeeded = 0;
                    const char *contentLengthText =
                        findRequestHeader(receiveBuffer, received, "Content-Length");
                    int have;
                    if (contentLengthText)
                        bodyNeeded = atoi(contentLengthText);
                    have = (int)(received - (separator - receiveBuffer) - 4);
                    while (have < bodyNeeded && received < (int)sizeof(receiveBuffer)) {
                        int more = recv(clientSocket, receiveBuffer + received,
                                        (int)sizeof(receiveBuffer) - received - 1, 0);
                        if (more <= 0)
                            break;
                        received += more;
                        have = (int)(received - (separator - receiveBuffer) - 4);
                    }
                    break;
                }
                /* 异常大且没有头结束符的请求直接放弃 */
                if (received >= 4 &&
                    !memmemFind(receiveBuffer, (size_t)received, "\r\n\r\n", 4) &&
                    received > 65536)
                    break;
            }
            if (received > 0)
                handleConnection(clientSocket, receiveBuffer, received);
        }
        closesocket(clientSocket);
    }
}

/* ============================================================================
 * 尝试在 hostAddr:port 上创建监听套接字(成功返回句柄)
 * ==========================================================================*/
static int tryBindAndListen(unsigned long hostAddress, int port, int *outPort) {
    SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in address;
    int reuseAddr = 1;
    if (listenSocket == INVALID_SOCKET)
        return -1;

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = hostAddress;
    address.sin_port = htons((u_short)port);
    setsockopt(listenSocket, SOL_SOCKET, SO_REUSEADDR,
               (const char *)&reuseAddr, sizeof(reuseAddr));
    if (bind(listenSocket, (struct sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
        closesocket(listenSocket);
        return -1;
    }
    if (listen(listenSocket, 8) == SOCKET_ERROR) {
        closesocket(listenSocket);
        return -1;
    }
    *outPort = port;
    return (int)listenSocket;
}

/* 获取本机第一个非回环 IPv4 地址(用于展示浏览器访问地址) */
static int getLanIPv4Address(char *buffer, int capacity) {
    char hostName[256];
    struct hostent *hostEntry;
    int addressIndex;
    if (gethostname(hostName, sizeof(hostName)) != 0)
        return 0;
    hostEntry = gethostbyname(hostName);
    if (!hostEntry)
        return 0;
    for (addressIndex = 0;
         hostEntry->h_addr_list && hostEntry->h_addr_list[addressIndex];
         addressIndex++) {
        const unsigned char *address =
            (const unsigned char *)hostEntry->h_addr_list[addressIndex];
        if (hostEntry->h_length >= 4) {
            /* 排除 回环(127.x)/0.x/链路本地(169.254.x) */
            if (address[0] != 127 && address[0] != 0 && address[0] != 169) {
                sprintf(buffer, "%u.%u.%u.%u",
                        address[0], address[1], address[2], address[3]);
                return 1;
            }
        }
    }
    return 0;
}

/* ============================================================================
 * 程序入口
 * ==========================================================================*/
int main(int argc, char **argv) {
    int requestedPort = 0;
    int argIndex;
    char hostArgument[64];
    unsigned long bindAddress;
    int loopbackOnly;
    SOCKET listenSocket = INVALID_SOCKET;
    int listenPort = 0;
    char lanAddress[64];
    int hasLanAddress;
    char openHost[64];
    char url[160];
    WSADATA wsaData;

    hostArgument[0] = '\0';

    /* ---- 命令行参数解析 ---- */
    for (argIndex = 1; argIndex < argc; argIndex++) {
        if (!strcmp(argv[argIndex], "--port") && argIndex + 1 < argc) {
            requestedPort = atoi(argv[++argIndex]);
        } else if (!strcmp(argv[argIndex], "--host") && argIndex + 1 < argc) {
            strncpy(hostArgument, argv[++argIndex], 63);
            hostArgument[63] = '\0';
        } else if (!strcmp(argv[argIndex], "--no-browser")) {
            /* 支持: 不自动打开浏览器 */
        } else if (!strcmp(argv[argIndex], "--run") && argIndex + 1 < argc) {
            /* 自测模式: 读配置 JSON -> 仿真 -> 打印结果 JSON 到 stdout */
            FILE *configFile;
            long fileSize;
            char *fileContent;
            size_t bytesRead;
            JsonValue *requestJson;
            StringBuffer output;

            if (getenv("COSIM_DBG"))
                g_cosim_dbg = 1;
            logxOpen(NULL);
            logxSetLevel(0);

            configFile = fopen(argv[argIndex + 1], "rb");
            if (!configFile) {
                printf("cannot open %s\n", argv[argIndex + 1]);
                return 3;
            }
            fseek(configFile, 0, SEEK_END);
            fileSize = ftell(configFile);
            fseek(configFile, 0, SEEK_SET);
            fileContent = (char *)malloc((size_t)fileSize + 1);
            bytesRead = fread(fileContent, 1, (size_t)fileSize, configFile);
            fclose(configFile);
            fileContent[bytesRead] = '\0';

            requestJson = jsonParse(fileContent);
            free(fileContent);
            if (!requestJson) {
                printf("json error: %s\n", jsonLastError());
                logxClose();
                return 4;
            }
            strbufInit(&output);
            runEngineRequest(requestJson, &output);
            fwrite(output.data, 1, (size_t)output.length, stdout);
            strbufFree(&output);
            jsonValueFree(requestJson);
            logxClose();
            return 0;
        }
    }

    /* ---- 启动 Winsock 并配置控制台 UTF-8 输出 ---- */
    SetConsoleOutputCP(65001);
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("WSAStartup failed\n");
        return 1;
    }

    logxOpen(NULL);
    logxSetLevel(0);
    logxLog(LOG_LEVEL_INFO, "仿真平台进程启动(argc=%d)", argc);

    /* 3080 为 DeepSeek Harness 保留端口, 本应用不占用 */
    if (requestedPort == 3080) {
        printf("端口 3080 为 DeepSeek Harness 保留, 已改用默认范围(8098-8128)。\n");
        logxLog(LOG_LEVEL_WARN, "请求端口 3080 被保留, 改用默认端口范围");
        requestedPort = 0;
    }

    /* 绑定地址: 默认 0.0.0.0(全部网卡), --host 可指定 */
    bindAddress = INADDR_ANY;
    loopbackOnly = 0;
    if (hostArgument[0]) {
        unsigned long parsed = inet_addr(hostArgument);
        if (parsed == INADDR_NONE) {
            printf("无效 --host: %s\n", hostArgument);
            logxLog(LOG_LEVEL_ERROR, "无效 --host: %s", hostArgument);
            WSACleanup();
            logxClose();
            return 2;
        }
        bindAddress = parsed;
        loopbackOnly = (parsed == htonl(INADDR_LOOPBACK)) ? 1 : 0;
    }

    /* 先试用户指定端口, 失败则在默认范围 8098-8128 中找空闲 */
    if (requestedPort > 0 && requestedPort < 65536) {
        listenSocket = (SOCKET)tryBindAndListen(bindAddress, requestedPort, &listenPort);
    }
    if (listenSocket == INVALID_SOCKET) {
        int candidatePort;
        for (candidatePort = 8098; candidatePort <= 8128; candidatePort++) {
            listenSocket = (SOCKET)tryBindAndListen(bindAddress, candidatePort, &listenPort);
            if (listenSocket != INVALID_SOCKET)
                break;
        }
    }
    if (listenSocket == INVALID_SOCKET) {
        printf("无法监听端口(8098-8128 被占用)\n");
        logxLog(LOG_LEVEL_ERROR, "端口范围 8098-8128 全部被占用, 服务启动失败");
        WSACleanup();
        logxClose();
        return 2;
    }
    logxLog(LOG_LEVEL_INFO, "监听成功: 地址端口=%d", listenPort);

    /* 浏览器访问地址: 优先局域网 IP(避免与 127.0.0.1 上的 DeepSeek Harness 混淆) */
    hasLanAddress = getLanIPv4Address(lanAddress, sizeof(lanAddress));
    if (loopbackOnly)
        strcpy(openHost, "127.0.0.1");
    else if (hasLanAddress)
        strcpy(openHost, lanAddress);
    else
        strcpy(openHost, "127.0.0.1");

    printf("==========================================================\n");
    printf("  软硬协同调度仿真平台  [C 引擎 v1.0]\n");
    printf("  访问地址:   http://%s:%d/  (局域网/本机均可)\n", openHost, listenPort);
    printf("  本机备用:   http://127.0.0.1:%d/\n", listenPort);
    printf("  正在打开默认浏览器 ...\n");
    printf("  关闭本窗口或按 Ctrl+C 即退出仿真服务\n");
    printf("  (注意: 3080 为 DeepSeek Harness 保留端口, 本应用使用 8098-8128)\n");
    printf("==========================================================\n");
    fflush(stdout);

    logxLog(LOG_LEVEL_INFO, "服务地址 http://%s:%d/, 打开默认浏览器", openHost, listenPort);
    sprintf(url, "http://%s:%d/", openHost, listenPort);
    ShellExecuteA(NULL, "open", url, NULL, NULL, SW_SHOWNORMAL);

    /* ---- 事件循环: 一直服务直到窗口被关闭 ---- */
    serveLoop(listenSocket);

    closesocket(listenSocket);
    WSACleanup();
    logxLog(LOG_LEVEL_INFO, "服务退出");
    logxClose();
    return 0;
}
