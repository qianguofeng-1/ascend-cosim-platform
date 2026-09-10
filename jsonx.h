/* ============================================================================
 * jsonx.h - 小型 JSON 解析/写出库(UTF-8, 零外部依赖)
 *
 * 提供:
 *   1) JsonValue 树(对象/数组/字符串/数值/布尔/空)及解析/释放;
 *   2) 对“对象”成员的便捷读取(jsonGetNumber / jsonGetInt / ...);
 *   3) 可变字符串缓冲 StringBuffer, 用于高效拼接输出 JSON。
 *
 * 注意: 本库解析与写出均只处理内存中的完整文本, 不做流式解析。
 * ==========================================================================*/
#ifndef JSONX_H
#define JSONX_H
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * JSON 值类型
 * ==========================================================================*/
typedef enum {
    JSON_OBJECT = 0,   /* 对象: 有序 键->值 成员表 */
    JSON_ARRAY  = 1,   /* 数组                     */
    JSON_STRING = 2,   /* 字符串(UTF-8)            */
    JSON_NUMBER = 3,   /* 数值(double)             */
    JSON_BOOL   = 4,   /* 布尔                     */
    JSON_NULL   = 5    /* 空值                     */
} JsonValueType;

/* ============================================================================
 * JSON 值节点: 根据 type 使用对应联合成员
 * ==========================================================================*/
typedef struct JsonValue JsonValue;
struct JsonValue {
    JsonValueType type;
    union {
        /* JSON_OBJECT */
        struct {
            char        **keys;    /* 成员名数组(每项 strdup, 需释放) */
            JsonValue  **values;   /* 成员值数组                      */
            int           count;   /* 成员个数                        */
        } object;
        /* JSON_ARRAY */
        struct {
            JsonValue **items;     /* 元素数组                        */
            int          count;    /* 元素个数                        */
        } array;
        char  *stringValue;        /* JSON_STRING                     */
        double numberValue;        /* JSON_NUMBER                     */
        int    boolValue;          /* JSON_BOOL                       */
    } u;
};

/* ============================================================================
 * 解析 / 释放 / 错误信息
 * ==========================================================================*/
extern JsonValue *jsonParse(const char *text);      /* 成功返回根节点, 失败 NULL */
extern void jsonValueFree(JsonValue *value);        /* 递归释放整棵树             */
extern const char *jsonLastError(void);             /* 最近一次解析错误描述       */

/* ============================================================================
 * 对象成员便捷读取(用于从请求 JSON 中取值)
 * ==========================================================================*/
extern JsonValue *jsonGetMember(JsonValue *object, const char *key);       /* 成员或 NULL   */
extern double     jsonGetNumber(JsonValue *object, const char *key, double defaultValue);
extern int        jsonGetInt(JsonValue *object, const char *key, int defaultValue);
extern const char *jsonGetString(JsonValue *object, const char *key, const char *defaultValue);
extern int        jsonArrayLength(JsonValue *array);                       /* 非数组返回 0  */

/* ============================================================================
 * 可变字符串缓冲(StringBuffer): 用于组装 HTTP 响应 JSON
 * 约定: data 始终以 '\0' 结尾, length 为不含结尾符的字节长度。
 * ==========================================================================*/
typedef struct {
    char *data;      /* 缓冲内容(以 '\0' 结尾) */
    int   length;    /* 有效字节数(不含结尾符) */
    int   capacity;  /* 已分配容量             */
} StringBuffer;

extern void strbufInit(StringBuffer *sb);
extern void strbufFree(StringBuffer *sb);
extern void strbufAppend(StringBuffer *sb, const char *text);               /* 追加普通文本   */
extern void strbufAppendFormat(StringBuffer *sb, const char *fmt, ...);     /* 追加格式化文本 */
extern void strbufAppendChar(StringBuffer *sb, char character);              /* 追加单字符     */
extern void strbufAppendNumber(StringBuffer *sb, double value);             /* 追加 JSON 数字(紧凑) */
extern void strbufAppendInt64(StringBuffer *sb, long long value);           /* 追加 JSON 整数 */
extern void strbufAppendEscapedString(StringBuffer *sb, const char *text);  /* 追加带引号+转义的 JSON 字符串 */

#endif /* JSONX_H */
