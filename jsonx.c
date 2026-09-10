/* ============================================================================
 * jsonx.c - 小型 JSON 解析/写出实现(UTF-8, 无外部依赖)
 *
 * 结构: 单趟递归下降解析器 + JsonValue 树 + StringBuffer 写出工具。
 * 本文件不包含任何仿真业务逻辑, 解析失败统一记录到 g_lastError 中,
 * 由 jsonLastError() 返回。
 * ==========================================================================*/
#include "jsonx.h"
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ---------------- 模块内部状态 ---------------- */
static const char *g_lastError = NULL;   /* 最近一次解析错误描述 */

const char *jsonLastError(void) {
    return g_lastError ? g_lastError : "";
}

/* ---------------- 内部小工具 ---------------- */

/* 新建指定类型的空节点 */
static JsonValue *jsonNew(JsonValueType type) {
    JsonValue *value = (JsonValue *)calloc(1, sizeof(JsonValue));
    value->type = type;
    return value;
}

/* 文本指针游标: 整个解析过程中向前移动 */
typedef struct {
    const char *pos;   /* 当前读取位置 */
} Cursor;

/* 跳过空白字符 */
static void skipWhitespace(Cursor *cur) {
    while (*cur->pos == ' ' || *cur->pos == '\t' ||
           *cur->pos == '\n' || *cur->pos == '\r')
        cur->pos++;
}

static JsonValue *parseValue(Cursor *cur);   /* 前向声明: 递归解析任意值 */

/* 解析 JSON 字符串字面量(含转义, \uXXXX 组装为 UTF-8) */
static JsonValue *parseString(Cursor *cur) {
    if (*cur->pos != '"') {
        g_lastError = "expect string";
        return NULL;
    }
    cur->pos++;                                  /* 跳过开引号 */

    StringBuffer text;
    strbufInit(&text);
    while (*cur->pos && *cur->pos != '"') {
        char currentChar = *cur->pos++;
        if (currentChar == '\\') {                 /* 转义字符 */
            char escapeChar = *cur->pos++;
            switch (escapeChar) {
                case 'n':  strbufAppendChar(&text, '\n'); break;
                case 't':  strbufAppendChar(&text, '\t'); break;
                case 'r':  strbufAppendChar(&text, '\r'); break;
                case 'b':  strbufAppendChar(&text, '\b'); break;
                case 'f':  strbufAppendChar(&text, '\f'); break;
                case '"':  strbufAppendChar(&text, '"');  break;
                case '/':  strbufAppendChar(&text, '/');  break;
                case '\\': strbufAppendChar(&text, '\\'); break;
                case 'u': {                       /* \uXXXX: 简化为 UTF-8 编码 */
                    unsigned codePoint = 0;
                    int digitIndex;
                    for (digitIndex = 0; digitIndex < 4; digitIndex++) {
                        char hexDigitChar = *cur->pos++;
                        codePoint <<= 4;
                        if (hexDigitChar >= '0' && hexDigitChar <= '9')
                            codePoint |= (unsigned)(hexDigitChar - '0');
                        else if (hexDigitChar >= 'a' && hexDigitChar <= 'f')
                            codePoint |= (unsigned)(hexDigitChar - 'a' + 10);
                        else if (hexDigitChar >= 'A' && hexDigitChar <= 'F')
                            codePoint |= (unsigned)(hexDigitChar - 'A' + 10);
                        else {
                            g_lastError = "bad \\u";
                            break;
                        }
                    }
                    if (codePoint < 0x80) {
                        strbufAppendChar(&text, (char)codePoint);
                    } else if (codePoint < 0x800) {
                        strbufAppendChar(&text, (char)(0xC0 | (codePoint >> 6)));
                        strbufAppendChar(&text, (char)(0x80 | (codePoint & 63)));
                    } else {
                        strbufAppendChar(&text, (char)(0xE0 | (codePoint >> 12)));
                        strbufAppendChar(&text, (char)(0x80 | ((codePoint >> 6) & 63)));
                        strbufAppendChar(&text, (char)(0x80 | (codePoint & 63)));
                    }
                    break;
                }
                default:
                    g_lastError = "bad esc";
                    strbufFree(&text);
                    return NULL;
            }
        } else if ((unsigned char)currentChar < 0x20) {  /* 禁止未转义控制字符 */
            g_lastError = "ctrl char";
            strbufFree(&text);
            return NULL;
        } else {
            strbufAppendChar(&text, currentChar);
        }
    }
    if (*cur->pos != '"') {
        g_lastError = "unterminated string";
        strbufFree(&text);
        return NULL;
    }
    cur->pos++;                                  /* 跳过闭引号 */

    JsonValue *value = jsonNew(JSON_STRING);
    value->u.stringValue = text.data;            /* 缓冲所有权转移给节点 */
    return value;
}

/* 解析 JSON 对象: { "key": value, ... } */
static JsonValue *parseObject(Cursor *cur) {
    cur->pos++;                                  /* 跳过 '{' */
    JsonValue *obj = jsonNew(JSON_OBJECT);
    int capacity = 8;
    obj->u.object.keys   = (char **)malloc((size_t)capacity * sizeof(char *));
    obj->u.object.values = (JsonValue **)malloc((size_t)capacity * sizeof(JsonValue *));
    obj->u.object.count  = 0;
    for (;;) {
        skipWhitespace(cur);
        if (*cur->pos == '}') {                  /* 空对象或收尾 */
            cur->pos++;
            break;
        }
        if (obj->u.object.count == capacity) {   /* 扩容 */
            capacity *= 2;
            obj->u.object.keys = (char **)realloc(obj->u.object.keys,
                                                  (size_t)capacity * sizeof(char *));
            obj->u.object.values = (JsonValue **)realloc(obj->u.object.values,
                                                         (size_t)capacity * sizeof(JsonValue *));
        }
        JsonValue *keyNode = parseString(cur);
        if (!keyNode) {
            jsonValueFree(obj);
            return NULL;
        }
        skipWhitespace(cur);
        if (*cur->pos != ':') {
            g_lastError = "expect ':'";
            jsonValueFree(obj);
            return NULL;
        }
        cur->pos++;
        JsonValue *value = parseValue(cur);
        if (!value) {
            jsonValueFree(obj);
            return NULL;
        }
        obj->u.object.keys[obj->u.object.count]   = keyNode->u.stringValue;
        free(keyNode);                           /* 节点壳释放, 字符串保留 */
        obj->u.object.values[obj->u.object.count] = value;
        obj->u.object.count++;
        skipWhitespace(cur);
        if (*cur->pos == ',') {
            cur->pos++;
            continue;
        }
        if (*cur->pos == '}') {
            cur->pos++;
            break;
        }
        g_lastError = "expect , or }";
        jsonValueFree(obj);
        return NULL;
    }
    return obj;
}

/* 解析 JSON 数组: [ value, ... ] */
static JsonValue *parseArray(Cursor *cur) {
    cur->pos++;                                  /* 跳过 '[' */
    JsonValue *arr = jsonNew(JSON_ARRAY);
    int capacity = 8;
    arr->u.array.items = (JsonValue **)malloc((size_t)capacity * sizeof(JsonValue *));
    arr->u.array.count = 0;
    for (;;) {
        skipWhitespace(cur);
        if (*cur->pos == ']') {                  /* 空数组或收尾 */
            cur->pos++;
            break;
        }
        if (arr->u.array.count == capacity) {    /* 扩容 */
            capacity *= 2;
            arr->u.array.items = (JsonValue **)realloc(arr->u.array.items,
                                                       (size_t)capacity * sizeof(JsonValue *));
        }
        JsonValue *value = parseValue(cur);
        if (!value) {
            jsonValueFree(arr);
            return NULL;
        }
        arr->u.array.items[arr->u.array.count++] = value;
        skipWhitespace(cur);
        if (*cur->pos == ',') {
            cur->pos++;
            continue;
        }
        if (*cur->pos == ']') {
            cur->pos++;
            break;
        }
        g_lastError = "expect , or ]";
        jsonValueFree(arr);
        return NULL;
    }
    return arr;
}

/* 解析任意 JSON 值(分发入口) */
static JsonValue *parseValue(Cursor *cur) {
    skipWhitespace(cur);
    char firstChar = *cur->pos;

    if (firstChar == '{')
        return parseObject(cur);
    if (firstChar == '[')
        return parseArray(cur);
    if (firstChar == '"')
        return parseString(cur);

    /* 字面量 */
    if (firstChar == 't') {
        if (!strncmp(cur->pos, "true", 4)) {
            cur->pos += 4;
            JsonValue *booleanValue = jsonNew(JSON_BOOL);
            booleanValue->u.boolValue = 1;
            return booleanValue;
        }
        g_lastError = "bad lit";
        return NULL;
    }
    if (firstChar == 'f') {
        if (!strncmp(cur->pos, "false", 5)) {
            cur->pos += 5;
            JsonValue *booleanValue = jsonNew(JSON_BOOL);
            booleanValue->u.boolValue = 0;
            return booleanValue;
        }
        g_lastError = "bad lit";
        return NULL;
    }
    if (firstChar == 'n') {
        if (!strncmp(cur->pos, "null", 4)) {
            cur->pos += 4;
            return jsonNew(JSON_NULL);
        }
        g_lastError = "bad lit";
        return NULL;
    }

    /* 数值 */
    if (firstChar == '-' || isdigit((unsigned char)firstChar)) {
        char *endPtr = NULL;
        double number = strtod(cur->pos, &endPtr);
        if (endPtr == cur->pos) {
            g_lastError = "bad num";
            return NULL;
        }
        cur->pos = endPtr;
        JsonValue *numberNode = jsonNew(JSON_NUMBER);
        numberNode->u.numberValue = number;
        return numberNode;
    }

    g_lastError = "unexpected char";
    return NULL;
}

/* ---------------- 对外接口: 解析/释放 ---------------- */

JsonValue *jsonParse(const char *text) {
    g_lastError = NULL;
    Cursor cursor;
    cursor.pos = text;
    skipWhitespace(&cursor);
    JsonValue *root = parseValue(&cursor);
    if (!root)
        return NULL;
    skipWhitespace(&cursor);
    if (*cursor.pos) {                           /* 根节点后还有残留内容 */
        jsonValueFree(root);
        g_lastError = "trailing";
        return NULL;
    }
    return root;
}

/* 递归释放一棵 JsonValue 树 */
static void jsonFreeRecursive(JsonValue *value) {
    int entryIndex;
    if (!value)
        return;
    if (value->type == JSON_OBJECT) {
        for (entryIndex = 0; entryIndex < value->u.object.count; entryIndex++) {
            free(value->u.object.keys[entryIndex]);
            jsonFreeRecursive(value->u.object.values[entryIndex]);
        }
        free(value->u.object.keys);
        free(value->u.object.values);
    } else if (value->type == JSON_ARRAY) {
        for (entryIndex = 0; entryIndex < value->u.array.count; entryIndex++)
            jsonFreeRecursive(value->u.array.items[entryIndex]);
        free(value->u.array.items);
    } else if (value->type == JSON_STRING) {
        free(value->u.stringValue);
    }
    free(value);
}

void jsonValueFree(JsonValue *value) {
    jsonFreeRecursive(value);
}

/* ---------------- 对象成员读取 ---------------- */

JsonValue *jsonGetMember(JsonValue *object, const char *key) {
    int memberIndex;
    if (!object || object->type != JSON_OBJECT)
        return NULL;
    for (memberIndex = 0; memberIndex < object->u.object.count; memberIndex++) {
        if (!strcmp(object->u.object.keys[memberIndex], key))
            return object->u.object.values[memberIndex];
    }
    return NULL;
}

double jsonGetNumber(JsonValue *object, const char *key, double defaultValue) {
    JsonValue *member = jsonGetMember(object, key);
    if (!member || member->type != JSON_NUMBER)
        return defaultValue;
    return member->u.numberValue;
}

int jsonGetInt(JsonValue *object, const char *key, int defaultValue) {
    JsonValue *member = jsonGetMember(object, key);
    if (!member)
        return defaultValue;
    if (member->type == JSON_NUMBER)
        return (int)member->u.numberValue;
    if (member->type == JSON_BOOL)              /* 布尔可当作 0/1 读取 */
        return member->u.boolValue ? 1 : 0;
    return defaultValue;
}

const char *jsonGetString(JsonValue *object, const char *key, const char *defaultValue) {
    JsonValue *member = jsonGetMember(object, key);
    if (!member || member->type != JSON_STRING)
        return defaultValue;
    return member->u.stringValue;
}

int jsonArrayLength(JsonValue *array) {
    return (array && array->type == JSON_ARRAY) ? array->u.array.count : 0;
}

/* ============================================================================
 * StringBuffer: 可变字符串缓冲实现
 * ==========================================================================*/
void strbufInit(StringBuffer *sb) {
    sb->length = 0;
    sb->capacity = 256;
    sb->data = (char *)malloc((size_t)sb->capacity);
    sb->data[0] = '\0';
}

void strbufFree(StringBuffer *sb) {
    if (sb->data)
        free(sb->data);
    sb->data = NULL;
    sb->length = sb->capacity = 0;
}

/* 保证还能容纳 extra 字节(不含结尾符) */
static void strbufReserve(StringBuffer *sb, int extraBytes) {
    if (sb->length + extraBytes + 1 > sb->capacity) {
        while (sb->length + extraBytes + 1 > sb->capacity)
            sb->capacity *= 2;
        sb->data = (char *)realloc(sb->data, (size_t)sb->capacity);
    }
}

void strbufAppendChar(StringBuffer *sb, char character) {
    strbufReserve(sb, 1);
    sb->data[sb->length++] = character;
    sb->data[sb->length] = '\0';
}

void strbufAppend(StringBuffer *sb, const char *text) {
    int textLength = (int)strlen(text);
    strbufReserve(sb, textLength);
    memcpy(sb->data + sb->length, text, (size_t)textLength);
    sb->length += textLength;
    sb->data[sb->length] = '\0';
}

void strbufAppendFormat(StringBuffer *sb, const char *fmt, ...) {
    va_list args;
    char temp[512];
    va_start(args, fmt);
    vsnprintf(temp, sizeof(temp), fmt, args);
    va_end(args);
    strbufAppend(sb, temp);
}

/* 追加 JSON 数字: 整数直接输出, 否则取最多 8 位有效数字的紧凑形式 */
void strbufAppendNumber(StringBuffer *sb, double value) {
    if (value != (double)(long long)value && fabs(value) < 1e13) {
        char text[64];
        snprintf(text, sizeof(text), "%.8g", value);
        strbufAppend(sb, text);
    } else {
        char text[64];
        snprintf(text, sizeof(text), "%lld", (long long)value);
        strbufAppend(sb, text);
    }
}

void strbufAppendInt64(StringBuffer *sb, long long value) {
    char text[32];
    snprintf(text, sizeof(text), "%lld", value);
    strbufAppend(sb, text);
}

/* 追加 JSON 字符串字面量(含两端引号与转义) */
void strbufAppendEscapedString(StringBuffer *sb, const char *text) {
    strbufAppendChar(sb, '"');
    for (; *text; text++) {
        unsigned char currentByte = (unsigned char)*text;
        switch (currentByte) {
            case '"':  strbufAppend(sb, "\\\""); break;
            case '\\': strbufAppend(sb, "\\\\"); break;
            case '\n': strbufAppend(sb, "\\n");  break;
            case '\r': strbufAppend(sb, "\\r");  break;
            case '\t': strbufAppend(sb, "\\t");  break;
            default:
                if (currentByte < 0x20)
                    strbufAppendFormat(sb, "\\u%04x", currentByte);
                else
                    strbufAppendChar(sb, (char)currentByte);
        }
    }
    strbufAppendChar(sb, '"');
}
