#pragma once
//#include "stdafx.h"
#include <windows.h>
#ifdef _DEBUG
#include <sstream>
#endif

#include <string>

typedef VOID (*WriteLogCallBack) (std::string);

namespace Logger 
{
    class LogManager 
    {
    private:
        WriteLogCallBack m_writeCallBack;
        CRITICAL_SECTION m_critSection; 
        VOID GetString(std::string& dest, CONST std::string& tag, CONST std::string& message, CONST CHAR* funcName, CONST CHAR* file, CONST UINT line);

    public:
        LogManager(VOID) 
        {
            InitializeCriticalSection(&m_critSection);
            SetWriteCallBack(NULL);
        }

        VOID Log(CONST std::string& tag, CONST std::string& message, CONST CHAR* funcName, CONST CHAR* file, CONST UINT line);
        VOID CriticalError(CONST std::string& tag, CONST std::string& message, CONST CHAR* funcName, CONST CHAR* file, CONST UINT line);
        VOID WriteTo(std::string& message);
        VOID Destroy(VOID);

        VOID SetWriteCallBack(WriteLogCallBack cb)
        {
            m_writeCallBack = cb;
        }

        ~LogManager(VOID) 
        {
            DeleteCriticalSection(&m_critSection);
        }
    };

    extern LogManager* s_pLogMgr;

    extern std::string* s_fileName;
    
    VOID Init(std::string fileName);

    VOID Destroy(VOID);

    VOID Log(CONST std::string& tag, CONST std::string& message, CONST CHAR* funcName, CONST CHAR* file, CONST UINT line);
}

#ifdef _DEBUG

#define _GET_TEXT(format, ...) \
    CHAR __tmp[2048]; \
    sprintf_s(__tmp, 2048, format, __VA_ARGS__); \
    std::string __msg__(__tmp);

#define LOG_CRITICAL_ERROR_A(cstr, ...) \
do \
{ \
    _GET_TEXT(cstr, __VA_ARGS__) \
    CmCriticalError("ERROR", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define LOG_ERROR_A(cstr, ...) \
    do \
{ \
    _GET_TEXT(cstr, __VA_ARGS__) \
    CmLog("ERROR", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define LOG_ERROR_NR_A(cstr, ...) { LOG_ERROR_A(cstr, __VA_ARGS__); return; }


#define LOG_WARNING_A(cstr, ...) \
do \
{ \
    _GET_TEXT(cstr, __VA_ARGS__) \
    CmLog("WARNING", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define LOG_INFO_A(cstr, ...) \
do \
{ \
    _GET_TEXT(cstr, __VA_ARGS__) \
    CmLog("INFO", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define DEBUG_OUT_A(cstr, ...) \
    do \
{ \
    _GET_TEXT(cstr, __VA_ARGS__) \
    OutputDebugStringA(__msg__.c_str()); \
} \
while(0)

//-----------------------------
#define LOG_CRITICAL_ERROR(cstr) \
    do \
{ \
    _GET_TEXT(cstr, "") \
    CmCriticalError("ERROR", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define LOG_ERROR(cstr) \
    do \
{ \
    _GET_TEXT(cstr) \
    CmLog("ERROR", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define LOG_ERROR_NR(cstr) { LOG_ERROR(cstr); return; }


#define LOG_WARNING(cstr) \
    do \
{ \
    _GET_TEXT(cstr, "") \
    CmLog("WARNING", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define LOG_INFO(cstr) \
    do \
{ \
    _GET_TEXT(cstr, "") \
    CmLog("INFO", __msg__, __FUNCTION__, __FILE__, __LINE__); \
} \
while(0)

#define DEBUG_OUT(cstr) \
    do \
{ \
    _GET_TEXT(cstr, "") \
    OutputDebugStringA(__msg__.c_str()); \
} \
while(0)

#else
#define LOG_CRITICAL_ERROR_A(message, ...) ((VOID)0)
#define LOG_WARNING_A(message, ...) ((VOID)0)
#define LOG_INFO_A(message, ...) ((VOID)0)
#define LOG_ERROR_A(message, ...) ((VOID)0)
#define DEBUG_OUT_A(message, ...) ((VOID)0)
#define LOG_ERROR_NR_A(message, ...) ((VOID)0)

#define LOG_CRITICAL_ERROR(message) ((VOID)0)
#define LOG_WARNING(message) ((VOID)0)
#define LOG_INFO(message) ((VOID)0)
#define LOG_ERROR(message) ((VOID)0)
#define DEBUG_OUT(message) ((VOID)0)
#define LOG_ERROR_NR(message) ((VOID)0)
#endif
