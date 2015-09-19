#pragma once
//#include "stdafx.h"
#include <windows.h>
#ifdef _DEBUG
#include <sstream>
#endif

typedef unsigned long ulong;
typedef unsigned int uint;

#include <string>

typedef void (*WriteLogCallBack) (std::string);

namespace Logger 
{
    class LogManager 
    {
    private:
        WriteLogCallBack m_writeCallBack;
        CRITICAL_SECTION m_critSection; 
        void GetString(std::string& dest, const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line);

    public:
        LogManager(void) 
        {
            InitializeCriticalSection(&m_critSection);
            SetWriteCallBack(NULL);
        }

        void Log(const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line);
        void CriticalError(const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line);
        void WriteTo(std::string& message);
        void Destroy(void);

        void SetWriteCallBack(WriteLogCallBack cb)
        {
            m_writeCallBack = cb;
        }

        ~LogManager(void) 
        {
            DeleteCriticalSection(&m_critSection);
        }
    };

    extern LogManager* s_pLogMgr;

    extern std::string* s_fileName;
    
    void Init(std::string fileName);

    void Destroy(void);

    void Log(const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line);
}

//#ifdef _DEBUG

#define _GET_TEXT(format, ...) \
    char __tmp[2048]; \
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

/*#else
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
#endif*/
