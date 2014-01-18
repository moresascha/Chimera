#include "Logger.h"
#include "stdafx.h"
#include <time.h>
namespace Logger 
{
    LogManager* s_pLogMgr = NULL;
    std::string* s_fileName = NULL;

    void Init(std::string fileName) 
    {
        if(!s_pLogMgr)
        {
            s_pLogMgr = new LogManager();
            s_fileName = new std::string(fileName);
            FILE* file = NULL;
            fopen_s(&file, fileName.c_str(), "w");
            if(file)
            {
                char date[9];
                _strdate_s(date);
                std::string d(fileName);
                d += ", ";
                d += date;
                d += "\n";
                fprintf(file, d.c_str());
                fclose(file);
            }
        }
    }

    void Destroy(void) 
    {
        SAFE_DELETE(s_pLogMgr);
        SAFE_DELETE(s_fileName);
    }

    void Log(const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line) 
    {
        s_pLogMgr->Log(tag, message, funcName, file, line);
    }

    void Error(const std::string& message, const char* funcName, const char* file, const uint line) 
    {
        s_pLogMgr->Log("Error", message, funcName, file, line);
    }

    void LogManager::WriteTo(std::string& message) 
    {
        EnterCriticalSection(&m_critSection);
        if(m_writeCallBack)
        {
            m_writeCallBack(message);
        }
        else
        {
            FILE* file = NULL;
            fopen_s(&file, s_fileName->c_str(), "a+");
            if(!file)
            {
                return;
            }
            fprintf(file, message.c_str());
            fclose(file);
        }
        LeaveCriticalSection(&this->m_critSection);
    }

    void LogManager::Log(const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line) 
    {
        std::string data;
        GetString(data, tag, message, funcName, file, line);
        WriteTo(data);
    }

    void LogManager::CriticalError(const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line) 
    {
        std::string data;
        GetString(data, tag, message, funcName, file, line);
        int id = MessageBoxA(NULL, data.c_str(), tag.c_str(), MB_ABORTRETRYIGNORE|MB_ICONERROR|MB_DEFBUTTON3);
        if(IDABORT == id)
        {
            DebugBreak();
        }
        else if(IDIGNORE == id)
        {
            return;
        }
        else
        {
            exit(-1);
        }
    }

    void LogManager::GetString(std::string& dest, const std::string& tag, const std::string& message, const char* funcName, const char* file, const uint line) 
    {
        std::string t = "["+tag+"] ";
        std::string inLine = "\n";
        for(uint i = 0; i < t.length(); ++i)
        {
            inLine += " ";
        }
        dest += t + message;
        std::string fun = funcName;
        dest += inLine + "Function: " + fun;
        std::string f = file;
        dest += inLine  + "File: " + f;
        char buffer[11];
        _itoa_s(line, buffer, 10);
        std::string l = buffer;
        dest += inLine + "Line: " + l;
        dest += "\n";
    }
};
