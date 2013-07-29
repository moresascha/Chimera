#pragma once
#include "stdafx.h"
namespace util {
class DLL_EXPORT Thread
{
protected:
    virtual VOID OnDone(VOID);
private:
    VOID(*m_pfCallBack)(LPVOID);
    DWORD (*m_fpThreadFunc)(LPVOID param);
    LPVOID m_param;
    static DWORD WINAPI Run(Thread* thread);
public:
    Thread(LPVOID param, DWORD (*threadFunc)(LPVOID param));
    VOID Start(VOID);
    inline VOID SetOnDone(VOID(*CallBack)(LPVOID param)) {
        this->m_pfCallBack = CallBack;
    }
    ~Thread(VOID);
};

}

