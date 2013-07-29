#pragma once
#include "stdafx.h"
#include "ts_queue.h"
#include "Process.h"
namespace proc
{
    class ProcessManager;
    class RealtimeProcess;
}
namespace tbd
{
    class Scheduler : public proc::RealtimeProcess
    {
    private:
        HANDLE m_event;
        util::Locker m_locker;
        util::ts_queue<std::shared_ptr<proc::RealtimeProcess>> m_processQueue;
        UINT m_systemCores;
        proc::ProcessManager* m_manager;
        std::shared_ptr<proc::RealtimeProcess>* m_currentRunning;
        UINT m_currentRunningSize;

        INT CreateFreeSlot(VOID);
        
        VOID FreeSlot(UCHAR slot);

    public:
        Scheduler(UINT cores, proc::ProcessManager* manager);

        VOID AddProcess(std::shared_ptr<proc::RealtimeProcess> proc);

        VOID VThreadProc(VOID);

        BOOL HasFreeSlot(VOID);

        ~Scheduler(VOID);
    };
}

