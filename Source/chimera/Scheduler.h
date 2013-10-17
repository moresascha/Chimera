#pragma once
#include "stdafx.h"
#include "ts_queue.h"
#include "Process.h"

namespace chimera
{
    class Scheduler : public RealtimeProcess, public IScheduler
    {
    private:
        HANDLE m_event;
        util::Locker m_locker;
        std::queue<std::shared_ptr<IProcess>> m_processQueue;
        UINT m_systemCores;
        ProcessManager* m_manager;
        std::shared_ptr<IProcess>* m_currentRunning;
        UINT m_currentRunningSize;

        INT CreateFreeSlot(VOID);
        
        VOID FreeSlot(UCHAR slot);

        BOOL HasFreeSlot(VOID);

    public:
        Scheduler(UINT cores, ProcessManager* manager);

        IProcess* VAddProcess(std::shared_ptr<IProcess> proc);

        VOID VThreadProc(VOID);

        ~Scheduler(VOID);
    };
}

