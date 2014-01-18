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
        uint m_systemCores;
        ProcessManager* m_manager;
        std::shared_ptr<IProcess>* m_currentRunning;
        uint m_currentRunningSize;

        int CreateFreeSlot(void);
        
        void FreeSlot(UCHAR slot);

        bool HasFreeSlot(void);

    public:
        Scheduler(uint cores, ProcessManager* manager);

        IProcess* VAddProcess(std::shared_ptr<IProcess> proc);

        void VThreadProc(void);

        ~Scheduler(void);
    };
}

