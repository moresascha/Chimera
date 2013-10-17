#pragma once
#include "stdafx.h"
#include "Process.h"
#include "Scheduler.h"

namespace chimera 
{
    class ProcessManager : public IProcessManager
    {
        friend class chimera::Scheduler;
    private:
        std::list<std::shared_ptr<IProcess>> m_processes;
        Scheduler* m_pScheduler;
        util::Locker m_lock;

    protected:
        VOID SchedulerAdd(std::shared_ptr<IProcess> process);

    public:
        ProcessManager(VOID);

        UINT VUpdate(ULONG delatMillis);

        IProcess* VAttach(std::shared_ptr<IProcess> process);

        IProcess* VAttachWithScheduler(std::shared_ptr<IProcess> process);

        VOID VAbortAll(BOOL immediate);

        UINT VGetProcessCount(VOID) CONST { return (UINT)m_processes.size(); }

        ~ProcessManager(VOID);
    };
};

