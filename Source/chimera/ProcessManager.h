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
        void SchedulerAdd(std::shared_ptr<IProcess> process);

    public:
        ProcessManager(void);

        uint VUpdate(ulong delatMillis);

        IProcess* VAttach(std::shared_ptr<IProcess> process);

        IProcess* VAttachWithScheduler(std::shared_ptr<IProcess> process);

        void VAbortAll(bool immediate);

        uint VGetProcessCount(void) const { return (uint)m_processes.size(); }

        ~ProcessManager(void);
    };
};

