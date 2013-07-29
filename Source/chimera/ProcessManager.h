#pragma once
#include "stdafx.h"
#include "Process.h"
#include "Scheduler.h"

namespace proc {
class ProcessManager
{
    friend class tbd::Scheduler;
private:
    std::list<std::shared_ptr<Process>> m_processes;
    tbd::Scheduler* m_pScheduler;
    util::Locker m_lock;

protected:
    VOID SchedulerAdd(std::shared_ptr<Process> process);

public:
    ProcessManager(VOID);

    UINT Update(ULONG delatMillis);

    std::weak_ptr<Process> Attach(std::shared_ptr<Process> process);

    std::weak_ptr<Process> AttachWithScheduler(std::shared_ptr<Process> process);

    VOID AbortAll(BOOL immediate);

    UINT GetProcessCount(VOID) CONST { return (UINT)m_processes.size(); }

    ~ProcessManager(VOID);
};

};

