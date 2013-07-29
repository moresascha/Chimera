#include "ProcessManager.h"

namespace proc 
{
    ProcessManager::ProcessManager(VOID)
    {
        //TODO move to app
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        DWORD numCPU = sysinfo.dwNumberOfProcessors;

        m_pScheduler = new tbd::Scheduler(numCPU, this);

        m_pScheduler->SetState(RUNNING);
        m_pScheduler->VOnInit();
    }

    std::weak_ptr<Process> ProcessManager::Attach(std::shared_ptr<Process> process) 
    {
        this->m_processes.push_back(process);
        return std::weak_ptr<Process>(process);
    }

    std::weak_ptr<Process> ProcessManager::AttachWithScheduler(std::shared_ptr<Process> process) 
    {
        if(process->VGetType() == REALTIME)
        {
            m_pScheduler->AddProcess(std::static_pointer_cast<proc::RealtimeProcess>(process));
        }
        else
        {
            this->m_processes.push_back(process);
        }
        return std::weak_ptr<Process>(process);
    }

    VOID ProcessManager::SchedulerAdd(std::shared_ptr<Process> process)
    {
        m_lock.Lock();
        this->m_processes.push_back(process);
        m_lock.Unlock();
    }

    UINT ProcessManager::Update(ULONG deltaMillis) {
    
        m_lock.Lock();
    
        for(auto it = m_processes.begin(); it != m_processes.end();)
        {
            std::shared_ptr<Process> proc = (*it);

            if(proc->GetState() == UNINITIALIZED)
            {
                proc->SetState(RUNNING);
                proc->VOnInit();
            }

            if(proc->GetState() == RUNNING)
            {
                proc->VOnUpdate(deltaMillis);
            }

            if(proc->IsDead())
            {
                switch(proc->GetState())
                {
                case SUCCEEDED: 
                    {
                        proc->VOnSuccess();
         
                        std::shared_ptr<Process> child = proc->RemoveChild();
                        if(child) 
                        {
                            Attach(child);
                        }
                    }  break;

                case ABORTED: proc->VOnAbort(); break;
                case FAILED: proc->VOnFail(); break;  
                }
                it = m_processes.erase(it);
            }
            else
            {
                ++it;
            }
        }
        m_lock.Unlock();
        return 0; // TODO
    }

    VOID ProcessManager::AbortAll(BOOL immediate) {
        for(auto it = m_processes.begin(); it != m_processes.end(); ++it)
        {
            if(immediate)
            {
                (*it)->SetState(proc::ABORTED);
                (*it)->VOnAbort();
            }
            else
            {
                (*it)->SetState(proc::ABORTED);
            }
        }
    }

    ProcessManager::~ProcessManager(VOID)
    {
        SAFE_DELETE(m_pScheduler);
        this->m_processes.clear();
    }
};
