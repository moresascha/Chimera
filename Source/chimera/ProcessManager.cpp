#include "ProcessManager.h"

namespace chimera 
{
    ProcessManager::ProcessManager(VOID)
    {
        //TODO move to app
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        DWORD numCPU = sysinfo.dwNumberOfProcessors;

        m_pScheduler = new Scheduler(numCPU, this);

        m_pScheduler->SetState(eProcessState_Running);
        m_pScheduler->VOnInit();
    }

    IProcess* ProcessManager::VAttach(std::shared_ptr<IProcess> process) 
    {
        m_processes.push_back(process);
        return m_processes.back().get();
    }

    IProcess* ProcessManager::VAttachWithScheduler(std::shared_ptr<IProcess> process) 
    {
        if(process->VGetType() == eProcessType_Realtime)
        {

            return m_pScheduler->VAddProcess(process);
        }
        else
        {
            m_processes.push_back(std::move(process));
            return m_processes.back().get();
        }
    }

    VOID ProcessManager::SchedulerAdd(std::shared_ptr<IProcess> process)
    {
        m_lock.Lock();
        m_processes.push_back(process);
        m_lock.Unlock();
    }

    UINT ProcessManager::VUpdate(ULONG deltaMillis) 
    {
        m_lock.Lock();
    
        for(auto it = m_processes.begin(); it != m_processes.end();)
        {
            IProcess* proc = it->get();

            if(proc->GetState() == eProcessState_Uninitialized)
            {
                proc->SetState(eProcessState_Running);
                proc->VOnInit();
            }

            if(proc->GetState() == eProcessState_Running)
            {
                proc->VOnUpdate(deltaMillis);
            }

            if(proc->IsDead())
            {
                switch(proc->GetState())
                {
                case eProcessState_Succed: 
                    {
                        proc->VOnSuccess();
         
                        std::shared_ptr<IProcess> child = std::move(proc->RemoveChild());
                        if(child) 
                        {
                            VAttach(child);
                        }
                    }  break;

                case eProcessState_Aborted: proc->VOnAbort(); break;
                case eProcessState_Failed: proc->VOnFail(); break;  
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

    VOID ProcessManager::VAbortAll(BOOL immediate) 
    {
        for(auto it = m_processes.begin(); it != m_processes.end(); ++it)
        {
            if(immediate)
            {
                (*it)->SetState(chimera::eProcessState_Aborted);
                (*it)->VOnAbort();
            }
            else
            {
                (*it)->SetState(chimera::eProcessState_Aborted);
            }
        }
    }

    ProcessManager::~ProcessManager(VOID)
    {
        TBD_FOR(m_processes)
        {
            it->reset();
        }
        m_processes.clear();
        SAFE_DELETE(m_pScheduler);
    }
};
