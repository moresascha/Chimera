#include "Scheduler.h"
#include "ProcessManager.h"
namespace chimera
{
    /*std::unique_ptr<IProcess> operator=(CONST std::unique_ptr<IProcess>& p1)
    {
        return std::move(p1);
    }*/

    Scheduler::Scheduler(UINT cores, ProcessManager* manager) : m_systemCores(cores), m_manager(manager), m_currentRunningSize(0), m_event(NULL)
    {
        m_currentRunning = new std::shared_ptr<IProcess>[m_systemCores];
        for(UCHAR i = 0; i < m_systemCores; ++i)
        {
            m_currentRunning[i] = NULL;
        }
        m_event = CreateEvent(NULL, FALSE, FALSE, NULL);
    }

    INT Scheduler::CreateFreeSlot(VOID)
    {
        for(UCHAR i = 0; i < m_systemCores; ++i)
        {
            if(m_currentRunning[i] == NULL)
            {
                m_currentRunningSize++;
                return i;
            }
        }
        return -1;
    }

    BOOL Scheduler::HasFreeSlot(VOID)
    {
        return m_currentRunningSize < m_systemCores;
    }

    VOID Scheduler::FreeSlot(UCHAR slot)
    {
        m_currentRunningSize--;
        m_currentRunning[slot] = NULL;
    }

    VOID Scheduler::VThreadProc(VOID)
    {
        while(CmGetApp()->VIsRunning())
        {
            while(!m_processQueue.empty() && HasFreeSlot())
            {
                if(HasFreeSlot())
                {
                    INT slot = CreateFreeSlot();
                    if(slot == -1)
                    {
                        LOG_CRITICAL_ERROR("slot should not be -1");
                    }
                    m_locker.Lock();
                    std::shared_ptr<IProcess> proc = m_processQueue.front();
                    m_processQueue.pop();
                    m_locker.Unlock();
                    m_currentRunning[slot] = proc;
                    m_manager->SchedulerAdd(proc);
                }
            }

            for(UCHAR i = 0; i < m_systemCores; ++i)
            {
                std::shared_ptr<IProcess>& proc = m_currentRunning[i];
                if(proc != NULL)
                {
                    if(proc->IsDead())
                    {
                        FreeSlot(i);
                    }
                }
            }

            if(0 == m_currentRunningSize && m_processQueue.empty())
            {
               // m_locker.Unlock();
                WaitForSingleObject(m_event, INFINITE);
                //DEBUG_OUT("running...");
            }
            //m_locker.Unlock();
            //Sleep(16); //TODO
        }
    }

    IProcess* Scheduler::VAddProcess(std::shared_ptr<IProcess> proc)
    {
        IProcess* raw = proc.get();
        m_locker.Lock();
        m_processQueue.push(proc);
        m_locker.Unlock();

      //  m_locker.Lock();
        SetEvent(m_event);
      //  m_locker.Unlock();

        return raw;
    }

    Scheduler::~Scheduler(VOID)
    {
        if(m_event)
        {
            CloseHandle(m_event);
        }
        SAFE_ARRAY_DELETE(m_currentRunning);
    }
}

