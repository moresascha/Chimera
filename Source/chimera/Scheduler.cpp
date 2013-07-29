#include "Scheduler.h"
#include "ProcessManager.h"
#include "GameApp.h"
namespace tbd
{

    Scheduler::Scheduler(UINT cores, proc::ProcessManager* manager) : m_systemCores(cores), m_manager(manager), m_currentRunningSize(0), m_event(NULL)
    {
        m_currentRunning = new std::shared_ptr<proc::RealtimeProcess>[m_systemCores];
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
        while(app::g_pApp->IsRunning())
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
                    std::shared_ptr<proc::RealtimeProcess>& proc =  m_processQueue.pop();
                    m_currentRunning[slot] = proc;
                    m_manager->SchedulerAdd(proc);
                }
            }
            for(UCHAR i = 0; i < m_systemCores; ++i)
            {
                std::shared_ptr<proc::RealtimeProcess>& proc = m_currentRunning[i];
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

    VOID Scheduler::AddProcess(std::shared_ptr<proc::RealtimeProcess> proc)
    {
        m_processQueue.push(proc);

      //  m_locker.Lock();
        SetEvent(m_event);
      //  m_locker.Unlock();
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

