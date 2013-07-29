#include "EventManager.h"
#include <sstream>
namespace event 
{
    IEventManager* g_pBlobalEventManger = NULL;

    IEventManager* IEventManager::Get(VOID) {
        return g_pBlobalEventManger;
    }

    IEventManager::IEventManager(BOOL setGLobal) {
        if(setGLobal)
        {
            if(g_pBlobalEventManger)
            {
                LOG_CRITICAL_ERROR("Global Eventmanager already set!");
            }
            else 
            {
                g_pBlobalEventManger = this;
            }
        }
    }

    EventManager::EventManager(BOOL setGlobal) : IEventManager(setGlobal), m_activeQueue(0) { 

    }

    BOOL EventManager::VAddEventListener(CONST EventListener& listener, CONST EventType& type) {

        std::list<EventListener>& list = m_eventListenerMap[type];
    
        for(auto it = list.begin(); it != list.end(); ++it)
        {
            if(*it == listener)
            {
                LOG_WARNING("VAddEventListener ignored: Attempting to register an EventListener twice!");
                return FALSE;
            }
        }
        list.push_back(listener);

        return TRUE;
    }

    BOOL EventManager::VRemoveEventListener(CONST EventListener& listener, CONST EventType& type) {

        auto it = m_eventListenerMap.find(type);

        if(it != m_eventListenerMap.end()) 
        {

            for(auto i = it->second.begin(); i != it->second.end(); ++i)
            {
                if(*i == listener) 
                {
                    it->second.erase(i);
                    return TRUE;
                }
            }
        }

        return FALSE;
    }

    BOOL EventManager::VTriggetEvent(CONST IEventPtr& event) {

        auto it = m_eventListenerMap.find(event->VGetEventType());

        if(it == m_eventListenerMap.end())
        {
            return FALSE;
        }

        for(auto i = it->second.begin(); i != it->second.end(); ++i)
        {
            EventListener listener = (*i);
            (*i)(event);
        }

        return TRUE;
    }

    BOOL EventManager::VQueueEventTest(CONST IEventPtr& event) {
        /*
        auto it = m_eventListenerMap.find(event->VGetEventType());

        if(it == m_eventListenerMap.end()) 
        {
    #ifdef _DEBUG
            std::stringstream ss;
            ss << event->VGetEventType();
            LOG_INFO("Failed to queue event, no listener for event: " + ss.str());
    #endif
            return FALSE;
        }

        m_eventQueues[m_activeQueue].push_back(event); */


        return TRUE;
    }

    BOOL EventManager::VQueueEvent(CONST IEventPtr& event) {

        auto it = m_eventListenerMap.find(event->VGetEventType());

        if(it == m_eventListenerMap.end()) 
        {
    #ifdef _DEBUG
            std::stringstream ss;
            ss << event->VGetEventType();
            LOG_INFO_A("Failed to queue event, no listener for event: %s", ss.str());
    #endif
            return FALSE;
        }

        //LOG_INFO("Event queued: "+std::string(event->VGetName()));

        m_eventQueues[m_activeQueue].push_back(event);


        return TRUE;
    }

    BOOL EventManager::VQueueEventThreadSave(CONST IEventPtr& event) {

        auto it = m_eventListenerMap.find(event->VGetEventType());

        if(it == m_eventListenerMap.end()) 
        {
            return FALSE;
        }

        m_threadSaveQueue.push(event);

        return TRUE;
    }

    BOOL EventManager::VAbortEvent(CONST EventType& type, BOOL all) {

        auto it = m_eventListenerMap.find(type);

        if(it == m_eventListenerMap.end()) 
        {
            return FALSE;
        }
        BOOL aborted = FALSE;

        if(all)
        {
            std::list<IEventPtr>& queue = m_eventQueues[m_activeQueue];
            for(auto i = queue.begin(); i != queue.end(); ++i)
            {
                if((*i)->VGetEventType() == type)
                {
                    i = queue.erase(i);
                    aborted = TRUE;
                }
            }
        }
        else
        {
            std::list<IEventPtr>& queue = m_eventQueues[m_activeQueue];
            for(auto i = queue.begin(); i != queue.end(); ++i)
            {
                if((*i)->VGetEventType() == type)
                {
                    i = queue.erase(i);
                    aborted = TRUE;
                    break;
                }
            }
        }

        return aborted;
    }

    BOOL EventManager::VUpdate(CONST ULONG maxMillis) {

        ULONG current = GetTickCount();
        ULONG max = maxMillis == -1 ? -1 : current += maxMillis;
        INT queue = m_activeQueue;
        m_activeQueue = (m_activeQueue + 1) % 2;
        m_eventQueues[m_activeQueue].clear();

#ifdef _DEBUG 
        m_lastEventsFired = 0;
#endif
        while(!m_eventQueues[queue].empty())
        {
            IEventPtr e = m_eventQueues[queue].front();

            m_eventQueues[queue].pop_front();

            auto it = m_eventListenerMap.find(e->VGetEventType());

            for(auto i = it->second.begin(); i != it->second.end(); ++i)
            {
                (*i)(e);
#ifdef _DEBUG
                m_lastEventsFired++;
#endif
            }

            if(maxMillis != -1 && GetTickCount() >= max)
            {
                LOG_CRITICAL_ERROR("Eventmanager had to abort eventstream update");
                break;
            }
        }

        if(!m_eventQueues[queue].empty())
        {
            while(!m_eventQueues[queue].empty())
            {
                IEventPtr e = m_eventQueues[queue].back();
                m_eventQueues[queue].pop_back();
                m_eventQueues[m_activeQueue].push_front(e);

            }
            return FALSE;
        }
    
        while(!m_threadSaveQueue.empty())
        {
            IEventPtr ptr = m_threadSaveQueue.pop();

            VQueueEvent(ptr);

            if(maxMillis != -1 && GetTickCount() > max)
            {
                LOG_CRITICAL_ERROR("ThreadSaveQueue took too long");
            }
        }

        return TRUE;
    }

    EventManager::~EventManager(VOID) {
        if(g_pBlobalEventManger == this)
        {
            g_pBlobalEventManger = NULL;
        }
    }
};
