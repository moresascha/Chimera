#include "EventManager.h"
#include <sstream>
namespace chimera 
{

    EventManager::EventManager(void) : m_activeQueue(0) 
    { 

    }

    bool EventManager::VAddEventListener(const EventListener& listener, const EventType& type) {

        std::list<EventListener>& list = m_eventListenerMap[type];
    
        for(auto it = list.begin(); it != list.end(); ++it)
        {
            if(*it == listener)
            {
                LOG_WARNING("VAddEventListener ignored: Attempting to register an EventListener twice!");
                return false;
            }
        }
        list.push_back(listener);

        return true;
    }

    bool EventManager::VRemoveEventListener(const EventListener& listener, const EventType& type) {

        auto it = m_eventListenerMap.find(type);

        if(it != m_eventListenerMap.end()) 
        {

            for(auto i = it->second.begin(); i != it->second.end(); ++i)
            {
                if(*i == listener) 
                {
                    it->second.erase(i);
                    return true;
                }
            }
        }

        return false;
    }

    bool EventManager::VTriggetEvent(const IEventPtr& chimera) {

        auto it = m_eventListenerMap.find(chimera->VGetEventType());

        if(it == m_eventListenerMap.end())
        {
            return false;
        }

        for(auto i = it->second.begin(); i != it->second.end(); ++i)
        {
            EventListener listener = (*i);
            (*i)(chimera);
        }

        return true;
    }

    bool EventManager::VQueueEventTest(const IEventPtr& chimera) {
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


        return true;
    }

    bool EventManager::VQueueEvent(const IEventPtr& chimera) {

        auto it = m_eventListenerMap.find(chimera->VGetEventType());

        if(it == m_eventListenerMap.end()) 
        {
    #ifdef _DEBUG
            std::stringstream ss;
            ss << chimera->VGetEventType();
            LOG_INFO_A("Failed to queue event, no listener for event: %s", ss.str());
    #endif
            return false;
        }

        //LOG_INFO("Event queued: "+std::string(event->VGetName()));

        m_eventQueues[m_activeQueue].push_back(chimera);


        return true;
    }

    bool EventManager::VQueueEventThreadSave(const IEventPtr& chimera) {

        auto it = m_eventListenerMap.find(chimera->VGetEventType());

        if(it == m_eventListenerMap.end()) 
        {
            return false;
        }

        m_threadSaveQueue.push(chimera);

        return true;
    }

    bool EventManager::VAbortEvent(const EventType& type, bool all) {

        auto it = m_eventListenerMap.find(type);

        if(it == m_eventListenerMap.end()) 
        {
            return false;
        }
        bool aborted = false;

        if(all)
        {
            std::list<IEventPtr>& queue = m_eventQueues[m_activeQueue];
            for(auto i = queue.begin(); i != queue.end(); ++i)
            {
                if((*i)->VGetEventType() == type)
                {
                    i = queue.erase(i);
                    aborted = true;
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
                    aborted = true;
                    break;
                }
            }
        }

        return aborted;
    }

    bool EventManager::VUpdate(const ulong maxMillis) {

        ulong current = GetTickCount();
        ulong max = maxMillis == -1 ? -1 : current += maxMillis;
        int queue = m_activeQueue;
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
            return false;
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

        return true;
    }

    EventManager::~EventManager(void) 
    {
    }
};
