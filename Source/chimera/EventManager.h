#pragma once
#include "stdafx.h"
#include "Locker.h"
#include "Event.h"
#include "ts_queue.h"

namespace chimera 
{
    class EventManager : public IEventManager 
    {
    
    private:
        BYTE m_activeQueue;
        std::map<EventType, std::list<EventListener>> m_eventListenerMap;
        std::list<IEventPtr> m_eventQueues[2];
        util::ts_queue<IEventPtr> m_threadSaveQueue;
        uint m_lastEventsFired;

    public:
        EventManager(void);

        bool VAddEventListener(const EventListener& listener, const EventType& type);

        bool VRemoveEventListener(const EventListener& listener, const EventType& type);

        bool VQueueEvent(const IEventPtr& event);

        bool VQueueEventTest(const IEventPtr& event);

        bool VQueueEventThreadSave(const IEventPtr& event);

        bool VTriggetEvent(const IEventPtr& event);

        bool VAbortEvent(const EventType& type, bool all = false);

        bool VUpdate(const ulong maxMillis = -1);

        uint LastEventsFired(void) { return m_lastEventsFired; }

        ~EventManager(void);
    };
};