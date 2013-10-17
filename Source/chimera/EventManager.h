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
        UINT m_lastEventsFired;

    public:
        EventManager(VOID);

        BOOL VAddEventListener(CONST EventListener& listener, CONST EventType& type);

        BOOL VRemoveEventListener(CONST EventListener& listener, CONST EventType& type);

        BOOL VQueueEvent(CONST IEventPtr& event);

        BOOL VQueueEventTest(CONST IEventPtr& event);

        BOOL VQueueEventThreadSave(CONST IEventPtr& event);

        BOOL VTriggetEvent(CONST IEventPtr& event);

        BOOL VAbortEvent(CONST EventType& type, BOOL all = FALSE);

        BOOL VUpdate(CONST ULONG maxMillis = -1);

        UINT LastEventsFired(VOID) { return m_lastEventsFired; }

        ~EventManager(VOID);
    };
};