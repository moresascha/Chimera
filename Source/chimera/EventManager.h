#pragma once
#include "stdafx.h"
#include "Locker.h"
#include "Event.h"
#include "ts_queue.h"

namespace event 
{
    class IEventManager
    {
    public:
        IEventManager(BOOL setGlobal);

        virtual BOOL VAddEventListener(CONST EventListener& listener, CONST EventType& type) = 0;

        virtual BOOL VRemoveEventListener(CONST EventListener& listener, CONST EventType& type) = 0;

        virtual BOOL VQueueEvent(CONST IEventPtr& event) = 0;

        virtual BOOL VQueueEventTest(CONST IEventPtr& event) = 0;

        virtual BOOL VQueueEventThreadSave(CONST IEventPtr& event) = 0;

        virtual BOOL VTriggetEvent(CONST IEventPtr& event) = 0;

        virtual BOOL VAbortEvent(CONST EventType& type, BOOL all = FALSE) = 0;

        virtual BOOL VUpdate(CONST ULONG maxMillis = -1) = 0;

        static IEventManager* Get(VOID);

        virtual ~IEventManager(VOID) {}
    };

    class EventManager : public IEventManager 
    {
    
    private:
        BYTE m_activeQueue;
        std::map<EventType, std::list<EventListener>> m_eventListenerMap;
        std::list<IEventPtr> m_eventQueues[2];
        util::ts_queue<IEventPtr> m_threadSaveQueue;
        UINT m_lastEventsFired;

    public:
        EventManager(BOOL setGlobal);

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

    extern IEventManager* g_pBlobalEventManger;

#define ADD_EVENT_LISTENER(_this, function, type) \
    { \
    event::EventListener listener = fastdelegate::MakeDelegate(_this, function); \
    event::IEventManager::Get()->VAddEventListener(listener, type); \
    }

#define ADD_EVENT_LISTENER_STATIC(function, type) \
    { \
    event::EventListener listener = function; \
    event::IEventManager::Get()->VAddEventListener(listener, type); \
    }

#define REMOVE_EVENT_LISTENER(_this, function, type) \
    { \
    event::EventListener listener = fastdelegate::MakeDelegate(_this, function); \
    event::IEventManager::Get()->VRemoveEventListener(listener, type); \
    }

#define QUEUE_EVENT(_eventPtr) \
    { \
        event::IEventPtr event(_eventPtr); \
        event::IEventManager::Get()->VQueueEvent(event); \
    }
};