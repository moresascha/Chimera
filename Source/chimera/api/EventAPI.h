#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IEvent
    {
    public:
        IEvent(VOID) {}
        virtual EventType VGetEventType(VOID) CONST = 0;
        virtual CONST CHAR* VGetName(VOID) CONST = 0;
        virtual ~IEvent(VOID) {}
    };

    class IEventManager
    {
    public:
        virtual BOOL VAddEventListener(CONST EventListener& listener, CONST EventType& type) = 0;

        virtual BOOL VRemoveEventListener(CONST EventListener& listener, CONST EventType& type) = 0;

        virtual BOOL VQueueEvent(CONST IEventPtr& event) = 0;

        virtual BOOL VQueueEventTest(CONST IEventPtr& event) = 0;

        virtual BOOL VQueueEventThreadSave(CONST IEventPtr& event) = 0;

        virtual BOOL VTriggetEvent(CONST IEventPtr& event) = 0;

        virtual BOOL VAbortEvent(CONST EventType& type, BOOL all = FALSE) = 0;

        virtual BOOL VUpdate(CONST ULONG maxMillis = -1) = 0;

        virtual ~IEventManager(VOID) {}
    };

    class IEventFactory
    {
    public:
        virtual IEventManager* VCreateEventManager(VOID) = 0;
    };
}