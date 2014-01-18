#pragma once
#include "CMTypes.h"

namespace chimera
{
    class IEvent
    {
    public:
        IEvent(void) {}
        virtual EventType VGetEventType(void) const = 0;
        virtual const char* VGetName(void) const = 0;
        virtual ~IEvent(void) {}
    };

    class IEventManager
    {
    public:
        virtual bool VAddEventListener(const EventListener& listener, const EventType& type) = 0;

        virtual bool VRemoveEventListener(const EventListener& listener, const EventType& type) = 0;

        virtual bool VQueueEvent(const IEventPtr& event) = 0;

        virtual bool VQueueEventTest(const IEventPtr& event) = 0;

        virtual bool VQueueEventThreadSave(const IEventPtr& event) = 0;

        virtual bool VTriggetEvent(const IEventPtr& event) = 0;

        virtual bool VAbortEvent(const EventType& type, bool all = false) = 0;

        virtual bool VUpdate(const ulong maxMillis = (ulong)-1) = 0;

        virtual ~IEventManager(void) {}
    };

    class IEventFactory
    {
    public:
        virtual IEventManager* VCreateEventManager(void) = 0;
    };
}