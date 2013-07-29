#pragma once
#include "stdafx.h"
#include "Actor.h"
#include "Vec3.h"

typedef ULONG EventType;

namespace event 
{

    class IEvent;
    typedef std::shared_ptr<IEvent> IEventPtr;
    typedef fastdelegate::FastDelegate1<IEventPtr> EventListener;

    class IEvent
    {
    public:
        IEvent(VOID) {}
        virtual EventType VGetEventType(VOID) = 0;
        virtual ULONG VGetTimeStamp(VOID) = 0;
        virtual CONST CHAR* VGetName(VOID) = 0;
        virtual ~IEvent(VOID) {}
    };

    class Event : public IEvent 
    {
    private:
        ULONG m_timeStamp;

    public:
        Event(ULONG timeStamp = 0) : m_timeStamp(timeStamp) { };
        ULONG VGetTimeStamp(VOID) { return m_timeStamp; }
        virtual EventType VGetEventType(VOID) = 0;
        virtual CONST CHAR* VGetName(VOID) = 0;
        virtual ~Event(VOID) { }
    };

    class SetSunPositionEvent : public Event
    {
    public:
        util::Vec3 m_position;
        SetSunPositionEvent(FLOAT x, FLOAT y, FLOAT z);
        static const EventType TYPE;
        EventType VGetEventType(VOID) { return TYPE; }
        CONST CHAR* VGetName(VOID) { return "SetSunPositionEvent"; }
    };

    /*
    class NewRenderComponentEvent : public Event {
    public:
        static const EventType TYPE;

        ActorId m_actorId;

        NewRenderComponentEvent(ActorId actorId) : m_actorId(actorId) { }

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "NewRenderComponentEvent"; }
    }; */


    class MoveActorEvent : public Event 
    {

    public:
        static const EventType TYPE;
        MoveActorEvent(ActorId id, util::Vec3 translation, util::Vec3 rotation, BOOL isDeltaMove = TRUE) : m_id(id), m_translation(translation), m_rotation(rotation) , m_isDelta(isDeltaMove) {}
        ActorId m_id;
        util::Vec3 m_translation;
        util::Vec3 m_rotation;
        BOOL m_isDelta;

        BOOL IsDeltaMove(VOID)
        {
            return m_isDelta;
        }

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "MoveActorEvent"; }

    };

    class ActorMovedEvent : public Event 
    {

    public:
        static const EventType TYPE;
        std::shared_ptr<tbd::Actor> m_actor;

        ActorMovedEvent(std::shared_ptr<tbd::Actor> actor) : m_actor(actor) {}

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "ActorMovedEvent"; }

    };

    /*
    class NewPhysicComponentEvent : public Event {

    public:
        static const EventType TYPE;
        ActorId m_id;

        NewPhysicComponentEvent(ActorId id) : m_id(id) {}

        EventType VGetEventType(VOID) { return TYPE; }
    }; */

    class NewComponentCreatedEvent : public Event 
    {
    public:
        static const EventType TYPE;
        ComponentId m_id;
        ActorId m_actorId;
        NewComponentCreatedEvent(ComponentId id, ActorId actorId) : m_id(id), m_actorId(actorId) {}

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "NewComponentCreatedEvent"; }
    };

    class CreateActorEvent : public Event 
    {

    public:

        BOOL m_appendToCurrentLevel;

        CreateActorEvent(VOID) : m_appendToCurrentLevel(TRUE) { }

        static const EventType TYPE;

        tbd::ActorDescription m_actorDesc;

        CreateActorEvent(tbd::ActorDescription desc) : m_actorDesc(desc), m_appendToCurrentLevel(TRUE) {}

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "CreateActorEvent"; }

    };

    class DeleteActorEvent : public Event 
    {
    public:
        static const EventType TYPE;
        ActorId m_id;
        DeleteActorEvent(ActorId id) : m_id(id) {}
        EventType VGetEventType(VOID) { return TYPE; }
        CONST CHAR* VGetName(VOID) { return "DeleteActorEvent"; }
    };

    /*
    class ComponentComponentEvent : public Event
    {

    public:
        static const EventType TYPE;
        ActorId m_actorId;
        ComponentId m_id;

        EventType VGetEventType(VOID) { return TYPE; }
        CONST CHAR* VGetName(VOID) { return "DeleteActorEvent"; }

    }; */

    class ActorDeletedEvent : public Event
    {
    public:
        static const EventType TYPE;
        ActorId m_id;
        ActorDeletedEvent(ActorId id) : m_id(id) {}
        EventType VGetEventType(VOID) { return TYPE; }
        CONST CHAR* VGetName(VOID) { return "ActorDeletedEvent"; }
    };

    class ActorCreatedEvent : public Event 
    {

    public:
        static const EventType TYPE;

        ActorId m_id;

        ActorCreatedEvent(ActorId id) : m_id(id) {}

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "ActorCreatedEvent"; }
    };

    class ResourceLoadedEvent : public Event
    {
        //TODO
    };

    class LoadingLevelEvent : public Event
    {
    public:
        static const EventType TYPE;

        std::string m_name;

        LoadingLevelEvent(std::string name) : m_name(name) {}

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "LoadingLevelEvent"; }
    };

    class LevelLoadedEvent : public Event
    {
    public:
        static const EventType TYPE;

        std::string m_name;

        LevelLoadedEvent(std::string name) : m_name(name) {}

        EventType VGetEventType(VOID) { return TYPE; }

        CONST CHAR* VGetName(VOID) { return "LevelLoadedEvent"; }
    };

    class CollisionEvent : public Event
    {
    public:
        static const EventType TYPE;
        ActorId m_actor0;
        ActorId m_actor1;
        CONST CHAR* VGetName(VOID) { return "CollisionEvent"; }
        EventType VGetEventType(VOID) { return TYPE; }
    };

    class ApplyForceEvent : public Event
    {
    public:
        static const EventType TYPE;
        ApplyForceEvent(VOID) : m_newtons(1) {}
        std::shared_ptr<tbd::Actor> m_actor;
        util::Vec3 m_dir;
        FLOAT m_newtons;
        CONST CHAR* VGetName(VOID) { return "ApplyForceEvent"; }
        EventType VGetEventType(VOID) { return TYPE; }
    };

    class ApplyTorqueEvent : public Event
    {
    public:
        static const EventType TYPE;
        ApplyTorqueEvent(VOID) : m_newtons(1) {}
        std::shared_ptr<tbd::Actor> m_actor;
        util::Vec3 m_torque;
        FLOAT m_newtons;
        CONST CHAR* VGetName(VOID) { return "ApplyTorqueEvent"; }
        EventType VGetEventType(VOID) { return TYPE; }
    };

    class TriggerEvent : public Event
    {
    public:
        static const EventType TYPE;
        ActorId m_didTriggerActor;
        ActorId m_triggerActor;
        CONST CHAR* VGetName(VOID) { return "TriggerEvent"; }
        EventType VGetEventType(VOID) { return TYPE; }
    };
};
