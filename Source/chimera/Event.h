#pragma once
#include "stdafx.h"
#include "Vec3.h"

#define CM_EVENT_COMPONENT_CREATED 0x87b0a3c0
#define CM_EVENT_MOVE_ACTOR 0xb0bd31cf
#define CM_EVENT_ACTOR_MOVED 0x5926636f
#define CM_EVENT_CREATE_ACTOR 0x6ed490ba
#define CM_EVENT_ACTOR_CREATED 0x7305a86e
#define CM_EVENT_DELETE_ACTOR 0x22b669c6
#define CM_EVENT_ACTOR_DELETED 0xe1af4381
#define CM_EVENT_LOADING_LEVEL 0x916182da
#define CM_EVENT_LOAD_LEVEL 0xe0d6fd3f
#define CM_EVENT_COLLISION 0x1c3fb533
#define CM_EVENT_SET_SUN_POSITION 0x47fa57b4
#define CM_EVENT_APPLY_FORCE 0x158be54a
#define CM_EVENT_APPLY_TORQUE 0x1d17f65
#define CM_EVENT_TRIGGER 0xa25e9d66
#define CM_EVENT_SET_PARENT_ACTOR 0xa5fd2f77
#define CM_EVENT_LEVEL_LOADED 0x2d40e793
#define CM_EVENT_RESOURCE_CHANGED 0xec7c39db
#define CM_EVENT_FILE_CHANGED 0x8a985188
#define CM_EVENT_CREATE_PROCESS 0xd3f662bb

#define CM_CREATE_EVENT_HEADER(__type, __name) \
    EventType VGetEventType(VOID) CONST { return __type; } \
    LPCSTR VGetName(VOID) CONST { return #__name; }

namespace chimera 
{
    typedef IEvent Event;
    class SetSunPositionEvent : public Event
    {
    public:
        util::Vec3 m_position;
        CM_INLINE SetSunPositionEvent(FLOAT x, FLOAT y, FLOAT z);

        CM_CREATE_EVENT_HEADER(CM_EVENT_SET_SUN_POSITION, SetSunPositionEvent);
    };

    class MoveActorEvent : public Event 
    {

    public:       
        CM_INLINE MoveActorEvent(ActorId id, util::Vec3 translation, util::Vec4 quat, BOOL isDeltaMove = TRUE);
        CM_INLINE MoveActorEvent(ActorId id, util::Vec3 translation, util::Vec3 axis, FLOAT angle, BOOL isDeltaMove = TRUE);
        CM_INLINE MoveActorEvent(ActorId id, util::Vec3 translation, BOOL isDeltaMove = TRUE);
        CM_INLINE MoveActorEvent(ActorId id, util::Vec4 quat, BOOL isDeltaMove = TRUE);
        CM_INLINE MoveActorEvent(ActorId id, util::Vec3 axis, FLOAT angel, BOOL isDeltaMove = TRUE);
        
        ActorId m_id;
        util::Vec3 m_translation;
        util::Vec4 m_quatRotation;
        util::Vec3 m_axis;
        FLOAT m_angle;
        
        BOOL m_isDelta;
        BOOL m_isJump;

        BOOL m_hasRotation;
        BOOL m_hasQuatRotation;
        BOOL m_hasAxisRotation;
        BOOL m_hasTranslation;

        CM_INLINE BOOL IsDeltaMove(VOID)
        {
            return m_isDelta;
        }

        CM_CREATE_EVENT_HEADER(CM_EVENT_MOVE_ACTOR, MoveActorEvent);
    };

    class ActorMovedEvent : public Event 
    {

    public:
        IActor* m_actor;

        ActorMovedEvent(IActor* actor) : m_actor(actor) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_ACTOR_MOVED, ActorMovedEvent);
    };

    class NewComponentCreatedEvent : public Event 
    {
    public:
        ComponentId m_id;
        ActorId m_actorId;
        NewComponentCreatedEvent(ComponentId id, ActorId actorId) : m_id(id), m_actorId(actorId) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_COMPONENT_CREATED, NewComponentCreatedEvent);
    };

    class CreateActorEvent : public Event 
    {

    public:
        BOOL m_appendToCurrentLevel;

        std::unique_ptr<ActorDescription> m_actorDesc;

        CreateActorEvent(VOID) : m_appendToCurrentLevel(TRUE) { }

        CreateActorEvent(std::unique_ptr<ActorDescription> desc) : m_actorDesc(std::move(desc)), m_appendToCurrentLevel(TRUE) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_CREATE_ACTOR, CreateActorEvent);
    };

    class DeleteActorEvent : public Event 
    {
    public:
        ActorId m_id;
        DeleteActorEvent(ActorId id) : m_id(id) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_DELETE_ACTOR, DeleteActorEvent);
    };

    class ActorDeletedEvent : public Event
    {
    public:
        ActorId m_id;
        ActorDeletedEvent(ActorId id) : m_id(id) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_ACTOR_DELETED, ActorDeletedEvent);
    };

    class ActorCreatedEvent : public Event 
    {

    public:
        ActorId m_id;

        ActorCreatedEvent(ActorId id) : m_id(id) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_ACTOR_CREATED, ActorCreatedEvent);
    };

    class ResourceLoadedEvent : public Event
    {
        //TODO
    };

    class LoadLevelEvent : public Event
    {
    public:
        std::string m_name;

        CM_CREATE_EVENT_HEADER(CM_EVENT_LOAD_LEVEL, LoadLevelEvent);
    };

    class LoadingLevelEvent : public Event
    {
    public:
        std::string m_name;

        LoadingLevelEvent(std::string name) : m_name(name) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_LOADING_LEVEL, LoadingLevelEvent);
    };

    class LevelLoadedEvent : public Event
    {
    public:
        std::string m_name;

        LevelLoadedEvent(std::string name) : m_name(name) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_LEVEL_LOADED, LevelLoadedEvent);
    };

    class CollisionEvent : public Event
    {
    public:
        ActorId m_actor0;
        ActorId m_actor1;
        util::Vec3 m_position;
        util::Vec3 m_impulse;
        
        CM_CREATE_EVENT_HEADER(CM_EVENT_COLLISION, CollisionEvent);
    };

    class ApplyForceEvent : public Event
    {
    public:
        ApplyForceEvent(VOID) : m_newtons(1) {}
        IActor* m_actor;
        util::Vec3 m_dir;
        FLOAT m_newtons;

        CM_CREATE_EVENT_HEADER(CM_EVENT_APPLY_FORCE, ApplyForceEvent);
    };

    class ApplyTorqueEvent : public Event
    {
    public:
        ApplyTorqueEvent(VOID) : m_newtons(1) {}
        IActor* m_actor;
        util::Vec3 m_torque;
        FLOAT m_newtons;

        CM_CREATE_EVENT_HEADER(CM_EVENT_APPLY_TORQUE, ApplyTorqueEvent);
    };

    class TriggerEvent : public Event
    {
    public:
        ActorId m_didTriggerActor;
        ActorId m_triggerActor;

        CM_CREATE_EVENT_HEADER(CM_EVENT_TRIGGER, TriggerEvent);
    };

    class SetParentActorEvent : public Event
    {
    public:
        ActorId m_actor;
        ActorId m_parentActor;

        CM_CREATE_EVENT_HEADER(CM_EVENT_SET_PARENT_ACTOR, SetParentActorEvent);
    };

    class ResourceChangedEvent : public Event
    {
    public:
        std::string m_resource;
        ResourceChangedEvent(std::string& res) : m_resource(res) {}

        CM_CREATE_EVENT_HEADER(CM_EVENT_RESOURCE_CHANGED, ResourceChangedEvent);
    };

    class FileChangedEvent : public Event
    {
    public:
        std::string m_file;
        CM_INLINE FileChangedEvent(LPCTSTR file);
        CM_INLINE FileChangedEvent(LPCSTR file);

        CM_CREATE_EVENT_HEADER(CM_EVENT_FILE_CHANGED, FileChangedEvent);
    };

    class CreateProcessEvent : public Event
    {
    public:
        IProcess* m_pProcess;
        CM_CREATE_EVENT_HEADER(CM_EVENT_CREATE_PROCESS, CreateProcessEvent);
    };


    SetSunPositionEvent::SetSunPositionEvent(FLOAT x, FLOAT y, FLOAT z)
    {
        m_position.x = x;
        m_position.y = y;
        m_position.z = z;
    }

    MoveActorEvent::MoveActorEvent(ActorId id, util::Vec3 translation, util::Vec4 quat, BOOL isDeltaMove) 
        : 
        m_id(id),
        m_translation(translation), 
        m_quatRotation(quat),
        m_isDelta(isDeltaMove),
        m_hasRotation(TRUE),
        m_isJump(FALSE),
        m_hasQuatRotation(TRUE),
        m_hasAxisRotation(FALSE),
        m_hasTranslation(TRUE)
    {

    }

    MoveActorEvent::MoveActorEvent(ActorId id, util::Vec3 translation, util::Vec3 axis, FLOAT angle, BOOL isDeltaMove)
        : 
        m_id(id),
        m_translation(translation), 
        m_axis(axis),
        m_angle(angle),
        m_isJump(FALSE),
        m_isDelta(isDeltaMove),
        m_hasRotation(TRUE),
        m_hasQuatRotation(FALSE),
        m_hasAxisRotation(TRUE),
        m_hasTranslation(TRUE)
    {

    }

    MoveActorEvent::MoveActorEvent(ActorId id, util::Vec3 translation, BOOL isDeltaMove)
        : 
        m_id(id),
        m_translation(translation), 
        m_isDelta(isDeltaMove),
        m_isJump(FALSE),
        m_hasRotation(FALSE),
        m_hasTranslation(TRUE)
    {

    }

    MoveActorEvent::MoveActorEvent(ActorId id, util::Vec4 quat, BOOL isDeltaMove)
        : 
        m_id(id),
        m_quatRotation(quat), 
        m_isDelta(isDeltaMove),
        m_isJump(FALSE),
        m_hasQuatRotation(TRUE),
        m_hasRotation(TRUE),
        m_hasTranslation(FALSE)
    {

    }

    MoveActorEvent::MoveActorEvent(ActorId id, util::Vec3 axis, FLOAT angle, BOOL isDeltaMove)
        : 
        m_id(id),
        m_isDelta(isDeltaMove),
        m_isJump(FALSE),
        m_axis(axis),
        m_angle(angle),
        m_hasRotation(TRUE),
        m_hasQuatRotation(FALSE),
        m_hasAxisRotation(TRUE),
        m_hasTranslation(FALSE)
    {

    }

    FileChangedEvent::FileChangedEvent(LPCTSTR file)
    {
        std::wstring ws(file);
        m_file = std::string(ws.begin(), ws.end());
    }

    FileChangedEvent::FileChangedEvent(LPCSTR file) : m_file(file)
    {

    }
};
