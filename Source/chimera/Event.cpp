#include "Event.h"
namespace event 
{
    //EventTypes
    //CONST EventType NewRenderComponentEvent::TYPE = 0x87b0a3c0;
    CONST EventType NewComponentCreatedEvent::TYPE = 0x87b0a3c0;
    CONST EventType MoveActorEvent::TYPE = 0xb0bd31cf;
    CONST EventType ActorMovedEvent::TYPE = 0x5926636f;
    //CONST EventType NewPhysicComponentEvent::TYPE = 0x8ef932ad;
    CONST EventType CreateActorEvent::TYPE = 0x6ed490ba;
    CONST EventType ActorCreatedEvent::TYPE = 0x7305a86e;
    CONST EventType DeleteActorEvent::TYPE = 0x22b669c6;
    CONST EventType ActorDeletedEvent::TYPE = 0xe1af4381;
    CONST EventType LoadingLevelEvent::TYPE = 0x916182da;
    CONST EventType LevelLoadedEvent::TYPE = 0xe0d6fd3f;
    CONST EventType CollisionEvent::TYPE = 0x1c3fb533;
    CONST EventType SetSunPositionEvent::TYPE = 0x47fa57b4;
    CONST EventType ApplyForceEvent::TYPE = 0x158be54a;
    CONST EventType ApplyTorqueEvent::TYPE = 0x1d17f65;
    CONST EventType TriggerEvent::TYPE = 0xa25e9d66;
    CONST EventType SetParentActorEvent::TYPE = 0xa5fd2f77;
    CONST EventType LoadLevelEvent::TYPE = 0x2d40e793;
    CONST EventType ResourceChangedEvent::TYPE = 0xec7c39db;

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
};
