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

    SetSunPositionEvent::SetSunPositionEvent(FLOAT x, FLOAT y, FLOAT z)
    {
        m_position.x = x;
        m_position.y = y;
        m_position.z = z;
    }
};
