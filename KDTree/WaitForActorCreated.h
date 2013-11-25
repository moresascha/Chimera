#pragma once
#include <ChimeraAPI.h>
#include <../Source/chimera/Components.h>
#include <../Source/chimera/Event.h>

typedef void(*__CallBack)(chimera::ActorId);

class WaitForActor
{
private:

    __CallBack m_cb;
    chimera::ActorId m_id;

    void listener(chimera::IEventPtr ptr)
    {
        std::shared_ptr<chimera::ActorCreatedEvent> e = std::static_pointer_cast<chimera::ActorCreatedEvent>(ptr);
        if(e->m_id == m_id)
        {
            if(m_cb)
            {
                m_cb(m_id);
            }
        }
    }
public:
    void listen(chimera::ActorId id, __CallBack cb)
    {
        m_cb = cb;
        m_id = id;
        ADD_EVENT_LISTENER(this, &WaitForActor::listener, CM_EVENT_ACTOR_CREATED);
    }

    void stop(void)
    {
        REMOVE_EVENT_LISTENER(this, &WaitForActor::listener, CM_EVENT_ACTOR_CREATED);
    }

    ~WaitForActor(VOID)
    {

    }
};