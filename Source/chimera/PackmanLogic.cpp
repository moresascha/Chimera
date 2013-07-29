#include "PackmanLogic.h"
#include "EventManager.h"
#include "Packman.h"
#include "AIView.h"
namespace packman
{
    BOOL PackmanLogic::VInit(VOID)
    {
        ADD_EVENT_LISTENER(this, &PackmanLogic::ComponentCreatedDelegate, event::NewComponentCreatedEvent::TYPE);

        return tbd::BaseGameLogic::VInit();
    }

    VOID PackmanLogic::ComponentCreatedDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<event::NewComponentCreatedEvent>(data);
        if(pCastEventData->m_id == packman::AIComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::Actor> actor = VFindActor(pCastEventData->m_actorId);
            AttachGameView(std::shared_ptr<tbd::IGameView>(new packman::EnemyAIView(GetMaze())), actor);
        }
    }

    PackmanLogic::~PackmanLogic(VOID)
    {
        REMOVE_EVENT_LISTENER(this, &PackmanLogic::ComponentCreatedDelegate, event::NewComponentCreatedEvent::TYPE);
    }
}
