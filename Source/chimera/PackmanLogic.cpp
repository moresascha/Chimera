#include "PackmanLogic.h"
#include "EventManager.h"
#include "Packman.h"
#include "AIView.h"
namespace packman
{
    BOOL PackmanLogic::VInit(VOID)
    {
        ADD_EVENT_LISTENER(this, &PackmanLogic::ComponentCreatedDelegate, chimera::NewComponentCreatedEvent::TYPE);

        return chimera::BaseGameLogic::VInit();
    }

    VOID PackmanLogic::ComponentCreatedDelegate(chimera::IEventPtr data)
    {
        std::shared_ptr<chimera::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<chimera::NewComponentCreatedEvent>(data);
        if(pCastEventData->m_id == packman::AIComponent::COMPONENT_ID)
        {
            std::shared_ptr<chimera::Actor> actor = VFindActor(pCastEventData->m_actorId);
            AttachGameView(std::shared_ptr<chimera::IGameView>(new packman::EnemyAIView(GetMaze())), actor);
        }
    }

    PackmanLogic::~PackmanLogic(VOID)
    {
        REMOVE_EVENT_LISTENER(this, &PackmanLogic::ComponentCreatedDelegate, chimera::NewComponentCreatedEvent::TYPE);
    }
}
