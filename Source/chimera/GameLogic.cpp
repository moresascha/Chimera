#include "GameLogic.h"
#include "Actor.h"
#include "Components.h"
#include "EventManager.h"
#include "GameApp.h"
#include "Resources.h"
#include "Commands.h"

namespace tbd 
{

    BaseGameLogic::BaseGameLogic() : m_pPhysics(NULL), m_gameState(ePause), m_pLevel(NULL)
    {

    }

    BOOL BaseGameLogic::VInit(VOID) 
    {
        if(this->m_pPhysics) return TRUE; //todo

        this->m_pProcessManager = new proc::ProcessManager();

        event::EventListener listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::MoveActorDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::MoveActorEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::CreateActorDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::CreateActorEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::ActorCreatedDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::ActorCreatedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::LevelLoadedDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::LevelLoadedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::DeleteActorDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::DeleteActorEvent::TYPE);

        this->m_pPhysics = new logic::PhysX();
        this->m_pPhysics->VInit();

        m_pCmdInterpreter = new tbd::CommandInterpreter();

        return TRUE;
    }

    VOID BaseGameLogic::AttachGameView(std::shared_ptr<tbd::IGameView> view, std::shared_ptr<tbd::Actor> actor) 
    {
        this->m_gameViewList.push_back(view);
        m_actorToViewMap[actor->GetId()] = view;
        view->VOnAttach((UINT)this->m_gameViewList.size(), actor);
    }

    VOID BaseGameLogic::AttachProcess(std::shared_ptr<proc::Process> process) 
    {
        this->m_pProcessManager->Attach(process);
    }

    VOID BaseGameLogic::VOnUpdate(ULONG millis) 
    {
        //TODO make member
        /**static UINT time = 0;
        time += millis;
        if(time < 16) 
        {
            return;
        }
        DEBUG_OUT(time);
        time = 0;
        millis = 16;*/
        switch(m_gameState)
        {
        case eRunning :
            {
                this->m_pPhysics->VUpdate(millis * 1e-3f);

                this->m_pProcessManager->Update(millis);

                for(auto it = this->m_gameViewList.begin(); it != this->m_gameViewList.end(); ++it)
                {
                    (*it)->VOnUpdate(millis);
                }

                /*for(auto it = this->m_actors.begin(); it != this->m_actors.end(); ++it) //not used at all
                {
                    it->second->Update(millis);
                } */

                app::g_pApp->GetLogic()->GetPhysics()->VSyncScene(); //PHX SYNC

            } break;
        case eLoadingLevel : 
            {
                this->m_pProcessManager->Update(millis);

                for(auto it = this->m_gameViewList.begin(); it != this->m_gameViewList.end(); ++it)
                {
                    (*it)->VOnUpdate(millis);
                }

                for(auto it = this->m_actors.begin(); it != this->m_actors.end(); ++it)
                {
                    it->second->Update(millis);
                }

            } break;

        case ePause : 
            {

            } break;

        default : LOG_CRITICAL_ERROR("unknown game state"); break;
        }

        /*
        if(this->m_pPhysics) 
        {
            this->m_pPhysics->VUpdate(millis * 1e-3f);
        } */
    }

    VOID BaseGameLogic::VOnRender(VOID) 
    {
        switch(m_gameState)
        {
        case eRunning : 
            {
                this->m_pPhysics->VSyncScene();
            } break;
        }

        for(auto it = this->m_gameViewList.begin(); it != this->m_gameViewList.end(); ++it)
        {
            (*it)->VPreRender();
        }

        for(auto it = this->m_gameViewList.begin(); it != this->m_gameViewList.end(); ++it)
        {
            (*it)->VOnRender(0, 0); // TODO
        }

        for(auto it = this->m_gameViewList.begin(); it != this->m_gameViewList.end(); ++it)
        {
            (*it)->VPostRender();
        }
    }

    std::shared_ptr<tbd::Actor> BaseGameLogic::VFindActor(ActorId id) 
    {
        auto it = m_actors.find(id);
        if(it == m_actors.end()) 
        {
            /*
            std::stringstream ss;
            ss << id;
            LOG_ERROR("Actor does not exist: " + ss.str()); */
            return m_pLevel->VFindActor(id);
        }
        else
        {
            return it->second;
        }
        return NULL;
    }

    std::shared_ptr<tbd::Actor> BaseGameLogic::VFindActor(LPCSTR name)
    {
        //this implementation is really slow and should only be used for debugging atm
        TBD_FOR(m_actors)
        {
            if(it->second->GetName() == name)
            {
                return it->second;
            }
        }
        return std::shared_ptr<tbd::Actor>(NULL);
    }

    std::shared_ptr<tbd::IGameView> BaseGameLogic::VFindGameView(GameViewId id)
    {
        //this implementation is really slow and should only be used for debugging atm
        TBD_FOR(m_gameViewList)
        {
            if((*it)->GetId() == id)
            {
                return *it;
            }
        }
        return std::shared_ptr<tbd::IGameView>(NULL);
    }

    std::shared_ptr<tbd::IGameView> BaseGameLogic::VFindGameView(LPCSTR name)
    {
        //this implementation is really slow and should only be used for debugging atm
        TBD_FOR(m_gameViewList)
        {
            if((*it)->GetName() == name)
            {
                return *it;
            }
        }
        return std::shared_ptr<tbd::IGameView>(NULL);
    }
    /*
    std::shared_ptr<tbd::Actor> BaseGameLogic::VCreateActor(TiXmlElement* pData) 
    {
        if(!pData) 
        {
            LOG_ERROR("TiXmlElement cant be NULL");
        }
        std::shared_ptr<tbd::Actor> actor = m_actorFactory.CreateActor(pData);
        if(actor)
        {
            this->m_actors[actor->GetId()] = actor;
            //actor factory takes care of it
            //event::IEventPtr actorCreatedEvent(new event::ActorCreatedEvent(actor->GetId()));
            //event::IEventManager::Get()->VQueueEvent(actorCreatedEvent);

            return actor;
        }
        LOG_ERROR("Failed to create actor");
        return std::shared_ptr<tbd::Actor>();
    } */

    std::shared_ptr<tbd::Actor> BaseGameLogic::VCreateActor(tbd::ActorDescription desc)
    {
        std::shared_ptr<tbd::Actor> actor = m_actorFactory.CreateActor(desc);

        m_actors[actor->GetId()] = actor;

        return actor;
    }

    std::shared_ptr<tbd::Actor> BaseGameLogic::VCreateActor(CONST CHAR* resource) 
    {
        if(!resource) 
        {
            LOG_CRITICAL_ERROR("Ressource cant be NULL");
        }
        std::string path = app::g_pApp->GetConfig()->GetString("sActorPath") + resource;
        std::shared_ptr<tbd::Actor> actor = m_actorFactory.CreateActor(path.c_str());
        if(actor)
        {
            //throw actor created event
            this->m_actors[actor->GetId()] = actor;

            return actor;
        }
        LOG_CRITICAL_ERROR_A("Failed to create actor: %s", resource);
        return std::shared_ptr<tbd::Actor>();
    }

    VOID BaseGameLogic::VRemoveActor(ActorId id) 
    {
        m_actors.erase(id);
        auto it = m_actorToViewMap.find(id);
        if(it != m_actorToViewMap.end())
        {
            m_actorToViewMap.erase(it);
        }
        m_pLevel->VRemoveActor(id);
    }

    BOOL BaseGameLogic::VLoadLevel(tbd::ILevel* level)
    {
        m_gameState = eLoadingLevel;

        event::IEventPtr loadingLevelEvent(new event::LoadingLevelEvent(std::string("test")));
        event::IEventManager::Get()->VQueueEventThreadSave(loadingLevelEvent);

        if(m_pLevel == level)
        {
            level->VUnload();
        }
        else
        {
            SAFE_DELETE(m_pLevel);
        }
        m_pLevel = level;
        m_pLevel->VLoad(FALSE);
        return TRUE;
    }

    BOOL BaseGameLogic::VLoadLevel(CONST CHAR* resource) 
    {
        m_gameState = eLoadingLevel;

        event::IEventPtr loadingLevelEvent(new event::LoadingLevelEvent(std::string(resource)));
        event::IEventManager::Get()->VQueueEventThreadSave(loadingLevelEvent);

        m_pLevel = new tbd::RandomLevel("asd", &m_actorFactory);
        //m_pLevel = new tbd::XMLLevel(resource); 

        m_pLevel->VLoad(FALSE);

        return TRUE;
    }

    FLOAT BaseGameLogic::GetLevelLoadProgress(VOID) CONST
    {
        return m_pLevel->VGetLoadingProgress();
    }

    UINT BaseGameLogic::GetLevelActorCount(VOID) CONST
    {
        return m_pLevel->VGetActorsCount();
    }

    VOID BaseGameLogic::ActorCreatedDelegate(event::IEventPtr eventData)
    {

    }

    VOID BaseGameLogic::MoveActorDelegate(event::IEventPtr eventData) 
    {
        std::shared_ptr<event::MoveActorEvent> data = std::static_pointer_cast<event::MoveActorEvent>(eventData);
        std::shared_ptr<tbd::Actor> actor = this->VFindActor(data->m_id);
        if(actor)
        {
            std::shared_ptr<tbd::PhysicComponent> physxCmp = actor->GetComponent<tbd::PhysicComponent>(tbd::PhysicComponent::COMPONENT_ID).lock();

            if(physxCmp)
            {
                //std::shared_ptr<tbd::CameraComponent> camCmp = actor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
                util::Vec4* rotatioQuat = NULL;
                util::Vec3* axis = NULL;
                util::Vec3* translation = NULL;
                if(data->m_hasRotation)
                {
                    if(data->m_hasQuatRotation)
                    {
                        rotatioQuat = &data->m_quatRotation;
                    }
                    else if(data->m_hasAxisRotation)
                    {
                        axis = &data->m_axis;
                    }
                    else
                    {
                        LOG_CRITICAL_ERROR("A rotation is set but no type of rotation!");
                    }
                }
                if(data->m_hasTranslation)
                {
                    translation = &data->m_translation;
                }
                if(rotatioQuat)
                {
                    m_pPhysics->VMoveKinematic(actor, translation, rotatioQuat, 0.5f, data->IsDeltaMove(), data->m_isJump);
                }
                else if(axis)
                {
                    m_pPhysics->VMoveKinematic(actor, translation, axis, data->m_angle, 0.5f, data->IsDeltaMove(), data->m_isJump);
                }
                else if(translation)
                {
                    m_pPhysics->VMoveKinematic(actor, translation, NULL, 0.5f, data->IsDeltaMove(), data->m_isJump);
                }
            }
            else
            {
                std::shared_ptr<tbd::TransformComponent> comp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
                if(comp)
                {
                    util::Mat4* transformation = comp->GetTransformation();
 
                    if(data->IsDeltaMove())
                    {
                        if(data->m_hasRotation)
                        {
                            if(data->m_hasQuatRotation)
                            {
                                transformation->RotateQuat(data->m_quatRotation);
                            }
                            else if(data->m_hasAxisRotation)
                            {
                                transformation->Rotate(data->m_axis, data->m_angle);
                            }
                            else
                            {
                                LOG_CRITICAL_ERROR("A rotation is set but no type of rotation!");
                            }
                        }

                        if(data->m_hasTranslation)
                        {
                            util::Vec3& translation = data->m_translation;
                            transformation->Translate(translation.x, translation.y, translation.z);
                        }
                    }
                    else
                    {
                        if(data->m_hasRotation)
                        {
                            if(data->m_hasQuatRotation)
                            {
                                transformation->SetRotateQuat(data->m_quatRotation);
                            }
                            else if(data->m_hasAxisRotation)
                            {
                                transformation->SetRotation(data->m_axis, data->m_angle);
                            }
                            else
                            {
                                LOG_CRITICAL_ERROR("A rotation is set but no type of rotation!");
                            }
                        }

                        if(data->m_hasTranslation)
                        {
                            util::Vec3& translation = data->m_translation;
                            transformation->SetTranslate(translation.x, translation.y, translation.z);
                        }
                    }
                }

                QUEUE_EVENT(new event::ActorMovedEvent(actor));
            }
        }
    }

    VOID BaseGameLogic::CreateActorDelegate(event::IEventPtr eventData) 
    {
        std::shared_ptr<event::CreateActorEvent> data = std::static_pointer_cast<event::CreateActorEvent>(eventData);
        if(data->m_appendToCurrentLevel)
        {
            m_pLevel->VAddActor(data->m_actorDesc);
        }
        else
        {
            std::shared_ptr<tbd::Actor> actor = m_actorFactory.CreateActor(data->m_actorDesc);
            m_actors[actor->GetId()] = actor;
        }

        //actor factory takes care of it
        /*event::IEventPtr actorCreatedEvent(new event::ActorCreatedEvent(actor->GetId()));
        event::IEventManager::Get()->VQueueEvent(actorCreatedEvent); */
    }

    VOID BaseGameLogic::DeleteActorDelegate(event::IEventPtr eventData)
    {
        std::shared_ptr<event::DeleteActorEvent> data = std::static_pointer_cast<event::DeleteActorEvent>(eventData);
        ActorId id = data->m_id;
        tbd::Actor* actor = VFindActor(id).get();
        if(actor)
        {
            for(auto cmps = actor->GetComponents().begin(); cmps != actor->GetComponents().end(); ++cmps)
            {
                //TODO extra event for components
            }
            //m_actors.erase(it);
            VRemoveActor(id);
            m_pPhysics->VRemoveActor(id);
            event::IEventPtr actorDeletedEvent(new event::ActorDeletedEvent(id));
            event::IEventManager::Get()->VQueueEvent(actorDeletedEvent);
        }
    }

    VOID BaseGameLogic::LevelLoadedDelegate(event::IEventPtr eventData)
    {
        m_gameState = eRunning;
    }

    BaseGameLogic::~BaseGameLogic() 
    {
        m_pLevel->VSave();

        event::EventListener listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::MoveActorDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::MoveActorEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::CreateActorDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::CreateActorEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::ActorCreatedDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorCreatedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::LevelLoadedDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::LevelLoadedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &BaseGameLogic::DeleteActorDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::DeleteActorEvent::TYPE);
    
        SAFE_DELETE(m_pLevel);

        m_gameViewList.clear();

        SAFE_DELETE(m_pCmdInterpreter);

        this->m_pProcessManager->AbortAll(TRUE);

        SAFE_DELETE(this->m_pProcessManager);

        SAFE_DELETE(m_pPhysics);

        for(auto it = this->m_actors.begin(); it != this->m_actors.end(); ++it)
        {
            it->second->Destroy();
        }

        this->m_actors.clear();
    }
}
