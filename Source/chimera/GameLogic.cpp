#include "GameLogic.h"
#include "Actor.h"
#include "Components.h"
#include "EventManager.h"
#include "Cache.h"
#include "Commands.h"
#include "ProcessManager.h"
#include "Process.h"
#include "GameView.h"
#include "PhysicsSystem.h"
#include "ActorFactory.h"
#include "Level.h"

namespace chimera 
{

    BaseGameLogic::BaseGameLogic() : m_pPhysics(NULL), m_gameState(CM_STATE_RUNNING), m_pLevel(NULL)
    {

    }

    BOOL BaseGameLogic::VInitialise(FactoryPtr* facts) 
    {
        IActorFactoryFactory* aff = FindFactory<IActorFactoryFactory>(facts, CM_FACTORY_ACTOR);
        RETURN_IF_FAILED(aff);
		m_pActorFactory = aff->VCreateActorFactroy();

        IHumanViewFactory* hgwf = FindFactory<IHumanViewFactory>(facts, CM_FACTORY_VIEW);
        RETURN_IF_FAILED(hgwf);
        m_pHumanView = hgwf->VCreateHumanView();

        m_pProcessManager = new ProcessManager();

        ADD_EVENT_LISTENER(this, &BaseGameLogic::MoveActorDelegate, CM_EVENT_MOVE_ACTOR);

        ADD_EVENT_LISTENER(this,  &BaseGameLogic::CreateActorDelegate, CM_EVENT_CREATE_ACTOR);

        ADD_EVENT_LISTENER(this,  &BaseGameLogic::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);

        ADD_EVENT_LISTENER(this,  &BaseGameLogic::LevelLoadedDelegate, CM_EVENT_LEVEL_LOADED);

        ADD_EVENT_LISTENER(this,  &BaseGameLogic::DeleteActorDelegate, CM_EVENT_DELETE_ACTOR);

        ADD_EVENT_LISTENER(this, &BaseGameLogic::CreateProcessDelegate, CM_EVENT_CREATE_PROCESS);

        ADD_EVENT_LISTENER(this, &BaseGameLogic::LoadLevelDelegate, CM_EVENT_LOAD_LEVEL);

        m_pPhysics = new PhysX();
        m_pPhysics->VInit();

        m_pCmdInterpreter = new CommandInterpreter();

        //m_pLevelManager = new chimera::LevelManager();

        return TRUE;
    }

    VOID BaseGameLogic::VAttachView(std::unique_ptr<IView> view, ActorId actor) 
    {
		IView* raw = view.get();
        m_actorToViewMap[actor] = raw;
		m_gameViewList.push_back(std::move(view));
        raw->VOnAttach((UINT)m_gameViewList.size(), VFindActor(actor));
    }

    VOID BaseGameLogic::VOnUpdate(ULONG millis) 
    {

        m_pPhysics->VUpdate(millis * 1e-3f);

        m_pProcessManager->VUpdate(millis);

        TBD_FOR(m_gameViewList)
        {
            (*it)->VOnUpdate(millis);
        }

        m_pHumanView->VOnUpdate(millis);

        CmGetApp()->VGetLogic()->VGetPhysics()->VSyncScene(); //PHX SYNC

        /*
        switch(m_gameState)
        {
        case eRunning :
            {
                m_pPhysics->VUpdate(millis * 1e-3f);

                m_pProcessManager->VUpdate(millis);

                for(auto it = this->m_gameViewList.begin(); it != this->m_gameViewList.end(); ++it)
                {
                    (*it)->VOnUpdate(millis);
                }

                chimera::g_pApp->GetLogic()->GetPhysics()->VSyncScene(); //PHX SYNC

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
        }*/

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
        case CM_STATE_RUNNING : 
            {
                m_pPhysics->VSyncScene();
            } break;
        }

        m_pHumanView->VOnRender();
    }

    IActor* BaseGameLogic::VFindActor(ActorId id) 
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
            return it->second.get();
        }
        return NULL;
    }

    IActor* BaseGameLogic::VFindActor(LPCSTR name)
    {
        //this implementation is really slow and should only be used for debugging atm
        TBD_FOR(m_actors)
        {
            if(it->second->GetName() == name)
            {
                return it->second.get();
            }
        }
        return NULL;
    }

    IView* BaseGameLogic::VFindView(GameViewId id)
    {
        //this implementation is really slow and should only be used for debugging atm
        TBD_FOR(m_gameViewList)
        {
            if((*it)->GetId() == id)
            {
                return it->get();
            }
        }
        return NULL;
    }

    IView* BaseGameLogic::VFindView(LPCSTR name)
    {
        //this implementation is really slow and should only be used for debugging atm
        TBD_FOR(m_gameViewList)
        {
            if((*it)->GetName() == name)
            {
                return it->get();
            }
        }
        return NULL;
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

    IActor* BaseGameLogic::VCreateActor(std::unique_ptr<ActorDescription> desc, BOOL appendToLevel)
    {
        IActor* actor = NULL;
        if(appendToLevel)
        {
            actor = m_pLevel->VAddActor(std::move(desc));
			return actor;
        }
        else
        {
            std::unique_ptr<IActor> upa = m_pActorFactory->VCreateActor(std::move(desc));
			actor = upa.get();
            m_actors[actor->GetId()] = std::move(upa);
			return actor;
        }
    }

    IActor* BaseGameLogic::VCreateActor(CONST CHAR* resource, BOOL appendToLevel) 
    {
        if(!resource) 
        {
            LOG_CRITICAL_ERROR("Ressource cant be NULL");
        }
        std::string path = CmGetApp()->VGetConfig()->VGetString("sActorPath") + resource;
        std::vector<std::unique_ptr<IActor>> actors;
        CMResource res(resource);
        std::unique_ptr<IActor> parent = m_pActorFactory->VCreateActor(res, actors);
        TBD_FOR(actors)
        {
            //throw actor created event
            m_actors[(*it)->GetId()] = std::move(*it);
        }
        if(!parent)
        {
            LOG_CRITICAL_ERROR_A("Failed to create actor: %s", resource);
        }
        return parent.get();
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

    BOOL BaseGameLogic::VLoadLevel(ILevel* level)
    {
        m_gameState = CM_STATE_LOADING;

        QUEUE_EVENT_TSAVE(new LoadingLevelEvent(level->VGetName()));

        if(m_pLevel == level)
        {
            level->VUnload();
        }
        else
        {
            SAFE_DELETE(m_pLevel);
        }
        m_pLevel = level;
        return m_pLevel->VLoad(FALSE);
    }

    BOOL BaseGameLogic::VLoadLevel(CONST CHAR* resource) 
    {
        m_gameState = CM_STATE_LOADING;

        QUEUE_EVENT_TSAVE(new LoadingLevelEvent(std::string(resource)));

        SAFE_DELETE(m_pLevel);

        //m_pLevel = new XMLLevel(resource, m_pActorFactory); 

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

    VOID BaseGameLogic::ActorCreatedDelegate(IEventPtr eventData)
    {

    }

    VOID BaseGameLogic::MoveActorDelegate(IEventPtr eventData) 
    {
        std::shared_ptr<MoveActorEvent> data = std::static_pointer_cast<MoveActorEvent>(eventData);
        IActor* actor = VFindActor(data->m_id);
        if(actor)
        {
            PhysicComponent* physxCmp = GetActorCompnent<PhysicComponent>(actor, CM_CMP_PHX);

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
                TransformComponent* comp = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM);
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

                QUEUE_EVENT(new ActorMovedEvent(actor));
            }
        }
    }

    VOID BaseGameLogic::CreateActorDelegate(IEventPtr eventData) 
    {
        std::shared_ptr<CreateActorEvent> data = std::static_pointer_cast<CreateActorEvent>(eventData);

        VCreateActor(std::move(data->m_actorDesc), data->m_appendToCurrentLevel);

        //actor factory takes care of it
        /*event::IEventPtr actorCreatedEvent(new event::ActorCreatedEvent(actor->GetId()));
        event::IEventManager::Get()->VQueueEvent(actorCreatedEvent); */
    }

    VOID BaseGameLogic::DeleteActorDelegate(IEventPtr eventData)
    {
        std::shared_ptr<DeleteActorEvent> data = std::static_pointer_cast<DeleteActorEvent>(eventData);
        ActorId id = data->m_id;
        IActor* actor = VFindActor(id);
        if(actor)
        {
            for(auto cmps = actor->VGetComponents().begin(); cmps != actor->VGetComponents().end(); ++cmps)
            {
                //TODO extra event for components
            }
            //m_actors.erase(it);
            VRemoveActor(id);
            m_pPhysics->VRemoveActor(id);
            QUEUE_EVENT(new ActorDeletedEvent(id));
        }
    }

    VOID BaseGameLogic::LoadLevelDelegate(IEventPtr eventData)
    {
        /*std::shared_ptr<LoadLevelEvent> data = std::static_pointer_cast<LoadLevelEvent>(eventData);
        VLoadLevel(new XMLLevel(data->m_name.c_str(), m_pActorFactory)); */
    }

    VOID BaseGameLogic::LevelLoadedDelegate(IEventPtr eventData)
    {
        m_gameState = eProcessState_Running;
    }

    VOID BaseGameLogic::CreateProcessDelegate(IEventPtr eventData)
    {
        std::shared_ptr<CreateProcessEvent> e = std::static_pointer_cast<CreateProcessEvent>(eventData);
        m_pProcessManager->VAttach(std::unique_ptr<IProcess>(e->m_pProcess));
    }

    BaseGameLogic::~BaseGameLogic() 
    {
        //m_pLevel->VSave();

        //SAFE_DELETE(m_pLevelManager);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::LoadLevelDelegate, CM_EVENT_LOAD_LEVEL);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::MoveActorDelegate, CM_EVENT_MOVE_ACTOR);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::CreateActorDelegate, CM_EVENT_CREATE_ACTOR);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::ActorCreatedDelegate, CM_EVENT_ACTOR_CREATED);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::LevelLoadedDelegate, CM_EVENT_LEVEL_LOADED);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::DeleteActorDelegate, CM_EVENT_DELETE_ACTOR);

        REMOVE_EVENT_LISTENER(this, &BaseGameLogic::CreateProcessDelegate, CM_EVENT_CREATE_PROCESS);
    
        SAFE_DELETE(m_pLevel);

        SAFE_DELETE(m_pCmdInterpreter);

        m_pProcessManager->VAbortAll(TRUE);

        SAFE_DELETE(m_pProcessManager);

        SAFE_DELETE(m_pPhysics);

        TBD_FOR(m_gameViewList)
        {
            it->reset();
        }

        TBD_FOR(m_actors)
        {
            it->second.reset();
        }

        m_gameViewList.clear();

        m_actors.clear();

        SAFE_DELETE(m_pHumanView);

        SAFE_DELETE(m_pActorFactory);
    }
}
