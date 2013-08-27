#include "GameView.h"
#include "Event.h"
#include "EventManager.h"
#include "GameApp.h"
#include "Actor.h"
#include "Components.h"
#include "Vec3.h"
#include "D3DRenderer.h"
#include "PointLightNode.h"
#include "GeometryFactory.h"
#include "GameLogic.h"
#include "ParticleSystem.h"
#include "ParticleNode.h"
#include "SceneGraph.h"
#include "Picker.h"
#include "tbdFont.h"
#include "ParticleManager.h"
#include "Camera.h"
#include "GuiComponent.h"
#include "Sound.h"
#include "SpotlightNode.h"

namespace tbd
{
    IGameView::IGameView(VOID) : m_id(-1), m_actor(NULL), m_name("undefined")
    {

    }

    VOID IGameView::VOnAttach(UINT viewId, std::shared_ptr<tbd::Actor> actor) 
    {
        this->m_id = viewId;
        VSetTarget(actor);
    }

    VOID IGameView::VSetTarget(std::shared_ptr<tbd::Actor> actor)
    {
        if(actor)
        {
            m_actor = actor;
        }
    }

    std::shared_ptr<tbd::Actor> IGameView::GetTarget(VOID)
    {
        return m_actor;
    }

    CONST std::string& IGameView::GetName(VOID) CONST
    {
        return m_name;
    }

    VOID IGameView::SetName(CONST std::string& name)
    {
        m_name = name;
    }

    HumanGameView::HumanGameView(VOID) : m_loadingDots(0), m_picker(NULL), m_currentScene(NULL)
    {

    }

    d3d::D3DRenderer* HumanGameView::GetRenderer(VOID)
    {
        return app::g_pApp->GetRenderer();
    }

    tbd::VRamManager* HumanGameView::GetVRamManager(VOID) CONST
    {
        return app::g_pApp->GetVRamManager();
    }

    VOID HumanGameView::Resize(UINT w, UINT h)
    {
        if(d3d::g_pSwapChain) 
        {
            d3d::Resize(w, h);
            VOnRestore();
            BOOL fullscreen = d3d::GetFullscreenState();
            app::g_pApp->GetInputHandler()->VSetCurserOffsets(fullscreen ? 0 : 8 , fullscreen ? 0 : 30); //todo, get systemmetrics
        }
    }

    VOID HumanGameView::ActorMovedDelegate(event::IEventPtr eventData)  
    {
        std::shared_ptr<event::ActorMovedEvent> movedEvent = std::static_pointer_cast<event::ActorMovedEvent>(eventData);
        
        if(movedEvent->m_actor->GetId() == m_actor->GetId())
        {
            std::shared_ptr<tbd::TransformComponent> comp = m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
            CONST util::Vec3& trans = comp->GetTransformation()->GetTranslation();

            m_pSceneGraph->GetCamera()->SetRotation(comp->m_phi, comp->m_theta);

            m_pSceneGraph->GetCamera()->MoveToPosition(trans);

            GetRenderer()->VPushViewTransform(m_pSceneGraph->GetCamera()->GetView(), m_pSceneGraph->GetCamera()->GetIView(), m_pSceneGraph->GetCamera()->GetEyePos());
        }
    }

    VOID HumanGameView::NewComponentDelegate(event::IEventPtr pEventData) 
    {

        std::shared_ptr<event::NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<event::NewComponentCreatedEvent>(pEventData);
        std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(pCastEventData->m_actorId);

        if(pCastEventData->m_id == tbd::RenderComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::RenderComponent> comp = actor->GetComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID).lock();

            std::shared_ptr<tbd::ISceneNode> node;

            if(comp->m_sceneNode)
            {
                node = comp->m_sceneNode;
                comp->m_sceneNode = NULL; //this prevents memory leaks
                node->VSetActor(pCastEventData->m_actorId);
            }
            else if(comp->m_type == "anchor")
            {
                node = std::shared_ptr<tbd::AnchorNode>(
                    new tbd::AnchorNode(eSPHERE, actor->GetId(), comp->m_info.c_str(), comp->m_anchorRadius, comp->m_drawType == "solid" ? eSolid : eWire));
            }
            else if(comp->m_type == "skydome")
            {
                node = std::shared_ptr<tbd::SkyDomeNode>(new tbd::SkyDomeNode(actor->GetId(), comp->m_info));
            }
            else if(!comp->m_instances.empty())
            {
                node = std::shared_ptr<tbd::InstancedMeshNode>(new tbd::InstancedMeshNode(actor->GetId(), comp->m_meshFile));
            }
            else
            {
                 node = std::shared_ptr<tbd::MeshNode>(new tbd::MeshNode(actor->GetId(), comp->m_meshFile));
            }

            m_pSceneGraph->AddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);

            comp->VSetHandled();
        }

        else if(pCastEventData->m_id == tbd::LightComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::LightComponent> comp = actor->GetComponent<tbd::LightComponent>(tbd::LightComponent::COMPONENT_ID).lock();

            std::shared_ptr<tbd::SceneNode> node;

            if(comp->m_type == "point")
            {
                node = std::shared_ptr<tbd::PointlightNode>(new tbd::PointlightNode(actor->GetId()));
            }
            else if(comp->m_type == "spot")
            {
                node = std::shared_ptr<tbd::SpotlightNode>(new tbd::SpotlightNode(actor->GetId()));
            }
            else
            {
                LOG_CRITICAL_ERROR_A("Unknown lighttype: %s", comp->m_type.c_str());
            }

            m_pSceneGraph->AddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);

            comp->VSetHandled();
        }

        else if(pCastEventData->m_id == tbd::ParticleComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::ParticleComponent> comp = actor->GetComponent<tbd::ParticleComponent>(tbd::ParticleComponent::COMPONENT_ID).lock();

            std::shared_ptr<tbd::ParticleNode> node = std::shared_ptr<tbd::ParticleNode>(new tbd::ParticleNode(actor->GetId()));

            m_pSceneGraph->AddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);

            comp->VSetHandled();
        }
        else if(pCastEventData->m_id == tbd::CameraComponent::COMPONENT_ID)
        {
            std::shared_ptr<tbd::SceneNode> node = std::shared_ptr<tbd::CameraNode>(new tbd::CameraNode(actor->GetId()));

            m_pSceneGraph->AddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);
        }
    }

    VOID HumanGameView::AddScreenElement(tbd::IScreenElement* element)
    {
        m_screenElements.push_back(element);
        element->VOnRestore();
    }

    tbd::IScreenElement* HumanGameView::GetScreenElementByName(LPCSTR name)
    {
        TBD_FOR(m_screenElements)
        {
            if((*it)->VGetName() == std::string(name))
            {
                return *it;
            }
        }
        return NULL;
    }

    VOID HumanGameView::DeleteActorDelegate(event::IEventPtr pEventData)
    {
        std::shared_ptr<event::ActorDeletedEvent> pCastEventData = std::static_pointer_cast<event::ActorDeletedEvent>(pEventData);

        m_pSceneGraph->RemoveChild(pCastEventData->m_id);
    }

    VOID HumanGameView::LevelLoadedDelegate(event::IEventPtr pEventData)
    {

    }

    VOID HumanGameView::LoadingLevelDelegate(event::IEventPtr pEventData)
    {
        
    }

    VOID HumanGameView::SetParentDelegate(event::IEventPtr pEventData)
    {
        //if(pEventData->VGetEventType() == event::SetParentActorEvent::TYPE) needed?
        {
            std::shared_ptr<event::SetParentActorEvent> e = std::static_pointer_cast<event::SetParentActorEvent>(pEventData);
            ActorId actor = e->m_actor;
            ActorId parent = e->m_parentActor;

            std::shared_ptr<tbd::ISceneNode> p = m_pSceneGraph->FindActorNode(parent);
            std::shared_ptr<tbd::ISceneNode> a = m_pSceneGraph->FindActorNode(actor);
            if(p)
            {
                p->VAddChild(a);
            }
            else
            {
                LOG_CRITICAL_ERROR("SetParentFailed - No Actor found");
            }
        }
    }

    VOID HumanGameView::ToggleConsole(VOID)
    {
        tbd::gui::GuiConsole* c = GetConsole();
        c->VSetActive(!c->VIsActive());
    }

    tbd::gui::GuiConsole* HumanGameView::GetConsole(VOID)
    {
        return (tbd::gui::GuiConsole*)m_pGui->GetComponent("console");
    }

    VOID HumanGameView::VOnUpdate(ULONG deltaMillis)
    {
        switch(app::g_pApp->GetLogic()->GetGameState())
        {
        case tbd::eRunning :
            {
                m_pSceneGraph->OnUpdate(deltaMillis);
                m_pGui->VUpdate(deltaMillis);
            } break;
        case tbd::eLoadingLevel :
            {
                //todo
            } break;

        case tbd::ePause : 
            {

            } break;
        }
    }

    VOID HumanGameView::AddScene(tbd::RenderScreen* screen)
    {
        m_scenes.push_back(screen);
        screen->VOnRestore();
        if(!m_currentScene)
        {
            ActivateScene(screen->VGetName());
        }
    }

    tbd::RenderScreen* HumanGameView::GetSceneByName(LPCSTR name)
    {
        TBD_FOR(m_scenes)
        {
            if((*it)->VGetName() == std::string(name))
            {
                return *it;
            }
        }
        return NULL;
    }

    VOID HumanGameView::ActivateScene(LPCSTR name)
    {
        TBD_FOR(m_scenes)
        {
            if((*it)->VGetName() == std::string(name))
            {
                m_currentScene = *it;
                m_currentScene->GetSettings()->VOnActivate();
            }
        }
    }

    HRESULT HumanGameView::VOnRestore()
    {
        GetRenderer()->VOnRestore();

        TBD_FOR(m_scenes)
        {
            (*it)->VOnRestore();
        }

        TBD_FOR(m_screenElements)
        {
            (*it)->VOnRestore();
        }

        if(m_pGui)
        {
            m_pGui->VOnRestore();
        }

        m_pSceneGraph->OnRestore();

        GetRenderer()->VSetProjectionTransform(m_pSceneGraph->GetCamera()->GetProjection(), m_pSceneGraph->GetCamera()->GetFar());
        GetRenderer()->VSetViewTransform(m_pSceneGraph->GetCamera()->GetView(), m_pSceneGraph->GetCamera()->GetIView(), m_pSceneGraph->GetCamera()->GetEyePos());

        return S_OK;
    }

    VOID HumanGameView::VOnAttach(UINT viewId, std::shared_ptr<tbd::Actor> actor) 
    {
        m_pSceneGraph = new tbd::SceneGraph();

        event::EventListener listener = fastdelegate::MakeDelegate(this, &HumanGameView::ActorMovedDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::ActorMovedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &HumanGameView::NewComponentDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::NewComponentCreatedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &HumanGameView::LoadingLevelDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::LoadingLevelEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &HumanGameView::LevelLoadedDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::LevelLoadedEvent::TYPE);

        listener = fastdelegate::MakeDelegate(this, &HumanGameView::DeleteActorDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::DeleteActorEvent::TYPE);

        ADD_EVENT_LISTENER(this, &HumanGameView::SetParentDelegate, event::SetParentActorEvent::TYPE);

        m_pSceneGraph->SetCamera(std::shared_ptr<util::ICamera>(new util::FPSCamera(GetRenderer()->VGetWidth(), GetRenderer()->VGetHeight(), 1e-2f, 1e3f)));

        GetRenderer()->VSetProjectionTransform(m_pSceneGraph->GetCamera()->GetProjection(), m_pSceneGraph->GetCamera()->GetFar());
        GetRenderer()->VSetViewTransform(m_pSceneGraph->GetCamera()->GetView(), m_pSceneGraph->GetCamera()->GetIView(), m_pSceneGraph->GetCamera()->GetEyePos());

        m_picker = new ActorPicker();

        m_picker->VCreate();

        GetVRamManager()->RegisterHandleCreator("ParticleQuadGeometry", new tbd::ParticleQuadGeometryHandleCreator()); //TODO not needed anymore

        m_pParticleManager = new tbd::ParticleManager();

        tbd::gui::GuiConsole* console = new tbd::gui::GuiConsole();

        tbd::Dimension dim;
        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width;
        dim.h = (INT)(d3d::g_height * 0.4f);
        console->VSetAlpha(0.5f);
        console->VSetBackgroundColor(0.25f,0.25f,0.25f);
        console->VSetDimension(dim);

        m_pGui = new tbd::gui::D3D_GUI();

        m_pGui->VOnRestore();

        m_pGui->AddComponent("console", console);

        console->VSetActive(FALSE);

        app::PostInitMessage("Loading Sound ...");
        m_pSoundSystem = new tbd::DirectSoundSystem();

        if(!m_pSoundSystem->VInit())
        {
            LOG_WARNING("Failed to create SoundSystem");
        }

        listener = fastdelegate::MakeDelegate(&m_soundEngine, &tbd::SoundEngine::CollisionEventDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::CollisionEvent::TYPE);

        listener = fastdelegate::MakeDelegate(&m_soundEngine, &tbd::SoundEngine::NewComponentDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::NewComponentCreatedEvent::TYPE);

        IGameView::VOnAttach(viewId, actor);
    }

    VOID HumanGameView::VPostRender(VOID)
    {
        //m_picker->VPostRender();
    }

    VOID HumanGameView::VOnRender(DOUBLE time, FLOAT elapsedTime) 
    {        
        app::g_pApp->GetRenderer()->ClearBackbuffer();
        switch(app::g_pApp->GetLogic()->GetGameState())
        {
        case tbd::eRunning : 
            {
                m_currentScene->VDraw();
                TBD_FOR(m_screenElements)
                {
                    (*it)->VDraw();
                }
                m_pGui->VDraw();

            } break;
        case tbd::eLoadingLevel : 
            {
                //TODO: GUI Label class maybe?
                app::g_pApp->GetHumanView()->GetRenderer()->VPreRender();
                std::string text("Loading Level");
                app::g_pApp->GetFontManager()->RenderText(text, 0.45f, 0.5f);
                text = "";
                for(UCHAR c = 0; c < app::g_pApp->GetLogic()->GetLevelLoadProgress() * 100; ++c)
                {
                    text += ".";
                }
                app::g_pApp->GetFontManager()->RenderText(text, 0.30f, 0.6f);
                app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();
            } break;
        case tbd::ePause : 
            {
                app::g_pApp->GetHumanView()->GetRenderer()->VPreRender();
                app::g_pApp->GetFontManager()->RenderText("Paused", 0.01f, 0.1f);
                app::g_pApp->GetHumanView()->GetRenderer()->VPostRender();
            } break;
        }
    }

    VOID HumanGameView::VSetTarget(std::shared_ptr<tbd::Actor> actor) 
    {
        if(actor)
        {
            std::shared_ptr<tbd::CameraComponent> comp = actor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();

            if(comp)
            {
                std::shared_ptr<util::ICamera> cam = comp->GetCamera();

                std::shared_ptr<tbd::TransformComponent> transComp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();

                if(!transComp)
                {
                    LOG_CRITICAL_ERROR("Camera actor has no TransformComponent"); //TODO
                }
      
                cam->MoveToPosition(transComp->GetTransformation()->GetTranslation());
   
                IGameView::VSetTarget(actor);

                GetRenderer()->VSetViewTransform(cam->GetView(), cam->GetIView(), cam->GetEyePos());
                GetRenderer()->VSetProjectionTransform(cam->GetProjection(), cam->GetFar() - cam->GetNear());

                m_pSceneGraph->SetCamera(cam);                
            }
            else
            {
                LOG_CRITICAL_ERROR("fix only camras as target");
            }
        }
    }

    HumanGameView::~HumanGameView(VOID) 
    {
        if(event::IEventManager::Get())
        {
            event::EventListener listener = fastdelegate::MakeDelegate(this, &HumanGameView::ActorMovedDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorMovedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::NewComponentDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::NewComponentCreatedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::LevelLoadedDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::LevelLoadedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::LoadingLevelDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::LoadingLevelEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::DeleteActorDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::DeleteActorEvent::TYPE);

            listener = fastdelegate::MakeDelegate(&m_soundEngine, &tbd::SoundEngine::CollisionEventDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::CollisionEvent::TYPE);

            listener = fastdelegate::MakeDelegate(&m_soundEngine, &tbd::SoundEngine::NewComponentDelegate);
            event::IEventManager::Get()->VRemoveEventListener(listener, event::NewComponentCreatedEvent::TYPE);

            REMOVE_EVENT_LISTENER(this, &HumanGameView::SetParentDelegate, event::SetParentActorEvent::TYPE);
        }

        TBD_FOR(m_screenElements)
        {
            SAFE_DELETE(*it);
        }

        TBD_FOR(m_scenes)
        {
            SAFE_DELETE(*it);
        }

        SAFE_DELETE(m_pSoundSystem);
        
        SAFE_DELETE(m_pGui);

        SAFE_DELETE(m_pParticleManager);

        SAFE_DELETE(m_picker);

        SAFE_DELETE(m_pSceneGraph);
    }
}