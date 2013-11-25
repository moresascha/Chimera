#include "GameView.h"
#include "Components.h"
#include "Event.h"
#include "VRamManager.h"
#include "ScreenElement.h"
#include "GraphicsSettings.h"
#include "SceneGraph.h"
#include "SceneNode.h"

namespace chimera
{
    //default creator

    ISceneNode* CreateRenderNode(IHumanView* gw, IActor* actor)
    {
        RenderComponent* comp = GetActorCompnent<RenderComponent>(actor, CM_CMP_RENDERING);
        ISceneNode* mn;
        if(comp->m_type == std::string("skydome"))
        {            
            return new SkyDomeNode(actor->GetId(), comp->m_resource);
        }
        else if(comp->m_geo)
        {
            mn = new GeometryNode(actor->GetId(), std::move(comp->m_geo));
        }
        else if(comp->m_vmemInstances)
        {
            mn = new InstancedMeshNode(actor->GetId(), comp->m_vmemInstances, comp->m_resource);
        }
        else
        {
            if(comp->m_resource == CMResource())
            {
                LOG_CRITICAL_ERROR("no meshfile specified!");
            }
			mn = new MeshNode(actor->GetId(), comp->m_resource);
        }

        if(comp->m_drawType == "wire")
        {
            RenderPath rp = mn->VGetRenderPaths();
            rp ^= CM_RENDERPATH_ALBEDO;
            rp ^= CM_RENDERPATH_SHADOWMAP;
            rp |= CM_RENDERPATH_ALBEDO_WIRE;
            mn->VSetRenderPaths(rp);
        }

        return mn;

    }

    ISceneNode* CreateInstancedMeshNode(IHumanView* gw, IActor* actor)
    {
        RenderComponent* comp = GetActorCompnent<RenderComponent>(actor, CM_CMP_RENDERING);
        return new MeshNode(actor->GetId(), comp->m_resource);
    }

    ISceneNode* CreateLightNode(IHumanView* gw, IActor* actor)
    {
        LightComponent* comp = GetActorCompnent<LightComponent>(actor, CM_CMP_LIGHT);
        return new MeshNode(actor->GetId(), "");
    }

    //default creator end

    HumanGameView::HumanGameView(VOID)
    {
        m_loadingDots = NULL;
        m_picker = NULL; 
        m_pCurrentScene = NULL;
        m_pSceneGraph = NULL;
        m_pSoundSystem = NULL;
        m_pSoundEngine = NULL;
        m_pRenderer = NULL;
        m_pEffectFactory = NULL;
        m_pFontManager = NULL;
        m_pGuiFactroy = NULL;
    }

    BOOL HumanGameView::VInitialise(FactoryPtr* facts)
    {
        m_pGraphicsFactory = FindAndCopyFactory<IGraphicsFactory>(facts, CM_FACTORY_GFX);
        if(m_pGraphicsFactory == NULL)
        {
            return FALSE;
        }

        m_pRenderer = m_pGraphicsFactory->VCreateRenderer();

        IVRamManagerFactory* vramF = FindFactory<IVRamManagerFactory>(facts, CM_FACTORY_VRAM);
        RETURN_IF_FAILED(vramF);

        m_pVramManager = vramF->VCreateVRamManager();

        IEffectFactoryFactory* eff = FindFactory<IEffectFactoryFactory>(facts, CM_FACTORY_EFFECT);
        RETURN_IF_FAILED(eff);

        m_pEffectFactory = eff->VCreateEffectFactroy();

        IFontFactory* ff = FindFactory<IFontFactory>(facts, CM_FACTORY_FONT);
        RETURN_IF_FAILED(ff);

        m_pGuiFactroy = FindAndCopyFactory<IGuiFactory>(facts, CM_FACTORY_GUI);
        RETURN_IF_FAILED(m_pGuiFactroy);

        m_pFontManager = ff->VCreateFontManager();
        m_pFontManager->VAddFont(std::string("default"), ff->VCreateFont());
        m_pFontManager->VSetFontRenderer(ff->VCreateFontRenderer());

        m_pSceneGraph = new SceneGraph();

        ADD_EVENT_LISTENER(this, &HumanGameView::ActorMovedDelegate, CM_EVENT_ACTOR_MOVED);

        ADD_EVENT_LISTENER(this, &HumanGameView::NewComponentDelegate, CM_EVENT_COMPONENT_CREATED);

        ADD_EVENT_LISTENER((VRamManager*)m_pVramManager, &VRamManager::OnResourceChanged, CM_EVENT_RESOURCE_CHANGED);

        VAddSceneNodeCreator(CreateRenderNode, CM_CMP_RENDERING);

        return TRUE;
    }

    BOOL HumanGameView::VOnRestore()
    {
        VGetRenderer()->VOnRestore();

        std::string fontFile = CmGetApp()->VGetConfig()->VGetString("sFontPath") + std::string("font_16.fnt");
        VGetFontManager()->VGetCurrentFont()->VCreate(fontFile);

        VGetFontManager()->VOnRestore();

        TBD_FOR(m_scenes)
        {
            (*it)->VOnRestore();
        }

        TBD_FOR(m_screenElements)
        {
            (*it)->VOnRestore();
        }

        /*if(m_pGui)
        {
            m_pGui->VOnRestore();
        } */

        m_pSceneGraph->VOnRestore();

        if(m_pSceneGraph->VGetCamera())
        {
            m_pSceneGraph->VGetCamera()->SetAspect(VGetRenderer()->VGetWidth(), VGetRenderer()->VGetHeight());
            VGetRenderer()->VSetProjectionTransform(m_pSceneGraph->VGetCamera()->GetProjection(), m_pSceneGraph->VGetCamera()->GetFar());
            VGetRenderer()->VSetViewTransform(m_pSceneGraph->VGetCamera()->GetView(), m_pSceneGraph->VGetCamera()->GetIView(), m_pSceneGraph->VGetCamera()->GetEyePos());
        }

        return TRUE;
    }

    IRenderer* HumanGameView::VGetRenderer(VOID)
    {
        return m_pRenderer.get();
    }

    VOID HumanGameView::VOnResize(UINT w, UINT h)
    {
       /* if(chimera::g_pSwapChain) 
        {
            chimera::Resize(w, h);
            VOnRestore();
            BOOL fullscreen = chimera::GetFullscreenState();
            chimera::g_pApp->GetInputHandler()->VSetCurserOffsets(fullscreen ? 0 : 8 , fullscreen ? 0 : 30); //todo, get systemmetrics
        } */
        m_pRenderer->VResize(w, h);

        VOnRestore();

        std::shared_ptr<ActorMovedEvent> movedEvent(new ActorMovedEvent(m_actor));
        ActorMovedDelegate(movedEvent);
    }

    VOID HumanGameView::ActorMovedDelegate(IEventPtr eventData)  
    {
        std::shared_ptr<ActorMovedEvent> movedEvent = std::static_pointer_cast<ActorMovedEvent>(eventData);

        if(movedEvent->m_actor->GetId() == m_actor->GetId())
        {
            TransformComponent* comp = GetActorCompnent<TransformComponent>(m_actor, CM_CMP_TRANSFORM);
            CONST util::Vec3& trans = comp->GetTransformation()->GetTranslation();

            m_pSceneGraph->VGetCamera()->SetRotation(comp->m_phi, comp->m_theta);

            m_pSceneGraph->VGetCamera()->MoveToPosition(trans);

            VGetRenderer()->VSetViewTransform(m_pSceneGraph->VGetCamera()->GetView(), m_pSceneGraph->VGetCamera()->GetIView(), m_pSceneGraph->VGetCamera()->GetEyePos());
        }
    }

    VOID HumanGameView::NewComponentDelegate(IEventPtr pEventData) 
    {
        std::shared_ptr<NewComponentCreatedEvent> pCastEventData = std::static_pointer_cast<NewComponentCreatedEvent>(pEventData);
        IActor* actor = chimera::CmGetApp()->VGetLogic()->VFindActor(pCastEventData->m_actorId);

        auto it = m_nodeCreators.find(pCastEventData->m_id);
        if(it == m_nodeCreators.end())
        {
            return;
        }

        std::unique_ptr<ISceneNode> node(it->second(this, actor));

        if(node)
        {
            node->VOnRestore(m_pSceneGraph);

            m_pSceneGraph->VAddChild(pCastEventData->m_actorId, std::move(node));
        }
        else
        {
            LOG_CRITICAL_ERROR_A("%s", "hmm");
        }

        //if(pCastEventData->m_id == CM_CMP_RENDERING)
        {
            //
            /*
            if(comp->m_sceneNode)
            {
                node = comp->m_sceneNode;
                comp->m_sceneNode = NULL; //this prevents memory leaks
                node->VSetActor(pCastEventData->m_actorId);
            }
            else if(comp->m_type == "anchor")
            {
                node = std::shared_ptr<chimera::AnchorNode>(
                    new chimera::AnchorNode(eSPHERE, actor->GetId(), comp->m_info.c_str(), comp->m_anchorRadius, comp->m_drawType == "solid" ? eSolid : eWire));
            }
            else if(comp->m_type == "skydome")
            {
                node = std::shared_ptr<chimera::SkyDomeNode>(new chimera::SkyDomeNode(actor->GetId(), comp->m_info));
            }
            else if(!comp->m_instances.empty())
            {
                node = std::shared_ptr<chimera::InstancedMeshNode>(new chimera::InstancedMeshNode(actor->GetId(), comp->m_meshFile));
            }
            else
            {
                
            }*/

            //
            
        }

        /*
        else if(pCastEventData->m_id == chimera::LightComponent::COMPONENT_ID)
        {
            std::shared_ptr<chimera::LightComponent> comp = actor->GetComponent<chimera::LightComponent>(chimera::LightComponent::COMPONENT_ID).lock();

            std::shared_ptr<chimera::ISceneNode> node;

            if(comp->m_type == "point")
            {
                node = std::shared_ptr<chimera::PointlightNode>(new chimera::PointlightNode(actor->GetId()));
            }
            else if(comp->m_type == "spot")
            {
                node = std::shared_ptr<chimera::SpotlightNode>(new chimera::SpotlightNode(actor->GetId()));
            }
            else
            {
                LOG_CRITICAL_ERROR_A("Unknown lighttype: %s", comp->m_type.c_str());
            }

            m_pSceneGraph->VAddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);

            comp->VSetHandled();
        }

        else if(pCastEventData->m_id == chimera::ParticleComponent::COMPONENT_ID)
        {
            std::shared_ptr<chimera::ParticleComponent> comp = actor->GetComponent<chimera::ParticleComponent>(chimera::ParticleComponent::COMPONENT_ID).lock();

            std::shared_ptr<chimera::ParticleNode> node = std::shared_ptr<chimera::ParticleNode>(new chimera::ParticleNode(actor->GetId()));

            m_pSceneGraph->VAddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);

            comp->VSetHandled();
        }
        else if(pCastEventData->m_id == chimera::CameraComponent::COMPONENT_ID)
        {
            std::shared_ptr<chimera::SceneNode> node = std::shared_ptr<chimera::CameraNode>(new chimera::CameraNode(actor->GetId()));

            m_pSceneGraph->VAddChild(pCastEventData->m_actorId, node);

            node->VOnRestore(m_pSceneGraph);
        }
        */
    }

    VOID HumanGameView::VAddSceneNodeCreator(SceneNodeCreator nc, ComponentId cmpid)
    {
         auto f = m_nodeCreators.find(cmpid);
         if(f == m_nodeCreators.end())
         {
             m_nodeCreators.insert(std::pair<ComponentId, SceneNodeCreator>(cmpid, nc));
         }
         else
         {
             LOG_CRITICAL_ERROR("CMPID already has a creator isntalled!");
         }
    }

    VOID HumanGameView::VRemoveSceneNodeCreator(ComponentId cmpid)
    {
        auto f = m_nodeCreators.find(cmpid);
        if(f != m_nodeCreators.end())
        {
            m_nodeCreators.erase(cmpid);
        }
    }

    VOID HumanGameView::VAddScreenElement(std::unique_ptr<IScreenElement> element)
    {
        element->VOnRestore();
        m_screenElements.push_back(std::move(element));
    }

    IScreenElement* HumanGameView::VGetScreenElementByName(LPCSTR name)
    {
        TBD_FOR(m_screenElements)
        {
            if((*it)->VGetName() == std::string(name))
            {
                return (*it).get();
            }
        }
        return NULL;
    }

    VOID HumanGameView::DeleteActorDelegate(IEventPtr pEventData)
    {
        std::shared_ptr<ActorDeletedEvent> pCastEventData = std::static_pointer_cast<ActorDeletedEvent>(pEventData);

        m_pSceneGraph->VRemoveChild(pCastEventData->m_id);
    }

    VOID HumanGameView::LevelLoadedDelegate(IEventPtr pEventData)
    {

    }

    VOID HumanGameView::LoadingLevelDelegate(IEventPtr pEventData)
    {
        
    }

    VOID HumanGameView::SetParentDelegate(IEventPtr pEventData)
    {
        //if(pEventData->VGetEventType() == event::SetParentActorEvent::TYPE) needed?
        {
            std::shared_ptr<SetParentActorEvent> e = std::static_pointer_cast<SetParentActorEvent>(pEventData);
            ActorId actor = e->m_actor;
            ActorId parent = e->m_parentActor;

            ISceneNode* p = m_pSceneGraph->VFindActorNode(parent);
            ISceneNode* a = m_pSceneGraph->VFindActorNode(actor);
            if(p)
            {
                //p->VAddChild(a);
                LOG_CRITICAL_ERROR("ASD");
            }
            else
            {
                LOG_CRITICAL_ERROR("SetParentFailed - No Actor found");
            }
        }
    }

    /*VOID HumanGameView::ToggleConsole(VOID)
    {
        chimera::gui::GuiConsole* c = GetConsole();
        c->VSetActive(!c->VIsActive());
    }

    chimera::gui::GuiConsole* HumanGameView::GetConsole(VOID)
    {
        return (chimera::gui::GuiConsole*)m_pGui->GetComponent("console");
    } */

    VOID HumanGameView::VOnUpdate(ULONG deltaMillis)
    {
        switch(CmGetApp()->VGetLogic()->VGetGameState())
        {
        case CM_STATE_RUNNING :
            {
                m_pSceneGraph->VOnUpdate(deltaMillis);
                m_pVramManager->VUpdate(deltaMillis);

                TBD_FOR(m_screenElements)
                {
                    (*it)->VUpdate(deltaMillis);
                }

            } break;
        case CM_STATE_LOADING :
            {
                //todo
            } break;

        case CM_STATE_PAUSED : 
            {

            } break;
        }
    }

    VOID HumanGameView::VAddScene(std::unique_ptr<IRenderScreen> screen)
    {
        screen->VOnRestore();
        std::string name(screen->VGetName());
        m_scenes.push_back(std::move(screen));
        if(!m_pCurrentScene)
        {
            VActivateScene(name.c_str());
        }
    }

    IRenderScreen* HumanGameView::VGetSceneByName(LPCSTR name)
    {
        TBD_FOR(m_scenes)
        {
            if((*it)->VGetName() == std::string(name))
            {
                return (*it).get();
            }
        }
        return NULL;
    }

    VOID HumanGameView::VActivateScene(LPCSTR name)
    {
        TBD_FOR(m_scenes)
        {
            if((*it)->VGetName() == std::string(name))
            {
                m_pCurrentScene = (*it).get();
                m_pCurrentScene->VGetSettings()->VOnActivate();
            }
        }
    }

    VOID HumanGameView::VOnAttach(UINT viewId, IActor* actor) 
    {
        IView::VOnAttach(viewId, actor);
    }

    VOID HumanGameView::VPostRender(VOID)
    {
        //m_picker->VPostRender();
    }

    VOID HumanGameView::VOnRender(VOID) 
    {        
        m_pCurrentScene->VDraw();
        TBD_FOR(m_screenElements)
        {
            (*it)->VDraw();
        }
        /*
        switch(chimera::CmGetApp()->VGetLogic()->VGetGameState())
        {
        case chimera::eRunning : 
            {
                m_pCurrentScene->VDraw();
               // m_pGui->VDraw();

            } break;
        case chimera::eLoadingLevel : 
            {
                VGetRenderer()->VClearAndBindBackBuffer();
                //TODO: GUI Label class maybe?
                VGetRenderer()->VPreRender();
                std::string text("Loading Level");
                VGetRenderer()->VPostRender();
                text = "";
                /*for(UCHAR c = 0; c < chimera::CmGetApp()->VGetLogic()->VGetLevelLoadProgress() * 100; ++c)
                {
                    text += ".";
                } */
                /*chimera::CmGetApp()->VGetFontManager()->VRenderText(text, 0.30f, 0.6f);
                VGetRenderer()->VPostRender();
            } break;
        case chimera::ePause : 
            {
                VGetRenderer()->VClearAndBindBackBuffer();
                VGetRenderer()->VPreRender();
                chimera::CmGetApp()->VGetFontManager()->VRenderText("Paused", 0.01f, 0.1f);
                VGetRenderer()->VPostRender();
            } break;
        } */

        VGetRenderer()->VPresent();  
    }

    VOID HumanGameView::VSetTarget(IActor* actor) 
    {
        if(actor)
        {
            CameraComponent* comp = GetActorCompnent<CameraComponent>(actor, CM_CMP_CAMERA);

            if(comp)
            {
                std::shared_ptr<ICamera> cam = comp->GetCamera();

                TransformComponent* transComp = GetActorCompnent<TransformComponent>(actor, CM_CMP_TRANSFORM);

                if(!transComp)
                {
                    LOG_CRITICAL_ERROR("Camera actor has no TransformComponent"); //TODO
                }
      
                cam->MoveToPosition(transComp->GetTransformation()->GetTranslation());
   
                IView::VSetTarget(actor);

                VGetRenderer()->VSetViewTransform(cam->GetView(), cam->GetIView(), cam->GetEyePos());
                VGetRenderer()->VSetProjectionTransform(cam->GetProjection(), cam->GetFar() - cam->GetNear());

                m_pSceneGraph->VSetCamera(cam);

                std::shared_ptr<ActorMovedEvent> movedEvent(new ActorMovedEvent(m_actor));
                ActorMovedDelegate(movedEvent);
            }
            else
            {
                LOG_CRITICAL_ERROR("fix only camras as target");
            }
        }
    }

    HumanGameView::~HumanGameView(VOID) 
    {
        REMOVE_EVENT_LISTENER(this, &HumanGameView::ActorMovedDelegate, CM_EVENT_ACTOR_MOVED);

        REMOVE_EVENT_LISTENER(this, &HumanGameView::NewComponentDelegate, CM_EVENT_COMPONENT_CREATED);

        REMOVE_EVENT_LISTENER((VRamManager*)m_pVramManager, &VRamManager::OnResourceChanged, CM_EVENT_RESOURCE_CHANGED);
        /*
        if(chimera::IEventManager::Get())
        {
            chimera::EventListener listener = fastdelegate::MakeDelegate(this, &HumanGameView::ActorMovedDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::ActorMovedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::NewComponentDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::NewComponentCreatedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::LevelLoadedDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::LevelLoadedEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::LoadingLevelDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::LoadingLevelEvent::TYPE);

            listener = fastdelegate::MakeDelegate(this, &HumanGameView::DeleteActorDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::DeleteActorEvent::TYPE);

            listener = fastdelegate::MakeDelegate(&m_soundEngine, &chimera::SoundEngine::CollisionEventDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::CollisionEvent::TYPE);

            listener = fastdelegate::MakeDelegate(&m_soundEngine, &chimera::SoundEngine::NewComponentDelegate);
            chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::NewComponentCreatedEvent::TYPE);

            REMOVE_EVENT_LISTENER(this, &HumanGameView::SetParentDelegate, chimera::SetParentActorEvent::TYPE);
        } */

        TBD_FOR(m_screenElements)
        {
            it->reset();
        }

        TBD_FOR(m_scenes)
        {
            it->reset();
        }

        SAFE_DELETE(m_pSoundSystem);
        
        //SAFE_DELETE(m_pGui);

        SAFE_DELETE(m_pGuiFactroy);

        SAFE_DELETE(m_picker);

        SAFE_DELETE(m_pSceneGraph);

        SAFE_DELETE(m_pEffectFactory);

        SAFE_DELETE(m_pFontManager);

        SAFE_DELETE(m_pVramManager);

        SAFE_DELETE(m_pGraphicsFactory);

        //always last
        SAFE_RESET(m_pRenderer);
    }
}