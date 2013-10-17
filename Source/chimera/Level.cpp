#include "Level.h"
#include "Resources.h"
#include "GameApp.h"
#include "tinyxml2.h"
#include "GameLogic.h"
#include "Components.h"
#include "EventManager.h"
#include "Event.h"
#include "CudaTransformationNode.h"
#include "Cudah.h"
#include "GeometryFactory.h"
#include "Process.h"
#include "CameraTracking.h"
#include <fstream>

namespace chimera
{
    VOID CreateStaticPlane(BaseLevel* level)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");

        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = "plane.obj";

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "static";
        physicComponent->m_shapeStyle = "plane";

        level->VAddActor(desc);
    }

    std::shared_ptr<chimera::Actor> CreateCube(CONST util::Vec3& pos, BaseLevel* level)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = "box.obj";

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "kinematic";
        physicComponent->m_shapeStyle = "box";
        physicComponent->m_radius = 1;

        desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        return level->VAddActor(desc);
    }

    std::shared_ptr<chimera::Actor> CreateSphere(CONST util::Vec3& pos, BaseLevel* level)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = "sphere.obj";

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "kinematic";
        physicComponent->m_shapeStyle = "sphere";
        physicComponent->m_radius = 1;

        return level->VAddActor(desc);
    }

    std::shared_ptr<chimera::Actor> CreateMesh(CONST util::Vec3& pos, BaseLevel* level, LPCSTR meshFile, LPCSTR physicShape, LPCSTR material)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = meshFile;

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = material;
        physicComponent->m_shapeStyle = physicShape;
        physicComponent->m_radius = 1;

        return level->VAddActor(desc);
    }

    std::shared_ptr<chimera::Actor> CreatePointlight(util::Vec3& pos, BaseLevel* level, util::Vec4& color, FLOAT radius, FLOAT intensity)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::LightComponent* lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "point";
        lightComponent->m_color = color;
        lightComponent->m_intensity = intensity;

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);
        comp->GetTransformation()->SetScale(radius);

        std::shared_ptr<chimera::Actor> actor = level->VAddActor(desc);

        return actor;
    }

    BaseLevel::BaseLevel(CONST std::string& file, chimera::ActorFactory* factory) : m_file(file), m_name("unnamed"), m_pActorFactory(factory)
    {
        chimera::EventListener listener = fastdelegate::MakeDelegate(this, &XMLLevel::ActorCreatedDelegate);
        chimera::IEventManager::Get()->VAddEventListener(listener, chimera::ActorCreatedEvent::TYPE);
    }

    std::shared_ptr<chimera::Actor> BaseLevel::VFindActor(ActorId id)
    {
        auto it = m_actors.find(id);
        if(it == m_actors.end()) 
        {
            return NULL;
        }
        return it->second;
    }

    UINT BaseLevel::VGetActorsCount(VOID)
    {
        return (UINT)m_actors.size();
    }

    VOID BaseLevel::VRemoveActor(ActorId id)
    {
        m_actors.erase(id);
    }

    VOID BaseLevel::ActorCreatedDelegate(chimera::IEventPtr eventData)
    {
        std::shared_ptr<chimera::ActorCreatedEvent> data = std::static_pointer_cast<chimera::ActorCreatedEvent>(eventData);
        auto it = std::find(m_idsToLoad.begin(), m_idsToLoad.end(), data->m_id);
        if(it != m_idsToLoad.end())
        {
            m_idsToLoad.erase(it);
            if(m_idsToLoad.empty())
            {
                chimera::IEventPtr levelLoadedEvent(new chimera::LevelLoadedEvent(std::string(VGetName())));
                chimera::IEventManager::Get()->VQueueEventThreadSave(levelLoadedEvent);
            }
        }
    }

    std::shared_ptr<chimera::Actor> BaseLevel::VAddActor(ActorDescription& desc)
    {
        std::shared_ptr<chimera::Actor> actor = m_pActorFactory->CreateActor(desc);
        this->m_actors[actor->GetId()] = actor;
        this->m_idsToLoad.push_back(actor->GetId());
        return actor;
    }

    FLOAT BaseLevel::VGetLoadingProgress(VOID)
    {
        return 1.0f - m_idsToLoad.size() / (FLOAT)VGetActorsCount();
    }

    VOID BaseLevel::VUnload(VOID)
    {
        for(auto it = m_actors.begin(); it != m_actors.end(); ++it)
        {
            chimera::IEventPtr deletActorEvent(new chimera::DeleteActorEvent(it->first));
            chimera::IEventManager::Get()->VQueueEvent(deletActorEvent);
        }
    }

    BaseLevel::~BaseLevel(VOID)
    {
        VUnload();
        chimera::EventListener listener = fastdelegate::MakeDelegate(this, &XMLLevel::ActorCreatedDelegate);
        chimera::IEventManager::Get()->VRemoveEventListener(listener, chimera::ActorCreatedEvent::TYPE);
    }

    XMLLevel::XMLLevel(CONST std::string& file, chimera::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    std::shared_ptr<chimera::Actor> XMLLevel::VAddActor(tinyxml2::XMLElement* pNode)
    {
        if(!pNode) 
        {
            LOG_CRITICAL_ERROR("TiXmlElement cant be NULL");
        }
        std::shared_ptr<chimera::Actor> actor = NULL;
        std::shared_ptr<chimera::Actor> parent = NULL;
        std::vector<std::shared_ptr<chimera::Actor>> actors;
        parent = m_pActorFactory->CreateActor(pNode, actors);
        TBD_FOR(actors)
        {
            actor = *it;
            this->m_actors[actor->GetId()] = actor;
            this->m_idsToLoad.push_back(actor->GetId());
        }
        return parent;
    }

    BOOL XMLLevel::VLoad(BOOL block)
    {
        tinyxml2::XMLDocument doc;

        chimera::CMResource r(chimera::g_pApp->GetConfig()->GetString("sLevelPath") + VGetFile().c_str());

        std::shared_ptr<chimera::ResHandle> handle = chimera::g_pApp->GetCache()->GetHandle(r);

        if(!handle)
        {
            return FALSE;
        }

        doc.Parse((CHAR*)handle->Buffer());

        tinyxml2::XMLElement* root = doc.RootElement();
        RETURN_IF_FAILED(root);

        for(tinyxml2::XMLElement* pNode = root->FirstChildElement(); pNode; pNode = pNode->NextSiblingElement())
        {
            VAddActor(pNode);
        }

        return TRUE;
    }

    BOOL XMLLevel::VSave(LPCSTR file)
    {
        SaveXMLLevel save;
        return save.VSaveLevel(this, file);
    }

    RandomLevel::RandomLevel(CONST std::string& file, chimera::ActorFactory* factory) : BaseLevel(file, factory)
    {
    }

    /*
    class SpawProcess : public proc::Process 
    {
        VOID VOnUpdate(ULONG deltaMillis) 
        {
            static ULONG time = 0, accu = 0;
            static LONG startTime = 0;
            
            time += deltaMillis;
            accu += deltaMillis;
            startTime += deltaMillis;
            if(startTime < 50000 && time < 100)
            {
                return;
            }
            time = 0;
            int count = 0;

            while(count++ < 200)
            {
                tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

                FLOAT scale = 2;
                FLOAT x = 2 * (rand() / (FLOAT)RAND_MAX - 0.5f);
                FLOAT z = 2 * (rand() / (FLOAT)RAND_MAX - 0.5f);
                FLOAT dy = rand() / (FLOAT)RAND_MAX;

                std::shared_ptr<tbd::TransformComponent> comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
                comp->GetTransformation()->SetTranslate(scale * x, 40 + 10 * dy, 20 + scale * z);

                FLOAT disDas = 1;//rand() / (FLOAT)RAND_MAX;

                std::shared_ptr<tbd::RenderComponent> renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
                renderComp->m_meshFile = disDas < 0.5 ? "box.obj" : "sphere.obj";

                std::shared_ptr<tbd::PhysicComponent> physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
                physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
                physicComponent->m_material = disDas < 0.5 ? "dynamic" : "bouncy";
                physicComponent->m_shapeType = disDas < 0.5 ? "box" : "sphere";
                physicComponent->m_radius = 1;

                desc->AddComponent<tbd::PickableComponent>("PickableComponent");

                event::IEventPtr createActorEvent(new event::CreateActorEvent(desc));
                event::IEventManager::Get()->VQueueEvent(createActorEvent);
            }

            //if(accu > 20000) 
            {
                this->Succeed();
            }
        }
    };*/

    VOID TriggerDelegate(chimera::IEventPtr data)
    {
        std::shared_ptr<chimera::TriggerEvent> te = std::static_pointer_cast<chimera::TriggerEvent>(data);
        ActorId trigger = te->m_triggerActor;
        QUEUE_EVENT(new chimera::DeleteActorEvent(trigger));
        DEBUG_OUT("On Trigger\n");
    }

    BOOL RandomLevel::VLoad(BOOL block)
    {
        CreateStaticPlane(this);

        for(CHAR i = 0; i < 1; ++i)
        {
            for(CHAR j = 0; j < 1; ++j)
            {
                chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
                desc->AddComponent<chimera::ParticleComponent>("ParticleComponent");
                chimera::TransformComponent* t = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
                t->GetTransformation()->SetTranslate(0,5,0);//3 * i - 3,0.1f, 3 * j - j);
                chimera::PickableComponent* pick = desc->AddComponent<chimera::PickableComponent>("PickableComponent");
                //std::shared_ptr<Actor> actor = VAddActor(desc);
            }
        } 

        
        //CreatePointlight(util::Vec3(0, 7, 0), this, util::Color(1,1,1,1), 40, 0.5f);
        return TRUE;
        FLOAT s = 10;

        for(INT i = 0; i < 0; ++i)
        {
            FLOAT x = -s  + 2 * s * rand() / (FLOAT)RAND_MAX;
            FLOAT y = 3 + 15 * rand() / (FLOAT)RAND_MAX;
            FLOAT z = -s + 2 * s * rand() / (FLOAT)RAND_MAX;
            CreateCube(util::Vec3(x,y,z), this);
        }

                
        /*
        desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
        tbd::TransformComponent* tcomp = desc->AddComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,3,5);

        tbd::RenderComponent* renderCmp = desc->AddComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID);
        renderCmp->m_drawType = "wire";
        renderCmp->m_anchorRadius = 50;
        renderCmp->m_type = "anchor";

        tbd::SoundEmitterComponent* em = desc->AddComponent<tbd::SoundEmitterComponent>(tbd::SoundEmitterComponent::COMPONENT_ID);
        em->m_soundFile = "ambient_dark.wav";
        em->m_loop = TRUE;
        em->m_radius = 50;

        desc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);
        

        VAddActor(desc); */
        
        ActorDescription desc = m_pActorFactory->CreateActorDescription();
        chimera::TransformComponent* tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        chimera::RenderComponent* renderCmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
        VAddActor(desc);

        desc = m_pActorFactory->CreateActorDescription();
        tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,-2,10);
        tcomp->GetTransformation()->SetScale(1.0f);
        renderCmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
        renderCmp->m_resource = "terrain.obj";

        chimera::PhysicComponent* phxCmp = desc->AddComponent<chimera::PhysicComponent>(chimera::PhysicComponent::COMPONENT_ID);
        phxCmp->m_shapeStyle = "static_mesh";
        phxCmp->m_meshFile = "terrain.obj";
        phxCmp->m_material = "static";
        VAddActor(desc);
        
        util::Vec3 p(0,1,10);
        std::shared_ptr<chimera::Actor> a;// = CreateCube(p, this);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");

        util::Vec3 pos(0,1,17);

        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);


        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = "box.obj";

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "static";
        physicComponent->m_shapeStyle = "box";
        physicComponent->m_radius = 1;

        desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        //VAddActor(desc);

        //spotlight thingie

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
        
        desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(0, 2, 2);

        renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = "spottwotest.obj";

        physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_material = "dynamic";
        physicComponent->m_shapeStyle = "sphere";
        physicComponent->m_radius = 1;

        a = VAddActor(desc);

        a->SetName("spotlightsphere");

        desc = m_pActorFactory->CreateActorDescription();

        chimera::LightComponent* lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "spot";
        lightComponent->m_color.x = 1;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;
        lightComponent->m_angle = 55;
        lightComponent->m_intensity = 24;

        tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetScale(50);
        tcomp->GetTransformation()->RotateX(-XM_PIDIV2);
        tcomp->GetTransformation()->Translate(0, 0, 0);

        //desc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        chimera::ParentComponent* pc = desc->AddComponent<chimera::ParentComponent>("ParentComponent");
        pc->m_parentId = a->GetId();

        VAddActor(desc);

        desc = m_pActorFactory->CreateActorDescription();

        lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "spot";
        lightComponent->m_color.x = 1;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;
        lightComponent->m_angle = 55;
        lightComponent->m_intensity = 24;

        tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetScale(50);
        tcomp->GetTransformation()->RotateX(XM_PIDIV2);
        tcomp->GetTransformation()->Translate(0, 0, 0);

        //desc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        pc = desc->AddComponent<chimera::ParentComponent>("ParentComponent");
        pc->m_parentId = a->GetId();

        VAddActor(desc);

        /*desc = m_pActorFactory->CreateActorDescription();
        tcomp = desc->AddComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,10);

        tbd::PhysicComponent* triggerCmp = desc->AddComponent<tbd::PhysicComponent>(tbd::PhysicComponent::COMPONENT_ID);
        triggerCmp->m_radius = 1;
        triggerCmp->m_shapeType = "trigger"; */
        
        /*VAddActor(desc); trigger

        ADD_EVENT_LISTENER_STATIC(&TriggerDelegate, event::TriggerEvent::TYPE);*/

        //std::shared_ptr<tbd::Actor> trackingShotActor = app::g_pApp->GetLogic()->VFindActor("free");//app::g_pApp->GetLogic()->VCreateActor("rendercamera.xml");
        //std::shared_ptr<proc::TrackingShot> ts = std::shared_ptr<proc::TrackingShot>(new proc::TrackingShot(trackingShotActor, TRUE));
        //ts->
        //ts->AddBasePoint(util::Vec3(0,2,-5), util::Vec3(5,10,10), util::Vec3(0,0,0), util::Vec3(0,0,0));
        FLOAT scalee = 30;
        /*
        ts->AddBasePoint(util::Vec3(-scalee,10,0), util::Vec3(0,0,0), 2000);
        ts->AddBasePoint(util::Vec3(0,10,scalee), util::Vec3(0,0,0), 2000);
        ts->AddBasePoint(util::Vec3(+scalee,10,0), util::Vec3(0,0,0), 2000);
        ts->AddBasePoint(util::Vec3(0,10,-scalee), util::Vec3(0,0,0), 2000);

        ts->AddBasePoint(util::Vec3(-scalee,10,0), util::Vec3(0,0,0), 2000);
        ts->AddBasePoint(util::Vec3(0,10,scalee), util::Vec3(0,0,0), 2000);
        ts->AddBasePoint(util::Vec3(+scalee,10,0), util::Vec3(0,0,0), 2000);
        ts->AddBasePoint(util::Vec3(0,10,-scalee), util::Vec3(0,0,0), 2000);
                ts->SetDivisions(100); */
        /*
        INT N = 8;
        ts->SetDivisions(100);

        for(int i = 0; i < N; ++i)
        {
            float s = i / (float)N;
            float freq = 1;
            
            util::Vec3 eye(sin(freq * XM_2PI * s), 5 * (i > N/2 ? s : 1-s), cos(freq * XM_2PI * s));
            util::Vec3 focus(sin(XM_PIDIV2 + freq * XM_2PI * s), 5 - eye.y, cos(XM_PIDIV2 + freq * XM_2PI * s));
            focus.Scale(4);
            eye.Scale(util::Vec3(10,1,10));
            ts->AddBasePoint(eye, focus, 2000);
        } 
        
        //app::g_pApp->GetLogic()->GetProcessManager()->Attach(ts);

        return TRUE;

        INT gridSize = 16;
        INT h = 5;
        INT scale = 10;
        util::Vec3 offset(-scale*gridSize/2.0f, 0, -scale*gridSize/2.0f);
        for(INT i = 0; i < gridSize; ++i)
        {
            for(INT j = 0; j < gridSize; ++j)
            {
                INT px = i;
                INT pz = j;
                for(INT k = 0; k < h; ++k)
                {
                    tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

                    tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
                    comp->GetTransformation()->SetTranslate(offset.x + scale * px, offset.y + 1 + 2*k, offset.z + scale * pz);

                    tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
                    renderComp->m_meshFile = "box.obj";

                    tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
                    physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
                    physicComponent->m_material = "kinematic";
                    physicComponent->m_shapeType = "box";
                    physicComponent->m_radius = 1;

                    if(k == 4)
                    {
                        if(gridSize-1 == i) continue;
                        std::shared_ptr<tbd::Actor> actor = VAddActor(desc);
                        CONST util::Vec3& start = comp->GetTransformation()->GetTranslation();
                        util::Vec3 end = start + util::Vec3(+10, 0, 0);//
                    } 
                    else if(k == 2)
                    {
                        if(gridSize-1 == j) continue;
                        std::shared_ptr<tbd::Actor> actor = VAddActor(desc);
                        CONST util::Vec3& start = comp->GetTransformation()->GetTranslation();
                        util::Vec3 end = start + util::Vec3(0, 0, +10);
                    }
                    else
                    {
                       VAddActor(desc);
                    }
                }
                
                if(i == 0 && j % 2 == 0)
                {
                tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

                tbd::LightComponent* lightComponent = desc->AddComponent<tbd::LightComponent>("LightComponent");
                lightComponent->m_type = "Point";
                lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
                lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
                lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
                lightComponent->m_color.w = 1;

                tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
                comp->GetTransformation()->SetTranslate(offset.x, offset.y + 9, offset.z + scale * pz + scale*0.5f);
                comp->GetTransformation()->SetScale(30.0f);
                //lights
                std::shared_ptr<tbd::Actor> actor = VAddActor(desc);
                CONST util::Vec3& start = comp->GetTransformation()->GetTranslation();
                util::Vec3 end = start + util::Vec3((FLOAT)(gridSize * (scale-1)), 0, 0);
                FLOAT timeLength = 10000;
                FLOAT startTime = rand() / (FLOAT)RAND_MAX * timeLength;
                BOOL a = rand() / (FLOAT)RAND_MAX < 0.5f;
                }
            } 
        }*/

        return TRUE;
    }

    BOOL RandomLevel::VSave(LPCSTR file)
    {
        SaveXMLLevel save;
        return save.VSaveLevel(this, file);
    }
    //---------------

        VOID Trans0(cudah::cuda_buffer buffer, cudah::cuda_buffer staticNormals, cudah::cuda_buffer staticPositions, cudah::cuda_buffer indices, UINT elements, UINT blockSize, ULONG time, cudah::cuda_stream stream)
    {
        /*
        UINT gws = cudah::cudah::GetThreadCount(elements, blockSize);

        sinCosRiddle((VertexData*)buffer->ptr, (float3*)staticNormals->ptr, (float3*)staticPositions->ptr, gws, blockSize, time * 1e-3f, elements, stream->GetPtr());
        //elements = 4;
        blockSize = 768;//4;
        UINT gridWidth = 27; //2
        UINT gridSize = blockSize;// gridWidth * gridWidth;
        UINT vertexStride = 1023 + 1;
        gws = cudah::cudah::GetThreadCount(elements, gridSize);

        //cudaStreamSynchronize(stream->GetPtr());
        
        INT blocksPerRow = (int)ceil(vertexStride / (float)gridWidth);

        INT rest = gridWidth - gridWidth * blocksPerRow % vertexStride;

        INT sharedMem = 3 * gridSize * sizeof(float3);

        comupteNormals((VertexData*)buffer->ptr, (int*)indices->ptr, gws / gridSize, blockSize, gridWidth, gridSize, vertexStride, blocksPerRow, rest, elements, sharedMem, stream->GetPtr()); */
    }

    VOID Trans1(cudah::cuda_buffer buffer, cudah::cuda_buffer staticNormals, cudah::cuda_buffer staticPositions, cudah::cuda_buffer iu, UINT elements, UINT blockSize, ULONG time, cudah::cuda_stream stream)
    {
        /*
        UINT gws = cudah::cudah::GetThreadCount(elements, blockSize);
        
        sinCosRiddle2((VertexData*)buffer->ptr, (float3*)staticNormals->ptr, (float3*)staticPositions->ptr, gws, blockSize, time * 1e-3f, elements, stream->GetPtr());
        
        blockSize = 64;

        UINT gridSize = blockSize;
        gws = cudah::cudah::GetThreadCount(elements, blockSize);
        
        UINT gridWidth = 8;
        
        UINT vertexStride = 256+1;

        //cudaStreamSynchronize(stream->GetPtr());
        INT blocksPerRow = (int)ceil(vertexStride / (float)gridWidth);

        INT rest = gridWidth - gridWidth * blocksPerRow % vertexStride;

        INT sharedMem = 0;//3 * gridSize * sizeof(float3);

        comupteNormals((VertexData*)buffer->ptr, NULL, gws / gridSize, blockSize, gridWidth, gridSize, vertexStride, blocksPerRow, rest, elements, sharedMem, stream->GetPtr()); */
    }

#define CONTROL_PNTS_WIDTH 32
#define THIRDS (CONTROL_PNTS_WIDTH-4)
#define VERTEX_STRIDE THIRDS * 12
#define SPLINE_SCALE 8

    VOID BSplineTrans(cudah::cuda_buffer buffer, cudah::cuda_buffer staticNormals, cudah::cuda_buffer staticPositions, cudah::cuda_buffer iu, UINT elements, UINT blockSize, ULONG time, cudah::cuda_stream stream)
    {
        /*UINT gws = cudah::cudah::GetThreadCount(CONTROL_PNTS_WIDTH * CONTROL_PNTS_WIDTH, blockSize);
        animateBSline(gws, blockSize, 
            (float3*)staticPositions->ptr, 
            app::g_pApp->GetUpdateTimer()->GetTime() * 1e-3f, CONTROL_PNTS_WIDTH * CONTROL_PNTS_WIDTH, CONTROL_PNTS_WIDTH, CONTROL_PNTS_WIDTH, stream->GetPtr());
        
        gws = cudah::cudah::GetThreadCount(elements, blockSize);
        comupteNormalsBSpline((VertexData*)buffer->ptr, gws, blockSize, VERTEX_STRIDE, elements, stream->GetPtr()); */
    }

    chimera::Geometry* CreateSphere(VOID)
    {
        return GeometryFactory::CreateSphere(255, 127, FALSE);
    }

    chimera::Geometry* CreateGrid(VOID)
    {
        return GeometryFactory::CreateNormedGrid(127, 127, 2, FALSE);
    }

    chimera::Geometry* CreateGridBSpline(VOID)
    {
        return GeometryFactory::CreateNormedGrid(VERTEX_STRIDE - 1, VERTEX_STRIDE - 1, SPLINE_SCALE, FALSE);
    }

    TransformShowRoom::TransformShowRoom(CONST std::string& file, chimera::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    BOOL TransformShowRoom::VLoad(BOOL block)
    {
        CreateStaticPlane(this);

        ActorDescription desc = m_pActorFactory->CreateActorDescription();
        chimera::TransformComponent* tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        chimera::RenderComponent* renderCmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
        VAddActor(desc);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(0, 2, 15);
        comp->GetTransformation()->SetRotateX(-XM_PIDIV2*0.25f);

        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        chimera::CudaTransformationNode* node = new chimera::CudaTransformationNode(&Trans1, &CreateGrid);
        renderComp->m_sceneNode = std::shared_ptr<chimera::CudaTransformationNode>(node);
        node->SetNormaleTexture("normal/leatherN.jpg");
        node->SetTexture("leather.jpg");
        node->GetMaterial().m_reflectance = 0.55f;
        node->GetMaterial().m_texScale = 8;

        VAddActor(desc);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(-5, 3, 5);
        comp->GetTransformation()->SetScale(1);

        renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        node = new chimera::CudaTransformationNode(&Trans0, &CreateSphere);
        renderComp->m_sceneNode = std::shared_ptr<chimera::CudaTransformationNode>(node);
        node->SetNormaleTexture("normal/fleshN.jpg");
        node->SetTexture("flesh.jpg");
        node->GetMaterial().m_reflectance = 0.75f;
        node->GetMaterial().m_texScale = 16;

        VAddActor(desc);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        node = new chimera::UniformBSplineNode(&CreateGridBSpline, &BSplineTrans, VERTEX_STRIDE, CONTROL_PNTS_WIDTH);

        util::UniformBSpline& spline = ((UniformBSplineNode*)node)->GetSpline();
        INT size = CONTROL_PNTS_WIDTH;

        for(INT y = 0; y < size; ++y)
        {
            for(INT x = 0; x < size; ++x)
            {
                util::Vec3 point;
                point.x = (-1.0f + 2.0f * x / (FLOAT)(size-1)) * SPLINE_SCALE; 
                point.y = 0;
                point.z = (-1.0f + 2.0f * y / (FLOAT)(size-1)) * SPLINE_SCALE;
                spline.AddPoint(point);
            }
        }

        renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_sceneNode = std::shared_ptr<chimera::CudaTransformationNode>(node);
        node->SetNormaleTexture("normal/tilesN.png");
        node->SetTexture("7992-D.jpg");
        node->GetMaterial().m_reflectance = 0.75f;
        node->GetMaterial().m_texScale = 16;

        comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(20, 1, 15);
        comp->GetTransformation()->SetScale(1);

        VAddActor(desc);

        //lights
        srand(0);

        chimera::ActorDescription ldesc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::LightComponent* lightComponent = ldesc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<chimera::TransformComponent>("TransformComponent");
        util::Vec3 point;
        point.x = 15; 
        point.y = 5;
        point.z = 15;
        comp->GetTransformation()->SetTranslate(point.x, point.y, point.z);
        comp->GetTransformation()->SetScale(10);

        ldesc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        VAddActor(ldesc);

        ldesc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = ldesc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(-10, 5, 5);
        comp->GetTransformation()->SetScale(10);
        
        ldesc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);
        
        VAddActor(ldesc);

        ldesc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = ldesc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(3, 5, 5);
        comp->GetTransformation()->SetScale(10);

        ldesc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        VAddActor(ldesc);

        ldesc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = ldesc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(0, 5, 15);
        comp->GetTransformation()->SetScale(10);

        ldesc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        VAddActor(ldesc);

        //camera tracking

        std::shared_ptr<chimera::Actor> trackingShotActor = chimera::g_pApp->GetLogic()->VCreateActor("freelookCamera.xml");
        trackingShotActor->SetName("track0");
        std::shared_ptr<chimera::TrackingShot> ts = std::shared_ptr<chimera::TrackingShot>(new chimera::TrackingShot(trackingShotActor, TRUE));
        
        util::Vec3 focus(20, 1, 15);
        FLOAT h = 6;
        FLOAT w = 12;
        util::Vec3 p0(focus.x - w, h, focus.z - w);
        util::Vec3 p1(focus.x - w, h, focus.z + w);
        util::Vec3 p2(focus.x + w, h, focus.z + w);
        util::Vec3 p3(focus.x + w, h, focus.z - w);

        ts->AddBasePoint(p0, focus);
        ts->AddBasePoint(p1, focus);
        ts->AddBasePoint(p2, focus);
        ts->AddBasePoint(p3, focus);
        ts->SetDivisions(256);
        
        chimera::g_pApp->GetLogic()->GetProcessManager()->Attach(ts);

        trackingShotActor = chimera::g_pApp->GetLogic()->VCreateActor("freelookCamera.xml");
        trackingShotActor->SetName("track1");
        ts = std::shared_ptr<chimera::TrackingShot>(new chimera::TrackingShot(trackingShotActor, TRUE));

        focus.Set(-5, 3, 5);
        h = 6;
        w = 5;
        p0.Set(focus.x - w, h, focus.z - w);
        p1.Set(focus.x - w, h, focus.z + w);
        p2.Set(focus.x + w, h, focus.z + w);
        p3.Set(focus.x + w, h, focus.z - w);

        ts->AddBasePoint(p0, focus);
        ts->AddBasePoint(p1, focus);
        ts->AddBasePoint(p2, focus);
        ts->AddBasePoint(p3, focus);
        ts->SetDivisions(256);

        chimera::g_pApp->GetLogic()->GetProcessManager()->Attach(ts);
       
        return TRUE;
    }

    BOOL TransformShowRoom::VSave(LPCSTR file)
    {
        return TRUE;
    }

#define CONTROL_PNTS_WIDTH0 8
#define THIRDS0 (CONTROL_PNTS_WIDTH0-3)
#define VERTEX_STRIDE0 THIRDS0 * 12
#define SPLINE_SCALE0 10

    BSplinePatchLevel::BSplinePatchLevel(CONST std::string& file, chimera::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    VOID BSplineTrans0(cudah::cuda_buffer buffer, cudah::cuda_buffer staticNormals, cudah::cuda_buffer staticPositions, cudah::cuda_buffer iu, UINT elements, UINT blockSize, ULONG time, cudah::cuda_stream stream)
    {
        UINT gws = cudahu::GetThreadCount(elements, blockSize);
        //comupteNormalsBSpline((VertexData*)buffer->ptr, gws, blockSize, VERTEX_STRIDE0, elements, stream->GetPtr());
    }

    chimera::Geometry* CreateGridBSpline0(VOID)
    {
        return GeometryFactory::CreateNormedGrid(VERTEX_STRIDE0 - 1, VERTEX_STRIDE0 - 1, SPLINE_SCALE0, FALSE);
    }

    BOOL BSplinePatchLevel::VLoad(BOOL block)
    {
        CreateStaticPlane(this);

        chimera::ActorDescription desc = m_pActorFactory->CreateActorDescription();
        chimera::TransformComponent* tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        chimera::RenderComponent* renderCmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
        VAddActor(desc);

        ADD_EVENT_LISTENER(this, &BSplinePatchLevel::OnControlPointMove, chimera::ActorMovedEvent::TYPE);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        m_node = new chimera::UniformBSplineNode(&CreateGridBSpline0, &BSplineTrans0, VERTEX_STRIDE0, CONTROL_PNTS_WIDTH0, TRUE);

        util::UniformBSpline& spline = ((UniformBSplineNode*)m_node)->GetSpline();
        INT size = CONTROL_PNTS_WIDTH0;

        m_controlPoints = new std::shared_ptr<chimera::Actor>[size * size];

        srand(0);

        //GetSerpent(size-1, size-1, 100);
        INT index = 0;
        for(INT y = 0; y < size; ++y)
        {
            for(INT x = 0; x < size; ++x)
            {
                util::Vec3 point;
                point.x = (-1.0f + 2.0f * x / (FLOAT)(size-1)) * SPLINE_SCALE0; 
                point.y = 10 * rand() / (FLOAT)RAND_MAX;
                point.z = (-1.0f + 2.0f * y / (FLOAT)(size-1)) * SPLINE_SCALE0;

                chimera::ActorDescription sdesc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

                chimera::TransformComponent* comp = sdesc->AddComponent<chimera::TransformComponent>("TransformComponent");
                comp->GetTransformation()->SetTranslate(point.x, point.y, point.z);
                comp->GetTransformation()->SetScale(0.25f);

                chimera::RenderComponent* renderComp = sdesc->AddComponent<chimera::RenderComponent>("RenderComponent");
                renderComp->m_resource = "debug_sphere.obj";

                sdesc->AddComponent<chimera::PickableComponent>("PickableComponent");

                m_controlPoints[index++] = VAddActor(sdesc);

                spline.AddPoint(point);
            }
        }

        renderCmp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderCmp->m_sceneNode = std::shared_ptr<chimera::CudaTransformationNode>(m_node);
        m_node->SetNormaleTexture("normal/tilesN.png");
        m_node->SetTexture("7992-D.jpg");
        m_node->GetMaterial().m_reflectance = 0.75f;
        m_node->GetMaterial().m_texScale = 16;

        tcomp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        tcomp->GetTransformation()->SetTranslate(+0, 0, 0);
        VAddActor(desc);

        TBD_FOR_INT(3)
        {
            chimera::ActorDescription ldesc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
            
            ldesc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

            chimera::LightComponent* lightComponent = ldesc->AddComponent<chimera::LightComponent>("LightComponent");
            lightComponent->m_type = "point";
            lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
            lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
            lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
            lightComponent->m_color.w = 1;

            chimera::TransformComponent* comp = ldesc->AddComponent<chimera::TransformComponent>("TransformComponent");
            util::Vec3 point;
            point.x = (-1.0f + 2.0f * (rand() / (FLOAT)RAND_MAX)) * (SPLINE_SCALE0/2); 
            point.y = 10;
            point.z = (-1.0f + 2.0f * (rand() / (FLOAT)RAND_MAX)) * (SPLINE_SCALE0/2);
            comp->GetTransformation()->SetTranslate(point.x, point.y, point.z);
            comp->GetTransformation()->SetScale(10);

            VAddActor(ldesc);
        }

        return TRUE;
    }

    VOID BSplinePatchLevel::OnControlPointMove(chimera::IEventPtr data)
    {
        std::shared_ptr<chimera::ActorMovedEvent> movedEvent = std::static_pointer_cast<chimera::ActorMovedEvent>(data);

        TBD_FOR_INT(CONTROL_PNTS_WIDTH0 * CONTROL_PNTS_WIDTH0)
        {
            if(m_controlPoints[i]->GetId() == movedEvent->m_actor->GetId())
            {
                std::string name("controlPoints");
                cudah::cuda_buffer buffer = m_node->GetCudaBuffer(name);
                cudah::cudah* cuda = m_node->GetCuda();
                chimera::TransformComponent* tc = m_controlPoints[i]->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock().get();
                float3 data;
                data.x = tc->GetTransformation()->GetTranslation().x;
                data.y = tc->GetTransformation()->GetTranslation().y;
                data.z = tc->GetTransformation()->GetTranslation().z;
                FLOAT* ptr = (FLOAT*)buffer->ptr;
                LOG_CRITICAL_ERROR_A("doesnt work!");
                //cuMemcpy(ptr + 3 * i, &data, 12, cudaMemcpyHostToDevice);
            }
        }
    }

    BOOL BSplinePatchLevel::VSave(LPCSTR file)
    {
        return TRUE;
    }

    BSplinePatchLevel::~BSplinePatchLevel(VOID)
    {
        REMOVE_EVENT_LISTENER(this, &BSplinePatchLevel::OnControlPointMove, chimera::ActorMovedEvent::TYPE);
        SAFE_ARRAY_DELETE(m_controlPoints);
    }

    GroupedObjLevel::GroupedObjLevel(LPCSTR file, chimera::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    BOOL GroupedObjLevel::VSave(LPCSTR file /* = NULL */)
    {
        SaveXMLLevel save;
        return save.VSaveLevel(this, file);
    }

    BOOL GroupedObjLevel::VLoad(BOOL block)
    {
        CreateStaticPlane(this);

        ActorDescription desc = m_pActorFactory->CreateActorDescription();
        chimera::TransformComponent* tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        chimera::RenderComponent* renderCmp = desc->AddComponent<chimera::RenderComponent>(chimera::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
        VAddActor(desc);

        std::string dir = "../Assets/" + chimera::g_pApp->GetConfig()->GetString("sMeshPath") + VGetFile();
        std::string objList = dir + "/filelist.txt";
        std::ifstream stream(objList.c_str());

        std::string meshFile;
        while(stream.good())
        {
            std::getline(stream, meshFile); 
            
            if(meshFile.size() == 0) continue;

            desc = m_pActorFactory->CreateActorDescription();
            chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
            chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
            renderComp->m_resource = VGetFile() + "/" + meshFile;

            chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
            physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
            physicComponent->m_material = "static";
            physicComponent->m_shapeStyle = "mesh";
            physicComponent->m_shapeType = "concave";
            physicComponent->m_radius = 1; 
            physicComponent->m_meshFile = VGetFile() + "/" + meshFile;

            VAddActor(desc);
        }
        return TRUE;
    }
}
