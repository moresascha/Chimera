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

namespace tbd
{

    VOID CreateStaticPlane(BaseLevel* level)
    {
        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_meshFile = "plane.obj";

        tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "static";
        physicComponent->m_shapeType = "plane";

        level->VAddActor(desc);
    }

    VOID CreateCube(CONST util::Vec3& pos, BaseLevel* level)
    {
        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_meshFile = "box.obj";

        tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "static";
        physicComponent->m_shapeType = "box";
        physicComponent->m_radius = 1;

        level->VAddActor(desc);
    }

    std::shared_ptr<tbd::Actor> CreateSphere(CONST util::Vec3& pos, BaseLevel* level)
    {
        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_meshFile = "sphere.obj";

        tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "kinematic";
        physicComponent->m_shapeType = "sphere";
        physicComponent->m_radius = 1;

        return level->VAddActor(desc);
    }

    BaseLevel::BaseLevel(CONST std::string& file, tbd::ActorFactory* factory) : m_file(file), m_name("unnamed"), m_pActorFactory(factory)
    {
        event::EventListener listener = fastdelegate::MakeDelegate(this, &XMLLevel::ActorCreatedDelegate);
        event::IEventManager::Get()->VAddEventListener(listener, event::ActorCreatedEvent::TYPE);
    }

    std::shared_ptr<tbd::Actor> BaseLevel::VFindActor(ActorId id)
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

    VOID BaseLevel::ActorCreatedDelegate(event::IEventPtr eventData)
    {
        std::shared_ptr<event::ActorCreatedEvent> data = std::static_pointer_cast<event::ActorCreatedEvent>(eventData);
        auto it = std::find(m_idsToLoad.begin(), m_idsToLoad.end(), data->m_id);
        if(it != m_idsToLoad.end())
        {
            m_idsToLoad.erase(it);
            if(m_idsToLoad.empty())
            {
                event::IEventPtr levelLoadedEvent(new event::LevelLoadedEvent(std::string(GetName())));
                event::IEventManager::Get()->VQueueEventThreadSave(levelLoadedEvent);
            }
        }
    }

    std::shared_ptr<tbd::Actor> BaseLevel::VAddActor(ActorDescription& desc)
    {
        std::shared_ptr<tbd::Actor> actor = m_pActorFactory->CreateActor(desc);
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
            event::IEventPtr deletActorEvent(new event::DeleteActorEvent(it->first));
            event::IEventManager::Get()->VQueueEvent(deletActorEvent);
        }
    }

    BaseLevel::~BaseLevel(VOID)
    {
        VUnload();
        event::EventListener listener = fastdelegate::MakeDelegate(this, &XMLLevel::ActorCreatedDelegate);
        event::IEventManager::Get()->VRemoveEventListener(listener, event::ActorCreatedEvent::TYPE);
    }

    XMLLevel::XMLLevel(CONST std::string& file, tbd::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    std::shared_ptr<tbd::Actor> XMLLevel::VAddActor(tinyxml2::XMLElement* pNode)
    {
        if(!pNode) 
        {
            LOG_CRITICAL_ERROR("TiXmlElement cant be NULL");
        }
        std::shared_ptr<tbd::Actor> actor = m_pActorFactory->CreateActor(pNode);
        if(actor)
        {
            this->m_actors[actor->GetId()] = actor;
            this->m_idsToLoad.push_back(actor->GetId());
            return actor;
        }
        LOG_CRITICAL_ERROR("Failed to create actor");
        return std::shared_ptr<tbd::Actor>();
    }

    BOOL XMLLevel::VLoad(BOOL block)
    {
        tinyxml2::XMLDocument doc;

        tbd::Resource r(app::g_pApp->GetConfig()->GetString("sLevelPath") + GetFile().c_str());

        std::shared_ptr<tbd::ResHandle> handle = app::g_pApp->GetCache()->GetHandle(r);

        doc.Parse((CHAR*)handle->Buffer());

        tinyxml2::XMLElement* root = doc.RootElement();
        RETURN_IF_FAILED(root);

        for(tinyxml2::XMLElement* pNode = root->FirstChildElement(); pNode; pNode = pNode->NextSiblingElement())
        {
            CONST CHAR* file = pNode->Attribute("file");
            if(file)
            {
                LOG_CRITICAL_ERROR("not supported yet!");
                /*std::shared_ptr<tbd::Actor> actor = VAddActor(file);
                m_idsToLoad.push_back(actor->GetId());
                for(TiXmlElement* comps = pNode->FirstChildElement(); comps; comps = comps->NextSiblingElement())
                {
                    m_actorFactory.ReplaceComponent(actor, comps);
                } */
            }
            else
            {
                VAddActor(pNode);
            }
        }

        doc.Clear();
        return TRUE;
    }

    BOOL XMLLevel::VSave(LPCSTR file)
    {
        
        tinyxml2::XMLDocument doc;
        /*
        TiXmlDeclaration* dec = new TiXmlDeclaration("1.0", "", "");
        TiXmlElement* root = new TiXmlElement("Level");
        doc.LinkEndChild(dec);
        for(auto it = m_actors.begin(); it != m_actors.end(); ++it)
        {
            tbd::Actor* actor = it->second.get();

            TiXmlElement* xmlactor = new TiXmlElement("Actor");
            root->LinkEndChild(xmlactor);

            for(auto it2 = actor->GetComponents()->begin(); it2 != actor->GetComponents()->end(); ++it2)
            {
                it2->second->VSave(xmlactor);
            }
        }
        doc.LinkEndChild(root);
        std::string s;
        s += app::g_pApp->GetCache()->GetFile().VGetName();
        s += "/";
        s += app::g_pApp->GetConfig()->GetString("sLevelPath");
        s += GetFile().c_str(); */

        return TRUE;//doc.SaveFile(file != NULL ? file : s.c_str());
    }

    RandomLevel::RandomLevel(CONST std::string& file, tbd::ActorFactory* factory) : BaseLevel(file, factory)
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

    VOID TriggerDelegate(event::IEventPtr data)
    {
        std::shared_ptr<event::TriggerEvent> te = std::static_pointer_cast<event::TriggerEvent>(data);
        ActorId trigger = te->m_triggerActor;
        QUEUE_EVENT(new event::DeleteActorEvent(trigger));
        DEBUG_OUT("On Trigger\n");
    }

    BOOL RandomLevel::VLoad(BOOL block)
    {
        CreateStaticPlane(this);
        
        for(CHAR i = 0; i < 1; ++i)
        {
            for(CHAR j = 0; j < 1; ++j)
            {
                tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
                desc->AddComponent<tbd::ParticleComponent>("ParticleComponent");
                tbd::TransformComponent* t = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
                t->GetTransformation()->SetTranslate(0,5,0);//3 * i - 3,0.1f, 3 * j - j);
                tbd::PickableComponent* pick = desc->AddComponent<tbd::PickableComponent>("PickableComponent");
                std::shared_ptr<Actor> actor = VAddActor(desc);
            }
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
        tbd::TransformComponent* tcomp = desc->AddComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        tbd::RenderComponent* renderCmp = desc->AddComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
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
        return TRUE;
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

    d3d::Geometry* CreateSphere(VOID)
    {
        return GeometryFactory::CreateSphere(255, 127, FALSE);
    }

    d3d::Geometry* CreateGrid(VOID)
    {
        return GeometryFactory::CreateNormedGrid(127, 127, 2, FALSE);
    }

    d3d::Geometry* CreateGridBSpline(VOID)
    {
        return GeometryFactory::CreateNormedGrid(VERTEX_STRIDE - 1, VERTEX_STRIDE - 1, SPLINE_SCALE, FALSE);
    }

    TransformShowRoom::TransformShowRoom(CONST std::string& file, tbd::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    BOOL TransformShowRoom::VLoad(BOOL block)
    {
        CreateStaticPlane(this);

        ActorDescription desc = m_pActorFactory->CreateActorDescription();
        tbd::TransformComponent* tcomp = desc->AddComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        tbd::RenderComponent* renderCmp = desc->AddComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
        VAddActor(desc);

        desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(0, 2, 15);
        comp->GetTransformation()->SetRotateX(-XM_PIDIV2*0.25);

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        tbd::CudaTransformationNode* node = new tbd::CudaTransformationNode(&Trans1, &CreateGrid);
        renderComp->m_sceneNode = std::shared_ptr<tbd::CudaTransformationNode>(node);
        node->SetNormaleTexture("normal/leatherN.jpg");
        node->SetTexture("leather.jpg");
        node->GetMaterial().m_reflectance = 0.55f;
        node->GetMaterial().m_texScale = 8;

        VAddActor(desc);

        desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(-5, 3, 5);
        comp->GetTransformation()->SetScale(1);

        renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        node = new tbd::CudaTransformationNode(&Trans0, &CreateSphere);
        renderComp->m_sceneNode = std::shared_ptr<tbd::CudaTransformationNode>(node);
        node->SetNormaleTexture("normal/fleshN.jpg");
        node->SetTexture("flesh.jpg");
        node->GetMaterial().m_reflectance = 0.75f;
        node->GetMaterial().m_texScale = 16;

        VAddActor(desc);

        desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        node = new tbd::UniformBSplineNode(&CreateGridBSpline, &BSplineTrans, VERTEX_STRIDE, CONTROL_PNTS_WIDTH);

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

        renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_sceneNode = std::shared_ptr<tbd::CudaTransformationNode>(node);
        node->SetNormaleTexture("normal/tilesN.png");
        node->SetTexture("7992-D.jpg");
        node->GetMaterial().m_reflectance = 0.75f;
        node->GetMaterial().m_texScale = 16;

        comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(20, 1, 15);
        comp->GetTransformation()->SetScale(1);

        VAddActor(desc);

        //lights
        srand(0);

        tbd::ActorDescription ldesc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::LightComponent* lightComponent = ldesc->AddComponent<tbd::LightComponent>("LightComponent");
        lightComponent->m_type = "Point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<tbd::TransformComponent>("TransformComponent");
        util::Vec3 point;
        point.x = 15; 
        point.y = 5;
        point.z = 15;
        comp->GetTransformation()->SetTranslate(point.x, point.y, point.z);
        comp->GetTransformation()->SetScale(10);

        ldesc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        VAddActor(ldesc);

        ldesc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = ldesc->AddComponent<tbd::LightComponent>("LightComponent");
        lightComponent->m_type = "Point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(-10, 5, 5);
        comp->GetTransformation()->SetScale(10);
        
        ldesc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);
        
        VAddActor(ldesc);

        ldesc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = ldesc->AddComponent<tbd::LightComponent>("LightComponent");
        lightComponent->m_type = "Point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(3, 5, 5);
        comp->GetTransformation()->SetScale(10);

        ldesc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        VAddActor(ldesc);

        ldesc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = ldesc->AddComponent<tbd::LightComponent>("LightComponent");
        lightComponent->m_type = "Point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;

        comp = ldesc->AddComponent<tbd::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(0, 5, 15);
        comp->GetTransformation()->SetScale(10);

        ldesc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        VAddActor(ldesc);

        //camera tracking

        std::shared_ptr<tbd::Actor> trackingShotActor = app::g_pApp->GetLogic()->VCreateActor("freelookCamera.xml");
        trackingShotActor->SetName("track0");
        std::shared_ptr<proc::TrackingShot> ts = std::shared_ptr<proc::TrackingShot>(new proc::TrackingShot(trackingShotActor, TRUE));
        
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
        
        app::g_pApp->GetLogic()->GetProcessManager()->Attach(ts);

        trackingShotActor = app::g_pApp->GetLogic()->VCreateActor("freelookCamera.xml");
        trackingShotActor->SetName("track1");
        ts = std::shared_ptr<proc::TrackingShot>(new proc::TrackingShot(trackingShotActor, TRUE));

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

        app::g_pApp->GetLogic()->GetProcessManager()->Attach(ts);
       
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

    BSplinePatchLevel::BSplinePatchLevel(CONST std::string& file, tbd::ActorFactory* factory) : BaseLevel(file, factory)
    {

    }

    VOID BSplineTrans0(cudah::cuda_buffer buffer, cudah::cuda_buffer staticNormals, cudah::cuda_buffer staticPositions, cudah::cuda_buffer iu, UINT elements, UINT blockSize, ULONG time, cudah::cuda_stream stream)
    {
        UINT gws = cudah::cudah::GetThreadCount(CONTROL_PNTS_WIDTH0 * CONTROL_PNTS_WIDTH0, blockSize);
        gws = cudah::cudah::GetThreadCount(elements, blockSize);
        //comupteNormalsBSpline((VertexData*)buffer->ptr, gws, blockSize, VERTEX_STRIDE0, elements, stream->GetPtr());
    }

    d3d::Geometry* CreateGridBSpline0(VOID)
    {
        return GeometryFactory::CreateNormedGrid(VERTEX_STRIDE0 - 1, VERTEX_STRIDE0 - 1, SPLINE_SCALE0, FALSE);
    }

    BOOL BSplinePatchLevel::VLoad(BOOL block)
    {
        CreateStaticPlane(this);

        tbd::ActorDescription desc = m_pActorFactory->CreateActorDescription();
        tbd::TransformComponent* tcomp = desc->AddComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetTranslate(0,0,0);

        tbd::RenderComponent* renderCmp = desc->AddComponent<tbd::RenderComponent>(tbd::RenderComponent::COMPONENT_ID);
        renderCmp->m_type = "skydome";
        renderCmp->m_info = "skydome3.jpg";
        VAddActor(desc);

        ADD_EVENT_LISTENER(this, &BSplinePatchLevel::OnControlPointMove, event::ActorMovedEvent::TYPE);

        desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        m_node = new tbd::UniformBSplineNode(&CreateGridBSpline0, &BSplineTrans0, VERTEX_STRIDE0, CONTROL_PNTS_WIDTH0, TRUE);

        util::UniformBSpline& spline = ((UniformBSplineNode*)m_node)->GetSpline();
        INT size = CONTROL_PNTS_WIDTH0;

        m_controlPoints = new std::shared_ptr<tbd::Actor>[size * size];

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

                tbd::ActorDescription sdesc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

                tbd::TransformComponent* comp = sdesc->AddComponent<tbd::TransformComponent>("TransformComponent");
                comp->GetTransformation()->SetTranslate(point.x, point.y, point.z);
                comp->GetTransformation()->SetScale(0.25f);

                tbd::RenderComponent* renderComp = sdesc->AddComponent<tbd::RenderComponent>("RenderComponent");
                renderComp->m_meshFile = "debug_sphere.obj";

                sdesc->AddComponent<tbd::PickableComponent>("PickableComponent");

                m_controlPoints[index++] = VAddActor(sdesc);

                spline.AddPoint(point);
            }
        }

        renderCmp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderCmp->m_sceneNode = std::shared_ptr<tbd::CudaTransformationNode>(m_node);
        m_node->SetNormaleTexture("normal/tilesN.png");
        m_node->SetTexture("7992-D.jpg");
        m_node->GetMaterial().m_reflectance = 0.75f;
        m_node->GetMaterial().m_texScale = 16;

        tcomp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        tcomp->GetTransformation()->SetTranslate(+0, 0, 0);
        VAddActor(desc);

        TBD_FOR_INT(3)
        {
            tbd::ActorDescription ldesc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();
            
            ldesc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

            tbd::LightComponent* lightComponent = ldesc->AddComponent<tbd::LightComponent>("LightComponent");
            lightComponent->m_type = "Point";
            lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
            lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
            lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
            lightComponent->m_color.w = 1;

            tbd::TransformComponent* comp = ldesc->AddComponent<tbd::TransformComponent>("TransformComponent");
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

    VOID BSplinePatchLevel::OnControlPointMove(event::IEventPtr data)
    {
        std::shared_ptr<event::ActorMovedEvent> movedEvent = std::static_pointer_cast<event::ActorMovedEvent>(data);

        TBD_FOR_INT(CONTROL_PNTS_WIDTH0 * CONTROL_PNTS_WIDTH0)
        {
            if(m_controlPoints[i]->GetId() == movedEvent->m_actor->GetId())
            {
                std::string name("controlPoints");
                cudah::cuda_buffer buffer = m_node->GetCudaBuffer(name);
                cudah::cudah* cuda = m_node->GetCuda();
                tbd::TransformComponent* tc = m_controlPoints[i]->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock().get();
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
        REMOVE_EVENT_LISTENER(this, &BSplinePatchLevel::OnControlPointMove, event::ActorMovedEvent::TYPE);
        SAFE_ARRAY_DELETE(m_controlPoints);
    }
}
