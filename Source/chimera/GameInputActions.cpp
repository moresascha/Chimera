#include "GameInputActions.h"
#include "GameApp.h"
#include "ProcessManager.h"
#include "GameLogic.h"
#include "GameView.h"
#include "EventManager.h"
#include "Picker.h"
#include "Components.h"
#include "math.h"
#include "Camera.h"
#include "D3DRenderer.h"
#include "CudaTransformationNode.h"
#include "SceneGraph.h"
#include "SceneNode.h"

namespace gameinput
{
    BOOL m_editMode = TRUE;
    BOOL m_bMovePicked = FALSE;
    FLOAT m_actorPlaceScale = 4;
    ActorId m_toModify = CM_INVALID_ACTOR_ID;
    BOOL m_kinematicPhysical = TRUE;
    INT g_rotationDir = 1;

    VOID PlayTestSound();

    BOOL ToggleFlashlight(chimera::Command& cmd)
    {
        std::shared_ptr<chimera::Actor> a = chimera::g_pApp->GetHumanView()->GetTarget();
        std::shared_ptr<chimera::ISceneNode> node = chimera::g_pApp->GetHumanView()->GetSceneGraph()->FindActorNode(a->GetId());
        if(node && node->GetChilds().size() > 0)
        {
            std::shared_ptr<chimera::ISceneNode>& child = node->GetChilds()[0];
            std::shared_ptr<chimera::Actor> light = chimera::g_pApp->GetLogic()->VFindActor(child->VGetActorId());
            if(light->HasComponent<chimera::LightComponent>(chimera::LightComponent::COMPONENT_ID))
            {
                std::shared_ptr<chimera::LightComponent> flashLight = light->GetComponent<chimera::LightComponent>(chimera::LightComponent::COMPONENT_ID).lock();
                flashLight->m_activated = !flashLight->m_activated;
            }
        }
        return TRUE;
    }

    BOOL CreateSpotlightThing(chimera::Command& cmd)
    {
        std::shared_ptr<chimera::CameraComponent> camera = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();
        CONST util::Vec3& dir = camera->GetCamera()->GetViewDir();
        util::Vec3 pos = camera->GetCamera()->GetEyePos();

        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->VCreateActorDescription();

        desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

        pos = pos + dir * 3;

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        comp->GetTransformation()->SetTranslate(pos);

        chimera::RenderComponent * renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = "spottwotest.obj";

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_material = "dynamic";
        physicComponent->m_shapeStyle = "sphere";
        physicComponent->m_radius = 1;

        std::shared_ptr<chimera::Actor> a = chimera::g_pApp->GetLogic()->VCreateActor(desc, TRUE);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::LightComponent* lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "spot";
        lightComponent->m_color.x = 1;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;
        lightComponent->m_angle = 55;
        lightComponent->m_intensity = 24;

        comp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        comp->GetTransformation()->SetScale(50);
        comp->GetTransformation()->RotateX(-XM_PIDIV2);
        comp->GetTransformation()->Translate(0, 0, 0);

        //desc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        chimera::ParentComponent* pc = desc->AddComponent<chimera::ParentComponent>("ParentComponent");
        pc->m_parentId = a->GetId();

        chimera::g_pApp->GetLogic()->VCreateActor(desc, TRUE);

        desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "spot";
        lightComponent->m_color.x = 1;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0;//0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;
        lightComponent->m_angle = 55;
        lightComponent->m_intensity = 24;

        comp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        comp->GetTransformation()->SetScale(50);
        comp->GetTransformation()->RotateX(XM_PIDIV2);
        comp->GetTransformation()->Translate(0, 0, 0);

        //desc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

        pc = desc->AddComponent<chimera::ParentComponent>("ParentComponent");
        pc->m_parentId = a->GetId();

        chimera::g_pApp->GetLogic()->VCreateActor(desc, TRUE);

        return TRUE;
    }

    BOOL SetRenderMode(chimera::Command& cmd)
    {
        std::string mode = cmd.GetNextCharStr();
        if(cmd.IsError())
        {
            return FALSE;
        }

        if(mode == "editor")
        {
            m_editMode = TRUE;
        }
        else
        {
            m_editMode = TRUE;
        }
        chimera::g_pApp->GetHumanView()->ActivateScene(mode.c_str());
        return TRUE;
    }

    BOOL SetDefaultPlayer(chimera::Command& cmd)
    {
        //set default camera
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("player");
        chimera::g_pApp->GetHumanView()->VSetTarget(player);
        chimera::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    //CSM debugging
    BOOL SetCascadeViewCamera(chimera::Command& cmd)
    {
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("cascadeViewCamera");
        chimera::g_pApp->GetHumanView()->VSetTarget(player);
        chimera::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeLightCamera(chimera::Command& cmd)
    {
        //set light camera
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("cascadeLightCamera");
        chimera::g_pApp->GetHumanView()->VSetTarget(player);
        chimera::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeCam0(chimera::Command& cmd)
    {
        //set cascade0 camera
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("cascadeCam0");
        chimera::g_pApp->GetHumanView()->VSetTarget(player);
        chimera::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeCam1(chimera::Command& cmd)
    {
        //set cascade0 camera
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("cascadeCam1");
        chimera::g_pApp->GetHumanView()->VSetTarget(player);
        chimera::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeCam2(chimera::Command& cmd)
    {
        //set cascade0 camera
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("cascadeCam2");
        chimera::g_pApp->GetHumanView()->VSetTarget(player);
        chimera::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL PickActor(chimera::Command& cmd)
    {
        chimera::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(chimera::g_pRasterizerStateFrontFaceSolid);
        chimera::g_pApp->GetHumanView()->GetPicker()->VRender();
        chimera::g_pApp->GetHumanView()->GetPicker()->VPostRender();
        chimera::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();

        if(chimera::g_pApp->GetHumanView()->GetPicker()->VHasPicked() && chimera::g_pApp->GetHumanView()->GetPicker()->VPick() != m_toModify)
        {
            m_toModify = chimera::g_pApp->GetHumanView()->GetPicker()->VPick();
            m_bMovePicked = TRUE;
        }
        else
        {
            m_toModify = CM_INVALID_ACTOR_ID;
            m_bMovePicked = FALSE;
        }
        return TRUE;
    }

    BOOL ApplyPlayerForce(chimera::Command& cmd)
    {
        std::shared_ptr<chimera::CameraComponent> camera = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();
        CONST util::Vec3& dir = camera->GetCamera()->GetViewDir();

        util::Vec3 tdir = dir;
        tdir.Scale(0.25f);
        std::stringstream ss;
        ss << "force ";
        ss << tdir.x << " ";
        ss << tdir.y << " ";
        ss << tdir.z;
        chimera::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(ss.str().c_str());
        
        ss.str("");
        ss << "torque ";
        ss << dir.x << " ";
        ss << dir.y << " ";
        ss << dir.z;
        chimera::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(ss.str().c_str());

        return TRUE;
    }

    BOOL ApplyForce(chimera::Command& cmd)
    {
        if(m_toModify == CM_INVALID_ACTOR_ID)
        {
            return TRUE;
        }

        FLOAT x = cmd.GetNextFloat();
        FLOAT y = cmd.GetNextFloat();
        FLOAT z = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);

        FLOAT n = cmd.GetNextFloat();

        n = n == 0 ? 1 : n;

        util::Vec3 dir(x, y, z);

        chimera::ApplyForceEvent* ev = new chimera::ApplyForceEvent();
        ev->m_actor = chimera::g_pApp->GetLogic()->VFindActor(m_toModify);
        ev->m_dir = dir;
        ev->m_newtons = 100000;

        QUEUE_EVENT(ev);

        //m_toModify = INVALID_ACTOR_ID;

        m_bMovePicked = FALSE;

        return TRUE;
    }

    BOOL ApplyTorque(chimera::Command& cmd)
    {
        if(m_toModify == CM_INVALID_ACTOR_ID)
        {
            return TRUE;
        }

        FLOAT x = cmd.GetNextFloat();
        FLOAT y = cmd.GetNextFloat();
        FLOAT z = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);

        FLOAT n = cmd.GetNextFloat();

        n = n == 0 ? 1 : n;

        util::Vec3 dir(x, y, z);

        chimera::ApplyTorqueEvent* ev = new chimera::ApplyTorqueEvent();
        ev->m_actor = chimera::g_pApp->GetLogic()->VFindActor(m_toModify);
        ev->m_torque = dir;
        ev->m_newtons = 100000;

        QUEUE_EVENT(ev);

        m_toModify = CM_INVALID_ACTOR_ID;

        m_bMovePicked = FALSE;

        return TRUE;
    }

    BOOL SpawnBasicMeshActor(chimera::Command& cmd)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        std::string meshFile = cmd.GetNextCharStr();
        std::string shapeType = "convex";
        std::string shapeStyle = "mesh";

        CHECK_COMMAND(cmd);

        chimera::CMResource r(meshFile);
        if(!chimera::g_pApp->GetCache()->HasResource(r))
        {
            return FALSE;
        }

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        std::shared_ptr<chimera::CameraComponent> cameraComp = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();

        util::Vec3 pos;
        util::Vec3 dir = cameraComp->GetCamera()->GetViewDir();
        dir.Scale(6.0f);
        pos = cameraComp->GetCamera()->GetEyePos() + dir;
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
        renderComp->m_resource = meshFile;

        chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = "dynamic";//m_kinematicPhysical ? "kinematic" : "dynamic";
        physicComponent->m_shapeStyle = shapeStyle;
        physicComponent->m_shapeType = shapeType;
        physicComponent->m_radius = 1; 
        physicComponent->m_meshFile = meshFile;

        desc->AddComponent<chimera::PickableComponent>("PickableComponent");

        chimera::IEventPtr createActorEvent(new chimera::CreateActorEvent(desc));
        chimera::IEventManager::Get()->VQueueEvent(createActorEvent);

        //PlayTestSound();
        return TRUE;
    }

    BOOL SpawnSpotLight(chimera::Command& cmd)
    {
        chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
        std::shared_ptr<chimera::CameraComponent> cameraComp = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();

        util::Vec3 pos;
        
        util::Vec3 dir = cameraComp->GetCamera()->GetViewDir();
        dir.Scale(6.0f);
        pos = cameraComp->GetCamera()->GetEyePos() + dir;
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);    
        comp->GetTransformation()->SetScale(10.0f);

        if(!m_kinematicPhysical)
        {
            chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
            physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
            physicComponent->m_material = "bouncy";
            physicComponent->m_shapeStyle = "sphere";
            physicComponent->m_radius = 1;
        }

        chimera::LightComponent* lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;
        //lightComponent->m_radius = 10.0f;// * rand() / (FLOAT)RAND_MAX;

        desc->AddComponent<chimera::PickableComponent>("PickableComponent");

        chimera::IEventPtr createActorEvent(new chimera::CreateActorEvent(desc));
        chimera::IEventManager::Get()->VQueueEvent(createActorEvent);

        //std::shared_ptr<proc::StrobeLightProcess> lightProc = std::shared_ptr<proc::StrobeLightProcess>(new proc::StrobeLightProcess(lightComponent, 0.5f, 100));
        //app::g_pApp->GetLogic()->AttachProcess(lightProc); 
        return TRUE;
    }

    BOOL ToggleActorPropPhysical(chimera::Command& cmd)
    {
        m_kinematicPhysical = !m_kinematicPhysical;
        return TRUE;
    }

    BOOL DeletePickedActor(chimera::Command& cmd)
    {
        if(m_toModify != CM_INVALID_ACTOR_ID)
        {
            chimera::IEventPtr deletActorEvent(new chimera::DeleteActorEvent(m_toModify));
            chimera::IEventManager::Get()->VQueueEvent(deletActorEvent);
            m_toModify = CM_INVALID_ACTOR_ID;
        }
        return TRUE;
    }

    VOID MouseWheelActorPositionModify(INT x, INT y, INT delta)
    {
        if(m_editMode && m_bMovePicked)
        {
            m_actorPlaceScale += delta / (6.0f * abs(delta));
            m_actorPlaceScale = CLAMP(m_actorPlaceScale, 4, 50);

            std::shared_ptr<chimera::CameraComponent> camera = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();
            util::Vec3 trans;
            util::Vec3 transDir = camera->GetCamera()->GetViewDir();
            transDir.Scale(m_actorPlaceScale);
            trans = camera->GetCamera()->GetEyePos() + transDir;// - targetTrans;

            std::shared_ptr<chimera::Actor> target = chimera::g_pApp->GetLogic()->VFindActor(m_toModify);
            if(target)
            {
                std::shared_ptr<chimera::TransformComponent> tc = target->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
                event::IEventPtr event(new event::MoveActorEvent(m_toModify, trans, FALSE));
                event::IEventManager::Get()->VQueueEvent(event);
            }
        }
    }

    VOID RotateActor(util::Vec3& axis, FLOAT angle)
    {
        if(m_toModify != CM_INVALID_ACTOR_ID)
        {
            event::IEventPtr event(new event::MoveActorEvent(m_toModify, axis, g_rotationDir * angle));
            event::IEventManager::Get()->VQueueEvent(event);
        }
    }

    BOOL ToggleRotationDir(chimera::Command& cmd)
    {
        g_rotationDir *= -1;
        return TRUE;
    }

    BOOL RotatXPickedActor(chimera::Command& cmd)
    {
        FLOAT rx = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);
        util::Vec3 r(1, 0, 0);
        RotateActor(r, rx);
        return TRUE;
    }

    BOOL RotatYPickedActor(chimera::Command& cmd)
    {
        FLOAT ry = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);
        util::Vec3 r(0, 1, 0);
        RotateActor(r, ry);
        return TRUE;
    }

    BOOL RotatZPickedActor(chimera::Command& cmd)
    {
        FLOAT rz = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);
        util::Vec3 r(0, 0, 1);
        RotateActor(r, rz);
        return TRUE;
    }

    BOOL FlushVRam(chimera::Command& cmd)
    {
        chimera::g_pApp->GetHumanView()->GetVRamManager()->Flush();
        return TRUE;
    }

    VOID MovePicked(chimera::IEventPtr data)
    {
        std::shared_ptr<chimera::ActorMovedEvent> moved = std::static_pointer_cast<chimera::ActorMovedEvent>(data);
        std::shared_ptr<chimera::Actor> player = chimera::g_pApp->GetLogic()->VFindActor("player");
        //DEBUG_OUT_A("%d, %d, %d, %d\n", moved->m_actor->GetId(), player->GetId(), m_bMovePicked, m_editMode);
        //TODO, CameraComponent
        if(m_editMode && m_bMovePicked && (moved->m_actor->GetId() == player->GetId() || moved->m_actor->GetId() == chimera::g_pApp->GetLogic()->VFindActor("free")->GetId()))
        {
            std::shared_ptr<chimera::CameraComponent> camera = chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();
            util::Vec3 trans;
            util::Vec3 transDir = camera->GetCamera()->GetViewDir();
            transDir.Scale(m_actorPlaceScale);
            trans = camera->GetCamera()->GetEyePos() + transDir;// - targetTrans;

            std::shared_ptr<chimera::Actor> target = chimera::g_pApp->GetLogic()->VFindActor(m_toModify);
            if(target)
            {
                std::shared_ptr<chimera::TransformComponent> tc = target->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
                event::IEventPtr event(new event::MoveActorEvent(m_toModify, trans, FALSE));
                event::IEventManager::Get()->VQueueEvent(event);
            }
        }
    }

    VOID ScaleActorAction(FLOAT factor)
    {
        if(m_toModify != CM_INVALID_ACTOR_ID)
        {
            std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VFindActor(m_toModify);
            std::shared_ptr<chimera::TransformComponent> cmp = actor->GetComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID).lock();
            CONST util::Vec3& scale = cmp->GetTransformation()->GetScale();
            FLOAT s = scale.x + factor;
            cmp->GetTransformation()->SetScale(s < 0.05f ? 0.05f : s);
            QUEUE_EVENT(new chimera::ActorMovedEvent(actor));
        }
    }

    BOOL ScaleActorBigger(chimera::Command& cmd)
    {
        ScaleActorAction(0.05f);
        return TRUE;
    }

    BOOL ScaleActorSmaller(chimera::Command& cmd)
    {
        ScaleActorAction(-0.05f);
        return TRUE;
    }

    BOOL Jump(chimera::Command& cmd)
    {
        FLOAT height = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);

        ActorId id = chimera::g_pApp->GetHumanView()->GetTarget()->GetId();
        chimera::MoveActorEvent* me = new chimera::MoveActorEvent(id, util::Vec3(0, height, 0));
        me->m_isJump = TRUE;
        QUEUE_EVENT(me);
        return TRUE;
    }

    BOOL ptToggle = FALSE;
    ActorId lastTorus = CM_INVALID_ACTOR_ID;

    BOOL ToogleCamera(chimera::Command& cmd)
    {
        ptToggle = !ptToggle;

        //we spawn a rotating torus to indicate the players position
        if(ptToggle)
        {
            chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

            chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
            util::Vec3 pos = 
                chimera::g_pApp->GetHumanView()->GetTarget()->GetComponent<chimera::TransformComponent>(
                chimera::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetTranslation();

            comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

            chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
            renderComp->m_resource = "torus.obj";

            desc->AddComponent<chimera::PickableComponent>(chimera::PickableComponent::COMPONENT_ID);

            /*std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VCreateActor(desc);
            lastTorus = actor->GetId();
            std::shared_ptr<proc::RotationProcess> proc = std::shared_ptr<proc::RotationProcess>(new proc::RotationProcess(actor, util::Vec3(0.1f, 0.1f, 0)));
            app::g_pApp->GetLogic()->AttachProcess(proc); */
        }
        else
        {
            std::shared_ptr<chimera::DeleteActorEvent> deleteEvent = std::shared_ptr<chimera::DeleteActorEvent>(new chimera::DeleteActorEvent(lastTorus));
            //event::EventManager::Get()->VQueueEvent(deleteEvent);
        }


        return chimera::commands::SetTarget(ptToggle ? "free" : "player");
    }

    BOOL SpawnSpheres(chimera::Command& cmd)
    {
        INT count = 0;
        while(count++ < 80)
        {
            chimera::ActorDescription desc = chimera::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

            FLOAT scale = 4;
            FLOAT x = 2 * (rand() / (FLOAT)RAND_MAX - 0.5f);
            FLOAT z = 2 * (rand() / (FLOAT)RAND_MAX - 0.5f);
            FLOAT dy = rand() / (FLOAT)RAND_MAX;

            chimera::TransformComponent* comp = desc->AddComponent<chimera::TransformComponent>("TransformComponent");
            comp->GetTransformation()->SetTranslate(scale * x, 40 + 10 * dy, 20 + scale * z);

            FLOAT disDas = rand() / (FLOAT)RAND_MAX;

            chimera::RenderComponent* renderComp = desc->AddComponent<chimera::RenderComponent>("RenderComponent");
            renderComp->m_resource = disDas < 0.5 ? "box.obj" : "sphere.obj";

            chimera::PhysicComponent* physicComponent = desc->AddComponent<chimera::PhysicComponent>("PhysicComponent");
            physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
            physicComponent->m_material = disDas < 0.5 ? "dynamic" : "bouncy";
            physicComponent->m_shapeStyle = disDas < 0.5 ? "box" : "sphere";
            physicComponent->m_radius = 1;

            desc->AddComponent<chimera::PickableComponent>("PickableComponent");

            chimera::IEventPtr createActorEvent(new chimera::CreateActorEvent(desc));
            chimera::IEventManager::Get()->VQueueEvent(createActorEvent);

        }
        return TRUE;
    }

    BOOL SetRasterState(chimera::Command& cmd)
    {
        std::string state = cmd.GetNextCharStr();
        CHECK_COMMAND(cmd);
        ID3D11RasterizerState* rs;

        if(state == "wire")
        {
            rs = chimera::g_pRasterizerStateWrireframe;
        }
        else if(state == "nocull")
        {
            rs = chimera::g_pRasterizerStateNoCullingSolid;
        }
        else if(state == "frontfaces")
        {
            rs = chimera::g_pRasterizerStateFrontFaceSolid;
        }
        else if(state == "backfaces")
        {
            rs = chimera::g_pRasterizerStateBackFaceSolid;
        }
        else
        {
            return FALSE;
        }

        chimera::g_pApp->GetRenderer()->SetDefaultRasterizerState(rs);

        return TRUE;
    }

    BOOL DrawCP(chimera::Command& cmd)
    {
        BOOL draw = cmd.GetNextBool();
        CHECK_COMMAND(cmd);
        chimera::UniformBSplineNode::drawCP_CP = draw;
        return TRUE;
    }


    BOOL SetFullscreen(chimera::Command& cmd)
    {
        BOOL fs = cmd.GetNextBool();
        CHECK_COMMAND(cmd);
        UINT w = 0;
        UINT h = 0;
        if(!fs)
        {
            w = chimera::g_pApp->GetConfig()->GetInteger("iWidth");
            h = chimera::g_pApp->GetConfig()->GetInteger("iHeight");
        }

        chimera::SetFullscreenState(fs, w, h);

        return TRUE;
    }

    //set actions here
    VOID RegisterCommands(chimera::ActorController& controller, chimera::CommandInterpreter& interpreter)
    {

        interpreter.RegisterCommand("sc0", SetCascadeCam0);
        interpreter.RegisterCommand("sc1", SetCascadeCam1);
        interpreter.RegisterCommand("sc2", SetCascadeCam2);

        interpreter.RegisterCommand("pick", PickActor);
        interpreter.RegisterCommand("spotlight", SpawnSpotLight);
        interpreter.RegisterCommand("physical", ToggleActorPropPhysical);
        interpreter.RegisterCommand("removepicked", DeletePickedActor);
        interpreter.RegisterCommand("scalesmall", ScaleActorSmaller);
        interpreter.RegisterCommand("scalebig", ScaleActorBigger);
        interpreter.RegisterCommand("rendermode", SetRenderMode, "rendermode [default,debug,editor,wirefilled]");
        interpreter.RegisterCommand("spawn", SpawnBasicMeshActor);
        interpreter.RegisterCommand("rasterstate", SetRasterState, "rasterstate [wire, nocull, fronfaces, backfaces]");
        interpreter.RegisterCommand("drawCP", DrawCP);
        interpreter.RegisterCommand("fullscreen", SetFullscreen);
        interpreter.RegisterCommand("toggleFL", ToggleFlashlight);

        interpreter.RegisterCommand("rotateX", RotatXPickedActor);
        interpreter.RegisterCommand("rotateY", RotatYPickedActor);
        interpreter.RegisterCommand("rotateZ", RotatZPickedActor);
        interpreter.RegisterCommand("toggleRotationDir", ToggleRotationDir);
        interpreter.RegisterCommand("sl", CreateSpotlightThing);
        
        interpreter.RegisterCommand("toggleCamera", ToogleCamera);
        interpreter.RegisterCommand("spawnz", SpawnSpheres);
        interpreter.RegisterCommand("jump", Jump);
        interpreter.RegisterCommand("force", ApplyForce, "force x y z [n]");
        interpreter.RegisterCommand("torque", ApplyTorque, "torque x y z [n]");
        interpreter.RegisterCommand("pforce", ApplyPlayerForce, "pforce n");
        interpreter.RegisterCommand("flushVR", FlushVRam, "");

        controller.SetMouseScrollAction(MouseWheelActorPositionModify);

        ADD_EVENT_LISTENER_STATIC(&MovePicked, chimera::ActorMovedEvent::TYPE);
    }
}