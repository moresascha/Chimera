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

namespace gameinput
{
    BOOL m_editMode = FALSE;
    BOOL m_bMovePicked = FALSE;
    FLOAT m_actorPlaceScale = 4;
    ActorId m_toModify = INVALID_ACTOR_ID;
    BOOL m_kinematicPhysical = TRUE;

    VOID PlayTestSound();

    BOOL SetRenderMode(tbd::Command& cmd)
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
            m_editMode = FALSE;
        }
        app::g_pApp->GetHumanView()->ActivateScene(mode.c_str());
        return TRUE;
    }

    BOOL SetDefaultPlayer(tbd::Command& cmd)
    {
        //set default camera
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("player");
        app::g_pApp->GetHumanView()->VSetTarget(player);
        app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    //CSM debugging
    BOOL SetCascadeViewCamera(tbd::Command& cmd)
    {
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("cascadeViewCamera");
        app::g_pApp->GetHumanView()->VSetTarget(player);
        app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeLightCamera(tbd::Command& cmd)
    {
        //set light camera
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("cascadeLightCamera");
        app::g_pApp->GetHumanView()->VSetTarget(player);
        app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeCam0(tbd::Command& cmd)
    {
        //set cascade0 camera
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("cascadeCam0");
        app::g_pApp->GetHumanView()->VSetTarget(player);
        app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeCam1(tbd::Command& cmd)
    {
        //set cascade0 camera
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("cascadeCam1");
        app::g_pApp->GetHumanView()->VSetTarget(player);
        app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL SetCascadeCam2(tbd::Command& cmd)
    {
        //set cascade0 camera
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("cascadeCam2");
        app::g_pApp->GetHumanView()->VSetTarget(player);
        app::g_pApp->GetLogic()->VFindGameView("GameController")->VSetTarget(player);
        return TRUE;
    }

    BOOL PickActor(tbd::Command& cmd)
    {
        if(app::g_pApp->GetHumanView()->GetPicker()->VHasPicked() && app::g_pApp->GetHumanView()->GetPicker()->VPick() != m_toModify)
        {
            m_toModify = app::g_pApp->GetHumanView()->GetPicker()->VPick();
            m_bMovePicked = TRUE;
        }
        else
        {
            m_toModify = INVALID_ACTOR_ID;
            m_bMovePicked = FALSE;
        }
        return TRUE;
    }

    BOOL ApplyPlayerForce(tbd::Command& cmd)
    {
        std::shared_ptr<tbd::CameraComponent> camera = app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
        CONST util::Vec3& dir = camera->GetCamera()->GetViewDir();
        std::stringstream ss;
        ss << "force ";
        ss << dir.x << " ";
        ss << dir.y << " ";
        ss << dir.z;
        app::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(ss.str().c_str());
        
        ss.str("");
        ss << "torque ";
        ss << dir.x << " ";
        ss << dir.y << " ";
        ss << dir.z;
        app::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand(ss.str().c_str());

        return TRUE;
    }

    BOOL ApplyForce(tbd::Command& cmd)
    {
        if(m_toModify == INVALID_ACTOR_ID)
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

        event::ApplyForceEvent* ev = new event::ApplyForceEvent();
        ev->m_actor = app::g_pApp->GetLogic()->VFindActor(m_toModify);
        ev->m_dir = dir;
        ev->m_newtons = 100000;

        QUEUE_EVENT(ev);

        //m_toModify = INVALID_ACTOR_ID;

        m_bMovePicked = FALSE;

        return TRUE;
    }

    BOOL ApplyTorque(tbd::Command& cmd)
    {
        if(m_toModify == INVALID_ACTOR_ID)
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

        event::ApplyTorqueEvent* ev = new event::ApplyTorqueEvent();
        ev->m_actor = app::g_pApp->GetLogic()->VFindActor(m_toModify);
        ev->m_torque = dir;
        ev->m_newtons = 100000;

        QUEUE_EVENT(ev);

        m_toModify = INVALID_ACTOR_ID;

        m_bMovePicked = FALSE;

        return TRUE;
    }

    BOOL SpawnBasicMeshActor(tbd::Command& cmd)
    {
        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        std::string meshFile = cmd.GetNextCharStr();
        std::string shapetype = cmd.GetNextCharStr();

        tbd::Resource r(meshFile);
        if(!app::g_pApp->GetCache()->HasResource(r))
        {
            return FALSE;
        }

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        std::shared_ptr<tbd::CameraComponent> cameraComp = app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();

        util::Vec3 pos;
        util::Vec3 dir = cameraComp->GetCamera()->GetViewDir();
        dir.Scale(6.0f);
        pos = cameraComp->GetCamera()->GetEyePos() + dir;
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

        tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
        renderComp->m_meshFile = meshFile;

        tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
        physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
        physicComponent->m_material = m_kinematicPhysical ? "kinematic" : "dynamic";
        physicComponent->m_shapeType = shapetype;
        physicComponent->m_radius = 1;

        desc->AddComponent<tbd::PickableComponent>("PickableComponent");

        event::IEventPtr createActorEvent(new event::CreateActorEvent(desc));
        event::IEventManager::Get()->VQueueEvent(createActorEvent);

        //PlayTestSound();
        return TRUE;
    }

    BOOL SpawnSpotLight(tbd::Command& cmd)
    {
        tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

        tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
        std::shared_ptr<tbd::CameraComponent> cameraComp = app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();

        util::Vec3 pos;
        
        util::Vec3 dir = cameraComp->GetCamera()->GetViewDir();
        dir.Scale(6.0f);
        pos = cameraComp->GetCamera()->GetEyePos() + dir;
        comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);    
        comp->GetTransformation()->SetScale(10.0f);

        if(!m_kinematicPhysical)
        {
            tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
            physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
            physicComponent->m_material = "bouncy";
            physicComponent->m_shapeType = "sphere";
            physicComponent->m_radius = 1;
        }

        tbd::LightComponent* lightComponent = desc->AddComponent<tbd::LightComponent>("LightComponent");
        lightComponent->m_type = "Point";
        lightComponent->m_color.x = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.y = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.z = 0.5f + 2 * rand() / (FLOAT)RAND_MAX;
        lightComponent->m_color.w = 1;
        //lightComponent->m_radius = 10.0f;// * rand() / (FLOAT)RAND_MAX;

        desc->AddComponent<tbd::PickableComponent>("PickableComponent");

        event::IEventPtr createActorEvent(new event::CreateActorEvent(desc));
        event::IEventManager::Get()->VQueueEvent(createActorEvent);

        //std::shared_ptr<proc::StrobeLightProcess> lightProc = std::shared_ptr<proc::StrobeLightProcess>(new proc::StrobeLightProcess(lightComponent, 0.5f, 100));
        //app::g_pApp->GetLogic()->AttachProcess(lightProc); 
        return TRUE;
    }

    BOOL ToggleActorPropPhysical(tbd::Command& cmd)
    {
        m_kinematicPhysical = !m_kinematicPhysical;
        return TRUE;
    }

    BOOL DeletePickedActor(tbd::Command& cmd)
    {
        if(m_toModify != INVALID_ACTOR_ID)
        {
            event::IEventPtr deletActorEvent(new event::DeleteActorEvent(m_toModify));
            event::IEventManager::Get()->VQueueEvent(deletActorEvent);
            m_toModify = INVALID_ACTOR_ID;
        }
        return TRUE;
    }

    VOID MouseWheelActorPositionModify(INT x, INT y, INT delta)
    {
        if(m_editMode && m_bMovePicked)
        {
            m_actorPlaceScale += delta / (6.0f * abs(delta));
            m_actorPlaceScale = CLAMP(m_actorPlaceScale, 4, 50);

            std::shared_ptr<tbd::CameraComponent> camera = app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
            util::Vec3 trans;
            util::Vec3 transDir = camera->GetCamera()->GetViewDir();
            transDir.Scale(m_actorPlaceScale);
            trans = camera->GetCamera()->GetEyePos() + transDir;// - targetTrans;

            event::IEventPtr event(new event::MoveActorEvent(m_toModify, trans, util::Vec3(), FALSE));
            event::IEventManager::Get()->VQueueEvent(event);
        }
    }

    BOOL FlushVRam(tbd::Command& cmd)
    {
        app::g_pApp->GetHumanView()->GetVRamManager()->Flush();
        return TRUE;
    }

    VOID MovePicked(event::IEventPtr data)
    {
        std::shared_ptr<event::ActorMovedEvent> moved = std::static_pointer_cast<event::ActorMovedEvent>(data);
        std::shared_ptr<tbd::Actor> player = app::g_pApp->GetLogic()->VFindActor("player");
        //DEBUG_OUT_A("%d, %d, %d, %d\n", moved->m_actor->GetId(), player->GetId(), m_bMovePicked, m_editMode);
        //TODO, CameraComponent
        if(m_editMode && m_bMovePicked && (moved->m_actor->GetId() == player->GetId() || moved->m_actor->GetId() == app::g_pApp->GetLogic()->VFindActor("free")->GetId()))
        {
            std::shared_ptr<tbd::CameraComponent> camera = app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
            util::Vec3 trans;
            util::Vec3 transDir = camera->GetCamera()->GetViewDir();
            transDir.Scale(m_actorPlaceScale);
            trans = camera->GetCamera()->GetEyePos() + transDir;// - targetTrans;

            event::IEventPtr event(new event::MoveActorEvent(m_toModify, trans, util::Vec3(), FALSE));
            event::IEventManager::Get()->VQueueEvent(event);
        }
    }

    VOID ScaleActorAction(FLOAT factor)
    {
        if(m_toModify != INVALID_ACTOR_ID)
        {
            std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(m_toModify);
            std::shared_ptr<tbd::TransformComponent> cmp = actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock();
            CONST util::Vec3& scale = cmp->GetTransformation()->GetScale();
            FLOAT s = scale.x + factor;
            cmp->GetTransformation()->SetScale(s < 0.05f ? 0.05f : s);
        }
    }

    BOOL ScaleActorBigger(tbd::Command& cmd)
    {
        ScaleActorAction(0.05f);
        return TRUE;
    }

    BOOL ScaleActorSmaller(tbd::Command& cmd)
    {
        ScaleActorAction(-0.05f);
        return TRUE;
    }

    BOOL Jump(tbd::Command& cmd)
    {
        FLOAT height = cmd.GetNextFloat();
        CHECK_COMMAND(cmd);

        ActorId id = app::g_pApp->GetHumanView()->GetTarget()->GetId();
        QUEUE_EVENT(new event::MoveActorEvent(id, util::Vec3(0, height, 0), util::Vec3()));
        return TRUE;
    }

    BOOL ptToggle = FALSE;
    ActorId lastTorus = INVALID_ACTOR_ID;

    BOOL ToogleCamera(tbd::Command& cmd)
    {
        ptToggle = !ptToggle;

        //we spawn a rotating torus to indicate the players position
        if(ptToggle)
        {
            tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

            tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
            util::Vec3 pos = 
                app::g_pApp->GetHumanView()->GetTarget()->GetComponent<tbd::TransformComponent>(
                tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->GetTranslation();

            comp->GetTransformation()->SetTranslate(pos.x, pos.y, pos.z);

            tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
            renderComp->m_meshFile = "torus.obj";

            desc->AddComponent<tbd::PickableComponent>(tbd::PickableComponent::COMPONENT_ID);

            /*std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VCreateActor(desc);
            lastTorus = actor->GetId();
            std::shared_ptr<proc::RotationProcess> proc = std::shared_ptr<proc::RotationProcess>(new proc::RotationProcess(actor, util::Vec3(0.1f, 0.1f, 0)));
            app::g_pApp->GetLogic()->AttachProcess(proc); */
        }
        else
        {
            std::shared_ptr<event::DeleteActorEvent> deleteEvent = std::shared_ptr<event::DeleteActorEvent>(new event::DeleteActorEvent(lastTorus));
            //event::EventManager::Get()->VQueueEvent(deleteEvent);
        }


        return tbd::commands::SetTarget(ptToggle ? "free" : "player");
    }

    BOOL SpawnSpheres(tbd::Command& cmd)
    {
        INT count = 0;
        while(count++ < 80)
        {
            tbd::ActorDescription desc = app::g_pApp->GetLogic()->GetActorFactory()->CreateActorDescription();

            FLOAT scale = 4;
            FLOAT x = 2 * (rand() / (FLOAT)RAND_MAX - 0.5f);
            FLOAT z = 2 * (rand() / (FLOAT)RAND_MAX - 0.5f);
            FLOAT dy = rand() / (FLOAT)RAND_MAX;

            tbd::TransformComponent* comp = desc->AddComponent<tbd::TransformComponent>("TransformComponent");
            comp->GetTransformation()->SetTranslate(scale * x, 40 + 10 * dy, 20 + scale * z);

            FLOAT disDas = rand() / (FLOAT)RAND_MAX;

            tbd::RenderComponent* renderComp = desc->AddComponent<tbd::RenderComponent>("RenderComponent");
            renderComp->m_meshFile = disDas < 0.5 ? "box.obj" : "sphere.obj";

            tbd::PhysicComponent* physicComponent = desc->AddComponent<tbd::PhysicComponent>("PhysicComponent");
            physicComponent->m_dim.x = 2; physicComponent->m_dim.z = 2; physicComponent->m_dim.y = 2;
            physicComponent->m_material = disDas < 0.5 ? "dynamic" : "bouncy";
            physicComponent->m_shapeType = disDas < 0.5 ? "box" : "sphere";
            physicComponent->m_radius = 1;

            desc->AddComponent<tbd::PickableComponent>("PickableComponent");

            event::IEventPtr createActorEvent(new event::CreateActorEvent(desc));
            event::IEventManager::Get()->VQueueEvent(createActorEvent);

        }
        return TRUE;
    }

    BOOL SetRasterState(tbd::Command& cmd)
    {
        std::string state = cmd.GetNextCharStr();
        CHECK_COMMAND(cmd);
        ID3D11RasterizerState* rs;

        if(state == "wire")
        {
            rs = d3d::g_pRasterizerStateWrireframe;
        }
        else if(state == "nocull")
        {
            rs = d3d::g_pRasterizerStateNoCullingSolid;
        }
        else if(state == "frontfaces")
        {
            rs = d3d::g_pRasterizerStateFrontFaceSolid;
        }
        else if(state == "backfaces")
        {
            rs = d3d::g_pRasterizerStateBackFaceSolid;
        }
        else
        {
            return FALSE;
        }

        app::g_pApp->GetRenderer()->SetDefaultRasterizerState(rs);

        return TRUE;
    }

    BOOL DrawCP(tbd::Command& cmd)
    {
        BOOL draw = cmd.GetNextBool();
        CHECK_COMMAND(cmd);
        tbd::UniformBSplineNode::drawCP_CP = draw;
        return TRUE;
    }

    BOOL LoadLevel(tbd::Command& cmd)
    {
        std::string name = cmd.GetNextCharStr();
        CHECK_COMMAND(cmd);
        
        if(name == "l0")
        {
            tbd::BaseLevel* level = new tbd::TransformShowRoom("patch0", app::g_pApp->GetLogic()->GetActorFactory());
            app::g_pApp->GetLogic()->VLoadLevel(level);
        } 
        else if(name == "l1")
        {
            tbd::BaseLevel* level = new tbd::BSplinePatchLevel("patch0", app::g_pApp->GetLogic()->GetActorFactory());
            app::g_pApp->GetLogic()->VLoadLevel(level);
        }

        return TRUE;
    }

    BOOL RunProc(tbd::Command& cmd)
    {
        std::string cmdStr = cmd.GetRemainingString();
        
        CHECK_COMMAND(cmd);

        STARTUPINFO si;
        ZeroMemory(&si, sizeof(si));
        AllocConsole();

        si.cb = sizeof(si);

        PROCESS_INFORMATION pi;
        ZeroMemory(&pi, sizeof(pi));

        SECURITY_ATTRIBUTES sa;
        ZeroMemory(&sa, sizeof(sa));
        sa.nLength = sizeof(sa);
        sa.lpSecurityDescriptor = NULL;
        sa.bInheritHandle = TRUE;

        std::wstring ws(cmdStr.begin(), cmdStr.end());
        LPTSTR szCmdline = _tcsdup(ws.c_str());

        if(!CreateProcess(NULL, szCmdline, &sa, NULL, FALSE, 0, NULL, NULL, &si, &pi))
        {
            return FALSE;
        }

        WaitForSingleObject(pi.hProcess, INFINITE);

        DWORD exitCode;
        GetExitCodeProcess(pi.hProcess, &exitCode);

        FreeConsole();

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        free(szCmdline);

        return exitCode;
    }

    //set actions here
    VOID RegisterCommands(tbd::ActorController& controller, tbd::CommandInterpreter& interpreter)
    {
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
        interpreter.RegisterCommand("loadlevel", LoadLevel);
        
        interpreter.RegisterCommand("toggleCamera", ToogleCamera);
        interpreter.RegisterCommand("spawnz", SpawnSpheres);
        interpreter.RegisterCommand("jump", Jump);
        interpreter.RegisterCommand("force", ApplyForce, "force x y z [n]");
        interpreter.RegisterCommand("torque", ApplyTorque, "torque x y z [n]");
        interpreter.RegisterCommand("pforce", ApplyPlayerForce, "pforce n");
        interpreter.RegisterCommand("runproc", RunProc);
        controller.SetMouseScrollAction(MouseWheelActorPositionModify);

        ADD_EVENT_LISTENER_STATIC(&MovePicked, event::ActorMovedEvent::TYPE);
    }
}