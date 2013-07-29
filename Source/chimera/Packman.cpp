#include "Packman.h"
#include "PackmanLogic.h"
#include "Camera.h"
#include "Input.h"
#include "GameView.h"
#include "GuiComponent.h"
#include "Maze.h"
#include "Script.h"
#include "luaplus/LuaPlus.h"
#include "LuaHelper.h"
#include "Process.h"
namespace packman
{
    CONST ComponentId AIComponent::COMPONENT_ID = 0x9fe588e8;
    class AIComponent;
    tbd::ActorComponent* CreateAIComponent(VOID) 
    {
        return new AIComponent;
    }

    INT IsWallFromScript(LuaPlus::LuaObject posVec3)
    {
        util::Vec3 pos;
        tbd::script::ConvertAndCheckTableToVec3(pos, posVec3);
        packman::Maze* m = (packman::Maze*)app::g_pApp->GetLogic()->Getlevel();
        BOOL isWall = m->IsWall(pos.x, pos.z);
        return isWall;
    }

    class WatchLuaFile : public proc::WatchFileModificationProcess
    {
    public:
        WatchLuaFile(LPCTSTR file, LPCTSTR dir) : WatchFileModificationProcess(file, dir)
        {

        }
        VOID VOnFileModification(VOID)
        {
            app::g_pApp->GetScript()->VRunFile("files/scripts/packmanai.lua");
        }
    };

    VOID Packman::VCreateLogicAndView(VOID)
    {
        m_pLogic = new packman::PackmanLogic();

        if(!m_pLogic->VInit())
        {
            LOG_CRITICAL_ERROR("Gamelogic init fail");
        }

        m_pHumanView = new tbd::HumanGameView();
        m_pHumanView->SetName("HGameView");
        std::shared_ptr<tbd::Actor> camera = m_pLogic->VCreateActor("camera.xml");
        
        camera->SetName("player");

        std::shared_ptr<tbd::CameraComponent> cameraComp = camera->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock();
        
        std::shared_ptr<tbd::PhysicComponent> phxComp = camera->GetComponent<tbd::PhysicComponent>(tbd::PhysicComponent::COMPONENT_ID).lock();
        if(phxComp)
        {
            cameraComp->GetCamera()->VSetYOffset(1.6f);
        }

        std::shared_ptr<tbd::Actor> freeLook = m_pLogic->VCreateActor("freeLookCamera.xml");
        freeLook->SetName("free");

        std::shared_ptr<tbd::ActorController> controller;
        
        controller = std::shared_ptr<tbd::ActorController>(new tbd::CharacterController());
        controller->SetName("GameController");
        controller->SetMaxSpeed(30);
        controller->SetMinSpeed(10);

        m_pInput->PushMouseListener(controller.get());
        m_pInput->PushKeyListener(controller.get());

        m_pLogic->AttachGameView(std::shared_ptr<tbd::HumanGameView>(m_pHumanView), camera);
        m_pLogic->AttachGameView(controller, camera);

        app::PostInitMessage("Loading GUI ...");
        tbd::gui::D3D_GUI* c = m_pHumanView->GetGUI();

        tbd::Dimension dim;
        tbd::gui::GuiSpriteComponent* crossHair = new tbd::gui::GuiSpriteComponent(0, 0, 8, 8);
        dim.x = d3d::g_width / 2 - 4;
        dim.y = d3d::g_height / 2 - 4;
        dim.h = 8;
        dim.w = 8;
        crossHair->VSetBackgroundColor(1,1,1);
        crossHair->VSetDimension(dim);
        crossHair->SetTexture("crosshair_dot.png");
        c->AddComponent("CH", crossHair);

        m_pLogic->GetActorFactory()->AddComponentCreator(CreateAIComponent, "AIComponent", AIComponent::COMPONENT_ID);

        packman::Maze* level = new packman::Maze(50, 1, m_pLogic->GetActorFactory());

        m_pLogic->VLoadLevel(level);
        m_pScript->ResgisterFunction("IsWall", &packman::IsWallFromScript);

        m_pLogic->AttachProcess(std::shared_ptr<WatchLuaFile>(new WatchLuaFile(L"packmanai.lua", L"files/scripts/")));

        app::g_pApp->GetScript()->VRunFile("files/scripts/packmanai.lua");

        app::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand("target free");
    }
}