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
    chimera::ActorComponent* CreateAIComponent(VOID) 
    {
        return new AIComponent;
    }

    INT IsWallFromScript(LuaPlus::LuaObject posVec3)
    {
        util::Vec3 pos;
        chimera::script::ConvertAndCheckTableToVec3(pos, posVec3);
        packman::Maze* m = (packman::Maze*)chimera::g_pApp->GetLogic()->Getlevel();
        BOOL isWall = m->IsWall(pos.x, pos.z);
        return isWall;
    }

    class WatchLuaFile : public chimera::WatchFileModificationProcess
    {
    public:
        WatchLuaFile(LPCTSTR file, LPCTSTR dir) : WatchFileModificationProcess(file, dir)
        {

        }
        VOID VOnFileModification(VOID)
        {
            chimera::g_pApp->GetScript()->VRunFile("files/scripts/packmanai.lua");
        }
    };

    VOID Packman::VCreateLogicAndView(VOID)
    {
        m_pLogic = new packman::PackmanLogic();

        if(!m_pLogic->VInit())
        {
            LOG_CRITICAL_ERROR("Gamelogic init fail");
        }

        m_pHumanView = new chimera::HumanGameView();
        m_pHumanView->SetName("HGameView");
        std::shared_ptr<chimera::Actor> camera = m_pLogic->VCreateActor("camera.xml");
        
        camera->SetName("player");

        std::shared_ptr<chimera::CameraComponent> cameraComp = camera->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock();
        
        std::shared_ptr<chimera::PhysicComponent> phxComp = camera->GetComponent<chimera::PhysicComponent>(chimera::PhysicComponent::COMPONENT_ID).lock();
        if(phxComp)
        {
            cameraComp->GetCamera()->VSetYOffset(1.6f);
        }

        std::shared_ptr<chimera::Actor> freeLook = m_pLogic->VCreateActor("freeLookCamera.xml");
        freeLook->SetName("free");

        std::shared_ptr<chimera::ActorController> controller;
        
        controller = std::shared_ptr<chimera::ActorController>(new chimera::CharacterController());
        controller->SetName("GameController");
        controller->SetMaxSpeed(30);
        controller->SetMinSpeed(10);

        m_pInput->PushMouseListener(controller.get());
        m_pInput->PushKeyListener(controller.get());

        m_pLogic->AttachGameView(std::shared_ptr<chimera::HumanGameView>(m_pHumanView), camera);
        m_pLogic->AttachGameView(controller, camera);

        chimera::PostInitMessage("Loading GUI ...");
        chimera::gui::GUI* c = m_pHumanView->GetGUI();

        chimera::CMDimension dim;
        chimera::gui::GuiSpriteComponent* crossHair = new chimera::gui::GuiSpriteComponent(0, 0, 8, 8);
        dim.x = chimera::g_width / 2 - 4;
        dim.y = chimera::g_height / 2 - 4;
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

        chimera::g_pApp->GetScript()->VRunFile("files/scripts/packmanai.lua");

        chimera::g_pApp->GetLogic()->GetCommandInterpreter()->CallCommand("target free");
    }
}