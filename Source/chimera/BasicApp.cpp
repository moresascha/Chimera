#include "BasicApp.h"
#include "GameInputActions.h"
#include "ScreenElement.h"
#include "GraphicsSettings.h"
#include "D3DRenderer.h"
#include "Camera.h"
#include "DebugStartup.h"

namespace app
{
    VOID BasicApp::VCreateLogicAndView(VOID)
    {
        m_pLogic = new tbd::BaseGameLogic();

        if(!m_pLogic->VInit())
        {
            LOG_CRITICAL_ERROR("Gamelogic init fail");
        }

        m_pHumanView = new tbd::HumanGameView();

        tbd::Dimension dim;
        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width;
        dim.h = d3d::g_height;

        std::shared_ptr<tbd::DefaultGraphicsSettings> gs = std::shared_ptr<tbd::DefaultGraphicsSettings>(new tbd::DefaultGraphicsSettings());
        tbd::RenderScreen* screen = new tbd::RenderScreen(gs);
        screen->VSetName("default");
        screen->VSetDimension(dim);
        m_pHumanView->AddScene(screen);

        std::shared_ptr<tbd::EditorGraphicsSettings> egs = std::shared_ptr<tbd::EditorGraphicsSettings>(new tbd::EditorGraphicsSettings());
        screen = new tbd::RenderScreen(egs);
        screen->VSetName("editor");
        screen->VSetDimension(dim);
        m_pHumanView->AddScene(screen);

        std::shared_ptr<tbd::ProfileGraphicsSettings> pgs = std::shared_ptr<tbd::ProfileGraphicsSettings>(new tbd::ProfileGraphicsSettings());
        screen = new tbd::RenderScreen(pgs);
        screen->VSetName("profile");
        screen->VSetDimension(dim);
        m_pHumanView->AddScene(screen);

//#ifndef FAST_STARTUP
        /*

        std::shared_ptr<tbd::DebugGraphicsSettings> dgs = std::shared_ptr<tbd::DebugGraphicsSettings>(new tbd::DebugGraphicsSettings());
        screen = new tbd::RenderScreen(dgs);
        screen->VSetName("debug");
        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width;
        dim.h = d3d::g_height;
        screen->VSetDimension(dim);
        m_pHumanView->AddScene(screen);

        std::shared_ptr<tbd::EditorGraphicsSettings> egs = std::shared_ptr<tbd::EditorGraphicsSettings>(new tbd::EditorGraphicsSettings(gs.get()));
        screen = new tbd::RenderScreen(egs);
        screen->VSetName("editor");
        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width;
        dim.h = d3d::g_height;
        screen->VSetDimension(dim);
        m_pHumanView->AddScene(screen);

        std::shared_ptr<tbd::WireFrameFilledSettings> wfs = std::shared_ptr<tbd::WireFrameFilledSettings>(new tbd::WireFrameFilledSettings(gs.get()));
        screen = new tbd::RenderScreen(wfs);
        screen->VSetName("wire");
        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width;
        dim.h = d3d::g_height;
        screen->VSetDimension(dim);
        m_pHumanView->AddScene(screen); */
//#else

        /*
        std::shared_ptr<tbd::SimpleSettings> gs = std::shared_ptr<tbd::SimpleSettings>(new tbd::SimpleSettings(FALSE)); */

//#endif

        /*std::shared_ptr<tbd::DebugGraphicsSettings> dgs = std::shared_ptr<tbd::DebugGraphicsSettings>(new tbd::DebugGraphicsSettings());
        screen = new tbd::RenderScreen(dgs);
        screen->VSetName("debug");

        dim.x = d3d::g_width * 3 / 4;
        dim.y = 0;
        dim.w = d3d::g_width / 4;
        dim.h = d3d::g_height / 4;
        screen->VSetDimension(dim);
        m_pHumanView->AddScreenElement(screen); */

        /*
        tbd::DefShaderRenderScreenContainer* con = new tbd::DefShaderRenderScreenContainer(dgs);
        con->VSetName("defshader_targets");

        tbd::DefShaderRenderScreen* s = new tbd::DefShaderRenderScreen(d3d::Diff_NormalsTarget);
        dim.x = d3d::g_width / 2;
        dim.y = 0;
        dim.w = d3d::g_width / 4;
        dim.h = d3d::g_height / 4;
        s->VSetDimension(dim);
        con->AddComponent(s);

        s = new tbd::DefShaderRenderScreen(d3d::Diff_WorldPositionTarget);
        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width / 4;
        dim.h = d3d::g_height / 4;
        s->VSetDimension(dim);
        con->AddComponent(s);

        s = new tbd::DefShaderRenderScreen(d3d::Diff_DiffuseColorSpecBTarget);
        dim.x = d3d::g_width * 1 / 4;
        dim.y = 0;
        dim.w = d3d::g_width / 4;
        dim.h = d3d::g_height / 4;
        s->VSetDimension(dim);
        con->AddComponent(s); 

        con->VSetDimension(dim);

        m_pHumanView->AddScreenElement(con);*/

        /*std::shared_ptr<tbd::EditorGraphicsSettings> egs = std::shared_ptr<tbd::EditorGraphicsSettings>(new tbd::EditorGraphicsSettings(gs.get()));
        screen = new tbd::RenderScreen(egs);
        screen->VSetName("editor");

        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width / 4;
        dim.h = d3d::g_height / 4;
        screen->VSetDimension(dim);
        m_pHumanView->AddScreenElement(screen); */

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

        gameinput::RegisterCommands(*controller.get(), *GetLogic()->GetCommandInterpreter());

            /*m_pLogic->GetSoundEngine()->ResgisterSound("dynamic", "dynamic", "collision.wav");
            m_pLogic->GetSoundEngine()->ResgisterSound("dynamic", "static", "collision.wav");
            m_pLogic->GetSoundEngine()->ResgisterSound("dynamic", "bouncy", "collision.wav");
            m_pLogic->GetSoundEngine()->ResgisterSound("dynamic", "kinematic", "collision.wav"); */
            //m_pLogic->GetSoundEngine()->ResgisterSound("static", "kinematic", "collision.wav");

        app::PostInitMessage("Loading GUI ...");
        tbd::gui::D3D_GUI* c = m_pHumanView->GetGUI();

        tbd::gui::InformationWindow* wnd = new tbd::gui::InformationWindow();
        wnd->VSetBackgroundColor(0,0,0);
        wnd->VSetAlpha(0);

        dim.x = (INT)(0.8 * d3d::g_width);
        dim.y = 12;
        dim.w = (INT)(0.2 * d3d::g_width);
        dim.h = 400;
        wnd->VSetDimension(dim);
        c->AddComponent("informationwindow", wnd);

        tbd::gui::GuiSpriteComponent* crossHair = new tbd::gui::GuiSpriteComponent(0, 0, 8, 8);
        dim.x = d3d::g_width / 2 - 4;
        dim.y = d3d::g_height / 2 - 4;
        dim.h = 8;
        dim.w = 8;
        crossHair->VSetBackgroundColor(1,1,1);
        crossHair->VSetDimension(dim);
        crossHair->SetTexture("crosshair_dot.png");
        c->AddComponent("CH", crossHair);

        
        //c->AddComponent("histogram", histo);

        //tbd::BaseLevel* level = new tbd::BSplinePatchLevel("patch1", m_pLogic->GetActorFactory());
        //tbd::BaseLevel* level = new tbd::TransformShowRoom("patch0", m_pLogic->GetActorFactory());
        tbd::BaseLevel* level = new tbd::RandomLevel("rnd", m_pLogic->GetActorFactory());
        m_pLogic->VLoadLevel(level);
    }
}