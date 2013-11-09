#include "BasicApp.h"
#include "GameInputActions.h"
#include "ScreenElement.h"
#include "GraphicsSettings.h"
#include "D3DRenderer.h"
#include "Camera.h"
#include "DebugStartup.h"
#include "Components.h"
#include "util.h"
#include "Actor.h"
#include "GameLogic.h"
#include "Components.h"
#include "GameApp.h"
#include "GuiHelper.h"
#include "Input.h"
#include "Commands.h"
#include "d3d.h"

namespace chimera
{
    VOID BasicApp::VCreateLogicAndView(VOID)
    {
        m_pLogic = new chimera::BaseGameLogic();

        if(!m_pLogic->VInit())
        {
            LOG_CRITICAL_ERROR("Gamelogic init fail");
        }

        m_pHumanView = new chimera::HumanGameView();

#ifndef FAST_STARTUP

        std::shared_ptr<chimera::DefaultGraphicsSettings> gs = std::shared_ptr<chimera::DefaultGraphicsSettings>(new chimera::DefaultGraphicsSettings());
        chimera::RenderScreen* screen = new chimera::RenderScreen(gs);
        screen->VSetName("default");
        m_pHumanView->AddScene(screen);
        
        std::shared_ptr<chimera::EditorGraphicsSettings> egs = std::shared_ptr<chimera::EditorGraphicsSettings>(new chimera::EditorGraphicsSettings());
        screen = new chimera::RenderScreen(egs);
        screen->VSetName("editor");
        m_pHumanView->AddScene(screen);

        std::shared_ptr<chimera::ProfileGraphicsSettings> pgs = std::shared_ptr<chimera::ProfileGraphicsSettings>(new chimera::ProfileGraphicsSettings());
        screen = new chimera::RenderScreen(pgs);
        screen->VSetName("profile");
        m_pHumanView->AddScene(screen);

        std::shared_ptr<chimera::BoundingGeoDebugSettings> bgs = std::shared_ptr<chimera::BoundingGeoDebugSettings>(new chimera::BoundingGeoDebugSettings());
        screen = new chimera::RenderScreen(bgs);
        screen->VSetName("bounding");
        m_pHumanView->AddScene(screen);
#else


#endif
        chimera::CMDimension dim;
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


        std::shared_ptr<chimera::Actor> camera = m_pLogic->VCreateActor("camera.xml");
        camera->SetName("player");
#ifndef FAST_STARTUP
        //flashlight
        chimera::ActorDescription desc = GetLogic()->GetActorFactory()->CreateActorDescription();
        chimera::LightComponent* lightComponent = desc->AddComponent<chimera::LightComponent>("LightComponent");
        lightComponent->m_type = "spot";
        lightComponent->m_color.x = 1;
        lightComponent->m_color.y = 1;
        lightComponent->m_color.z = 1;
        lightComponent->m_color.w = 1;
        lightComponent->m_angle = 60;
        lightComponent->m_intensity = 15;
        chimera::TransformComponent* tcomp = desc->AddComponent<chimera::TransformComponent>(chimera::TransformComponent::COMPONENT_ID);
        tcomp->GetTransformation()->SetScale(20);
        tcomp->GetTransformation()->Translate(0, 1.5f, 0);
        chimera::ParentComponent* pc = desc->AddComponent<chimera::ParentComponent>("ParentComponent");
        pc->m_parentId = camera->GetId();
        m_pLogic->VCreateActor(desc);
        //flashlight end
#endif // !FAST_STARTUP

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

        gameinput::RegisterCommands(*controller.get(), *GetLogic()->GetCommandInterpreter());

        chimera::g_pApp->GetHumanView()->GetSoundEngine()->RegisterSound("dynamic", "dynamic", "collision.wav");
        chimera::g_pApp->GetHumanView()->GetSoundEngine()->RegisterSound("dynamic", "static", "collision.wav");
        chimera::g_pApp->GetHumanView()->GetSoundEngine()->RegisterSound("dynamic", "bouncy", "collision.wav");
        chimera::g_pApp->GetHumanView()->GetSoundEngine()->RegisterSound("dynamic", "kinematic", "collision.wav");
        chimera::g_pApp->GetHumanView()->GetSoundEngine()->RegisterSound("static", "kinematic", "collision.wav");

        chimera::PostInitMessage("Loading GUI ...");
        chimera::gui::GUI* c = m_pHumanView->GetGUI();

        chimera::gui::InformationWindow* wnd = new chimera::gui::InformationWindow();
        wnd->VSetBackgroundColor(0,0,0);
        wnd->VSetAlpha(0);
        
        dim.x = (INT)(0.8 * chimera::g_width);
        dim.y = 12;
        dim.w = (INT)(0.2 * chimera::g_width);
        dim.h = 400;
        wnd->VSetDimension(dim);
        c->AddComponent("informationwindow", wnd);

        chimera::gui::GuiSpriteComponent* crossHair = new chimera::gui::GuiSpriteComponent(0, 0, 7, 7);
        dim.x = (chimera::g_width / 2 - 3);
        dim.y = (chimera::g_height / 2 - 3);
        dim.h = 7;
        dim.w = 7;
        crossHair->VSetBackgroundColor(1,1,1);
        crossHair->VSetDimension(dim);
        crossHair->VSetName("CH");
        crossHair->SetTexture("crosshair_cross.png");
        c->AddComponent("CH", crossHair);

        /*tbd::ActorPicker* ap = (tbd::ActorPicker*)app::g_pApp->GetHumanView()->GetPicker();
        tbd::RendertargetScreen* shadowMap = new tbd::RendertargetScreen(ap->GetTarget());

        dim.x = 0;
        dim.y = 0;
        dim.w = d3d::g_width / 4;
        dim.h = d3d::g_height / 4;
        shadowMap->VSetDimension(dim);
        m_pHumanView->AddScreenElement(shadowMap);
        m_pHumanView->SetName("tmpmap"); */
        
        //c->AddComponent("histogram", histo);

        //tbd::BaseLevel* level = new tbd::BSplinePatchLevel("patch1", m_pLogic->GetActorFactory());
        //tbd::BaseLevel* level = new tbd::TransformShowRoom("patch0", m_pLogic->GetActorFactory());
        
        //tbd::BaseLevel* level = new tbd::GroupedObjLevel("grouped", m_pLogic->GetActorFactory()); m_pLogic->VLoadLevel(level);
        //m_pLogic->VLoadLevel("test.xml");
        m_pLogic->VLoadLevel(new chimera::RandomLevel("RNG", m_pLogic->GetActorFactory()));
    }
}