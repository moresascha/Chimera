#include "../Resource.h"
#include "GameApp.h"
#include "GameLogic.h"
#include "Resources.h"
#include "EventManager.h"
#include "GameView.h"
#include "Input.h"
#include "tbdFont.h"
#include "Commands.h"
#include "d3d.h"
#include "Logger.h"
#include "GeometryFactory.h"
#include "tbdFont.h"
#include "PointLightNode.h"
#include "cudah.h"
#include "Effect.h"
#include "util.h"
#include "ShaderProgram.h"
#include "D3DRenderer.h"
#include "VRamManager.h"
#include "Script.h"
#include "GuiComponent.h"
#include "ScriptEvent.h"
#include "DebugStartup.h"
#include "SpotlightNode.h"

namespace app
{
    GameApp* g_pApp = NULL;
    HINSTANCE GameApp::g_hInstance = NULL;

    LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
    INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);

#ifdef _DEBUG
    template <class T>
    VOID CreateShaderWatcher(VOID)
    {
        for(auto it = T::GetShader()->begin(); it != T::GetShader()->end(); ++it)
        {
            T* prog = it->second;
            std::string name = it->first;

            //really nasty string conversation shit
            std::wstring ws(prog->GetFile());
            std::string s(ws.begin(), ws.end());
            std::vector<std::string> split = util::split(s, '/');
            std::string ss = split[split.size() - 1];
            std::wstring finalName(ss.begin(), ss.end());

            std::shared_ptr<proc::WatchShaderFileModificationProcess> modProc = 
                std::shared_ptr<proc::WatchShaderFileModificationProcess>(new proc::WatchShaderFileModificationProcess(prog, finalName.c_str(), L"../Assets/shader/"));
            app::g_pApp->GetLogic()->AttachProcess(modProc);
        }
    }
#endif

    VOID PostInitMessage(LPCSTR message)
    {
        float c[] = {0,0,0,0};
        d3d::g_pContext->ClearRenderTargetView(d3d::g_pBackBufferView, c);
        app::g_pApp->GetFontManager()->RenderText(message, 12.0f / (FLOAT)d3d::g_width, 12.0f / (FLOAT)d3d::g_height);
        d3d::g_pSwapChain->Present(0, 0);
    }

    VOID LogToConsole(std::string message)
    {
        std::vector<std::string> line = util::split(message, '\n');
        TBD_FOR(line)
        {
            app::g_pApp->GetHumanView()->GetConsole()->AppendText(*it);
        }
    }

    VOID LogToIDEConsole(std::string message)
    {
        DEBUG_OUT(message.c_str());
    }

    class DI : public tbd::InputHandler
    {
        BOOL VOnMessage(CONST MSG& msg) { return TRUE; }

        BOOL VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height) { return TRUE; }

        BOOL VOnRestore(VOID) { return TRUE; }

        BOOL VIsEscPressed(VOID) { return TRUE; }

        BOOL VGrabMouse(BOOL aq) { return TRUE; }

        BOOL VIsMouseGrabbed(VOID) { return TRUE; }
    };

    GameApp::GameApp(VOID) 
    {
        if(g_pApp)
        {
            LOG_CRITICAL_ERROR("Only one GameApp allowed!");
        }
        g_pApp = this;
        m_pLogic = NULL;
        m_pCache = NULL;
        m_pEventMgr = NULL;
        m_pHumanView = NULL;
        m_pFontManager = NULL;
        m_pFontRenderer = NULL;
        m_pInput = NULL;
        m_pRenderer = NULL;
        m_pVramManager = NULL;
        m_pScript = NULL;
        m_pScriptEventManager = NULL;
    }

    BOOL GameApp::Init(CONST HINSTANCE hInstance, LPCWSTR title, std::string resCache, std::string logFile, ULONG ival) 
    {
        Logger::Init(logFile);
        g_hInstance = hInstance;
        m_updateInterval = ival;
        m_updateFreqMillis = 1000.0 / (DOUBLE)m_updateInterval;

        DI di; //to avoid crashes, this does nothing than not throwing an NULL pointer...
        m_pInput = &di;

        //Logger::s_pLogMgr->SetWriteCallBack(LogToIDEConsole);

        if(!m_config.LoadConfigFile("config.ini"))
        {
            LOG_CRITICAL_ERROR("Failed to init config file");
            return FALSE;
        }

        UINT width = (UINT)m_config.GetInteger("iWidth");
        UINT height = (UINT)m_config.GetInteger("iHeight");

        if(!d3d::Init(WndProc, hInstance, title, width, height))
        {
            LOG_CRITICAL_ERROR("Failed to init Direct3D");
            return FALSE;
        }

        BOOL fullscreen = m_config.GetBool("bFullscreen");
        if(fullscreen)
        {
            d3d::Resize(0,0, fullscreen);
        }

        if(!cudah::Init(d3d::g_pDevice))
        {
            LOG_CRITICAL_ERROR("Failed to load Cuda device");
            return FALSE;
        }

        //clear screen to black TODO: some image?
        float c[] = {0,0,0,0};
        d3d::g_pContext->ClearRenderTargetView(d3d::g_pBackBufferView, c);
        d3d::g_pContext->ClearDepthStencilView(d3d::g_pDepthStencilView, D3D11_CLEAR_STENCIL | D3D11_CLEAR_DEPTH, 1, 1);
        d3d::g_pSwapChain->Present(0, 0);

        m_pEventMgr = new event::EventManager(TRUE);
 
        m_pCache = new tbd::ResourceCache(1000, new tbd::ResourceFolder(resCache));
    
        if(!m_pCache->Init())
        {
            LOG_CRITICAL_ERROR("Failed to init Cache");
            return FALSE;
        }

        m_pVramManager = new tbd::VRamManager(500);

        m_pRenderer = new d3d::D3DRenderer();

        m_pCache->RegisterLoader(std::shared_ptr<tbd::IResourceLoader>(new tbd::ImageLoader("png", m_config.GetString("sTexturePath"))));
        m_pCache->RegisterLoader(std::shared_ptr<tbd::IResourceLoader>(new tbd::ImageLoader("jpg", m_config.GetString("sTexturePath"))));
        m_pCache->RegisterLoader(std::shared_ptr<tbd::IResourceLoader>(new tbd::ObjLoader(m_config.GetString("sMeshPath"))));
        m_pCache->RegisterLoader(std::shared_ptr<tbd::IResourceLoader>(new tbd::MaterialLoader(m_config.GetString("sMaterialPath"))));
        m_pCache->RegisterLoader(std::shared_ptr<tbd::IResourceLoader>(new tbd::WaveLoader(m_config.GetString("sSoundPath"))));

        if(FAILED(m_pRenderer->VOnRestore()))
        {
            LOG_CRITICAL_ERROR("Failed to create renderer");
            return FALSE;
        }

        d3d::Geometry::Create();

        d3d::ShaderProgram::Create();

        m_pFontManager = new tbd::FontManager();
        m_pFontRenderer = new tbd::D3DFontRenderer();
        m_pFontManager->SetFontRenderer(m_pFontRenderer);

        tbd::IFont* f = new tbd::BMFont();

        if(!f->VCreate(m_config.GetString("sFontPath") + "font_16.fnt"))
        {
            LOG_CRITICAL_ERROR_A("Failed to create font %s", f->VGetStyle().name.c_str());
            return FALSE;
        }

        m_pFontManager->AddFont("Arial", f);

        PostInitMessage("Creating Input ...");
        m_pInput = new tbd::DefaultWinInputHandler();
        if(!m_pInput->VInit(app::g_pApp->g_hInstance, d3d::g_hWnd, d3d::g_width, d3d::g_height))
        {
            LOG_CRITICAL_ERROR("failed to create input");
            return FALSE;
        }
   
        m_pInput->VSetCurserOffsets(fullscreen ? 0 : 8 , fullscreen ? 0 : 30); //todo, get systemmetrics

        m_pInput->VGrabMouse(fullscreen);

        app::PostInitMessage("Creating Script ...");
        m_pScript = new tbd::script::LuaScript();

        if(!m_pScript->VInit())
        {
            LOG_CRITICAL_ERROR("failed to init script system");
            return FALSE;
        }

        m_pScriptEventManager = new tbd::script::LuaScriptEventManager();

        tbd::script::internalexports::Register();

        //todo move this to view
        tbd::PointlightNode::Create();

        //todo move this to view
        tbd::SpotlightNode::Create();

        //todo move this to view
        d3d::EffectChain::StaticCreate();

        PostInitMessage("Loading Shader ...");
        std::shared_ptr<d3d::ShaderProgram> shader = d3d::ShaderProgram::CreateProgram("DefShader", L"DefShading.hlsl", "DefShading_VS", "DefShading_PS", NULL);
        shader->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shader->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shader->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        shader->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> shaderInstanced = d3d::ShaderProgram::CreateProgram("DefShaderInstanced", L"DefShading.hlsl", "DefShadingInstanced_VS", "DefShading_PS", NULL);
        shaderInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shaderInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shaderInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        shaderInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        shaderInstanced->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> rttProgram = d3d::ShaderProgram::CreateProgram("ScreenQuad", L"ScreenQuad.hlsl", "RT_VS", "RT_PS", NULL);
        rttProgram->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        rttProgram->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        rttProgram->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> particles = d3d::ShaderProgram::CreateProgram("Particles", L"Particles.hlsl", "Particle_VS", "Particle_PS", NULL);
        particles->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        particles->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        particles->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        particles->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32A32_FLOAT);
        particles->GenerateLayout();

#ifndef FAST_STARTUP

        std::shared_ptr<d3d::ShaderProgram> globalLighting = d3d::ShaderProgram::CreateProgram("GlobalLighting", L"Lighting.hlsl", "Lighting_VS", "GlobalLighting_PS", NULL);
        globalLighting->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        globalLighting->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        globalLighting->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> wireShader = d3d::ShaderProgram::CreateProgram("DefShaderWire", L"DefShading.hlsl", "DefShading_VS", "DefShadingWire_PS", NULL);
        wireShader->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShader->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShader->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        wireShader->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> wireShaderInstanced = d3d::ShaderProgram::CreateProgram("DefShaderWireInstanced", L"DefShading.hlsl", "DefShadingInstanced_VS", "DefShadingWire_PS", NULL);
        wireShaderInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShaderInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShaderInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        wireShaderInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShaderInstanced->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> pointLight = d3d::ShaderProgram::CreateProgram("PointLight", L"Lighting.hlsl", "Lighting_VS", "PointLighting_PS", NULL);
        pointLight->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        pointLight->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        pointLight->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> spottLight = d3d::ShaderProgram::CreateProgram("SpotLight", L"Lighting.hlsl", "Lighting_VS", "SpotLighting_PS", NULL);
        spottLight->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spottLight->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        spottLight->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> boundingDebug = d3d::ShaderProgram::CreateProgram("BoundingGeo", L"BoundingGeo.hlsl", "Sphere_VS", "Sphere_PS", NULL);
        boundingDebug->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        boundingDebug->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        boundingDebug->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        boundingDebug->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> sky = d3d::ShaderProgram::CreateProgram("Sky", L"Sky.hlsl", "Sky_VS", "Sky_PS", NULL);
        sky->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        sky->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        sky->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> render2cubemap = d3d::ShaderProgram::CreateProgram("PointLightShadowMap", L"PointLightShadowMap.hlsl", "RenderCubeMap_VS", "RenderCubeMap_PS", "RenderCubeMap_GS");
        render2cubemap->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemap->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemap->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        render2cubemap->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> spotLightShadow = d3d::ShaderProgram::CreateProgram("SpotLightShadowMap", L"SpotLightShadowMap.hlsl", "SpotLightShadow_VS", "SpotLightShadow_PS");
        spotLightShadow->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadow->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadow->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        spotLightShadow->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> render2cubemapInstanced = d3d::ShaderProgram::CreateProgram("PointLightShadowMapInstanced", L"PointLightShadowMap.hlsl", "RenderCubeMapInstanced_VS", "RenderCubeMap_PS", "RenderCubeMap_GS");
        render2cubemapInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        render2cubemapInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapInstanced->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> spotLightShadowInstanced = d3d::ShaderProgram::CreateProgram("SpotLightShadowMapInstanced", L"SpotLightShadowMap.hlsl", "SpotLightShadowInstanced_VS", "SpotLightShadow_PS");
        spotLightShadowInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadowInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadowInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        spotLightShadowInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadowInstanced->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> render2cubemapparticles = d3d::ShaderProgram::CreateProgram("PointLightShadowMap_Particles",  
            L"PointLightShadowMap.hlsl", "RenderCubeMapParticles_VS", "RenderCubeMap_PS", "RenderCubeMap_GS");

        render2cubemapparticles->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapparticles->SetInputAttrInstanced("TANGENT", 1, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapparticles->GenerateLayout();
    
        std::shared_ptr<d3d::ShaderProgram> transparancy = d3d::ShaderProgram::CreateProgram("Transparency", L"Transparency.hlsl", "Transparency_VS", "Transparency_PS", NULL);
        transparancy->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancy->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancy->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        transparancy->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> transparancyInstanced = d3d::ShaderProgram::CreateProgram("TransparencyInstanced", L"Transparency.hlsl", "TransparencyInstanced_VS", "Transparency_PS", NULL);
        transparancyInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancyInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancyInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        transparancyInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancyInstanced->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> csm = d3d::ShaderProgram::CreateProgram("CSM", L"CascadedShadowMap.hlsl", "CSM_VS",  "CSM_PS", NULL);
        csm->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csm->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csm->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        csm->GenerateLayout();

        std::shared_ptr<d3d::ShaderProgram> csmInstanced = d3d::ShaderProgram::CreateProgram("CSM_Instanced", L"CascadedShadowMap.hlsl", "CSM_Instanced_VS", "CSM_PS", NULL);
        csmInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csmInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csmInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        csmInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        csmInstanced->GenerateLayout();
#endif

        app::PostInitMessage("Creating Logic and View ...");
        VCreateLogicAndView();

#ifdef _DEBUG
        CreateShaderWatcher<d3d::VertexShader>();
        CreateShaderWatcher<d3d::PixelShader>();
        CreateShaderWatcher<d3d::GeometryShader>();
#endif

        app::PostInitMessage("Loading Commands ...");
        tbd::commands::AddCommandsToInterpreter(*GetLogic()->GetCommandInterpreter());

        if(!tbd::CommandInterpreter::LoadCommands("controls.ini", *GetLogic()->GetCommandInterpreter()))
        {
            LOG_CRITICAL_ERROR("Failed to load controls");
        }

        m_timer0.Reset();
        m_updateTimer.Reset();
        m_renderingTimer.Reset();

        //Logger::s_pLogMgr->SetWriteCallBack(LogToConsole);

        return TRUE;
    }

    HWND GameApp::GetWindowHandle(VOID)
    {
        return d3d::g_hWnd;
    }

    VOID GameApp::Run(VOID)
    {
        MSG msg;

        g_pApp->GetUpdateTimer()->Reset();

        HACCEL hAccelTable;

        // Initialize global strings
        /*LoadString(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
        LoadString(hInstance, IDC_D3D, szWindowClass, MAX_LOADSTRING); */

        hAccelTable = LoadAccelerators(g_hInstance, MAKEINTRESOURCE(IDR_ACCELERATOR1));

        while(g_pApp->IsRunning()) 
        {
            if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
            {
                if(!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                if(msg.message == WM_QUIT || msg.wParam == VK_ESCAPE) 
                {
                    g_pApp->End();
                    continue;
                }
            } 
            else 
            {
                g_pApp->Update();

                if(g_pApp->GetInputHandler()->VIsEscPressed())
                {
                    g_pApp->End();
                }

                g_pApp->Render();

                g_pApp->GetHumanView()->GetRenderer()->VPresent();
            }
        }
    }

    VOID GameApp::Update(VOID) 
    {
        m_timer0.Tick();
        ULONG time = m_timer0.GetTime();
        //DEBUG_OUT_A("%u\n", time);
        if(time < m_updateFreqMillis) // - m_renderingTimer.GetLastMillis() TODO
        {
            return;
        }
        //LOG_INFO_A("%u", m_timer0.GetTime());
        //LOG_INFO("---");
        m_timer0.Reset();
        //logic update
        m_pInput->OnUpdate();
        this->GetEventMgr()->VUpdate();
        this->GetLogic()->VOnUpdate(time);
        m_pVramManager->Update(time);
        m_updateTimer.Tick();
    }

    VOID GameApp::Render(VOID)
    {
        cuCtxSynchronize();//TODO
        this->GetLogic()->VOnRender();
        m_renderingTimer.Tick();
    }

    UINT GameApp::GetWindowWidth(VOID) CONST
    {
        return d3d::GetWindowWidth();
    }

    UINT GameApp::GetWindowHeight(VOID) CONST
    {
        return d3d::GetWindowHeight();
    }

    GameApp::~GameApp(VOID)
    {

        tbd::SpotlightNode::Destroy();

        tbd::PointlightNode::Destroy();

        d3d::EffectChain::StaticDestroy();

        GeometryFactory::Destroy();

        SAFE_DELETE(m_pScriptEventManager);

        SAFE_DELETE(m_pScript);

        SAFE_DELETE(m_pLogic);

        SAFE_DELETE(m_pVramManager);

        SAFE_DELETE(m_pCache);

        SAFE_DELETE(m_pEventMgr);

        d3d::ShaderProgram::Destroy();

        d3d::Geometry::Destroy();

        SAFE_DELETE(m_pFontRenderer);

        SAFE_DELETE(m_pFontManager);

        SAFE_DELETE(m_pInput);

        SAFE_DELETE(m_pRenderer);        

        cudah::Destroy();
        d3d::Release();
        Logger::Destroy();

        g_pApp = NULL; //there should be only one
    }

    //windows callbacks

    //
    //  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
    //
    //  PURPOSE:  Processes messages for the main window.
    //
    //  WM_COMMAND     - process the application menu
    //  WM_PAINT     - Paint the main window
    //  WM_DESTROY     - post a quit message and return
    //
    //
    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {

        int wmId, wmEvent;
        PAINTSTRUCT ps;
        HDC hdc;

        switch (message)
        {
        case WM_SIZE:
            {
                UINT w = LOWORD(lParam);
                UINT h = HIWORD(lParam);
                BOOL fullscreen = (wParam == SIZE_MAXIMIZED);
                //app::g_pApp->GetHumanView()->Resize(w, h, fullscreen); todo
            } break;

        case WM_MOUSEMOVE:
        case WM_KEYDOWN:
        case WM_KEYUP:
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MOUSEWHEEL:
            MSG msg;
            msg.hwnd = hWnd;
            msg.lParam = lParam;
            msg.wParam = wParam;
            msg.message = message;

            g_pApp->GetInputHandler()->VOnMessage(msg);
            break;
        case WM_SYSKEYDOWN:
            {
                break;
            }
        case WM_COMMAND:
            wmId    = LOWORD(wParam);
            wmEvent = HIWORD(wParam);
            // Parse the menu selections:
            //switch (wmId)
            {
            /*case IDM_ABOUT:
                //DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break; */
            //default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
            break;
        case WM_PAINT:
            hdc = BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code here...
            EndPaint(hWnd, &ps);
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            break;
        } 
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    // Message handler for about box.
    INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
    {
        UNREFERENCED_PARAMETER(lParam);
        switch (message)
        {
        case WM_INITDIALOG:
            return (INT_PTR)TRUE;

        case WM_COMMAND:
            if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
            {
                EndDialog(hDlg, LOWORD(wParam));
                return (INT_PTR)TRUE;
            }
            break;
        }
        return (INT_PTR)FALSE;
    }
}