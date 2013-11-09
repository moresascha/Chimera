#include "Application.h"
#include "../Resource.h"
#include "Config.h"
#include "Cache.h"
#include "util.h"
#include "Timer.h"

namespace chimera
{
    Application* g_pApp = NULL;
    HINSTANCE Application::g_hInstance = NULL;

    LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

    Application::Application(VOID)
    {
        g_pApp = this;
        m_running = TRUE;
    }

    BOOL Application::VInitialise(FactoryPtr* facts)
    {
        m_pConfig = new util::Config();

        RETURN_IF_FAILED(m_pConfig->VLoadConfigFile(CM_CONFIG_FILE));

        chimera::util::InitGdiplus(); //TODO

        IInputFactory* iif = FindFactory<IInputFactory>(facts, CM_FACTORY_INPUT);
        RETURN_IF_FAILED(iif);

        m_pInput = iif->VCreateInputHanlder();

        IEventFactory* iemf = FindFactory<IEventFactory>(facts, CM_FACTORY_EVENT);
        RETURN_IF_FAILED(iemf);
        m_pEventMgr = iemf->VCreateEventManager();

        IResourceFactory* rcf = FindFactory<IResourceFactory>(facts, CM_FACTROY_CACHE);
        RETURN_IF_FAILED(rcf);

        m_pCache = rcf->VCreateCache();

        ILogicFactory* logicf = FindFactory<ILogicFactory>(facts, CM_FACTORY_LOGIC);
        RETURN_IF_FAILED(logicf);
        m_pLogic = logicf->VCreateLogic();
        m_pLogic->VInitialise(facts);

        m_timer0 = new chimera::util::Timer();
        m_pUpdateTimer = new chimera::util::Timer();
        m_pRenderingTimer = new chimera::util::Timer();

        return TRUE;
    }

    VOID Application::VClose(VOID)
    {
        SAFE_DELETE(m_timer0);
        SAFE_DELETE(m_pUpdateTimer);
        SAFE_DELETE(m_pRenderingTimer);
        SAFE_DELETE(m_pLogic);
        chimera::util::DestroyGdiplus(); //TODO
        SAFE_DELETE(m_pInput);
        SAFE_DELETE(m_pCache);
        SAFE_DELETE(m_pEventMgr);
        SAFE_DELETE(m_pConfig);
    }

    VOID Application::Render(VOID)
    {
        m_pLogic->VGetHumanView()->VOnRender();
        m_pRenderingTimer->VTick(); 
    }

    VOID Application::Update(VOID) 
    {
        m_timer0->VTick();
        ULONG time = m_timer0->VGetTime();
        if(time < m_updateFreqMillis)
        {
            return;
        }
        m_timer0->VReset();
        //logic update
        m_pInput->VOnUpdate();
        VGetEventManager()->VUpdate();
        VGetLogic()->VOnUpdate(m_pUpdateTimer->VGetLastMillis());
        m_pUpdateTimer->VTick();
    }

    static BOOL s_mimnimized = FALSE;
    static BOOL s_initialized = FALSE;
    VOID Application::VRun(VOID)
    {
        MSG msg;
        m_running = TRUE;
        s_initialized = TRUE;
        m_pUpdateTimer->VReset();
        m_pRenderingTimer->VReset();

        HACCEL hAccelTable;

        hAccelTable = LoadAccelerators(g_hInstance, MAKEINTRESOURCE(IDR_ACCELERATOR1));

        while(VIsRunning()) 
        {
            if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
            {
                if(!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            } 
            else 
            {
                g_pApp->Update();
                if(!s_mimnimized) //todo
                {
                    g_pApp->Render(); 
                }
            }
        }
    }

    HWND Application::VGetWindowHandle(VOID)
    {
        return m_pLogic->VGetHumanView()->VGetRenderer()->VGetWindowHandle();
    }

    UINT Application::VGetWindowWidth(VOID) CONST
    {
        return m_pLogic->VGetHumanView()->VGetRenderer()->VGetWidth();
    }

    UINT Application::VGetWindowHeight(VOID) CONST
    {
        return m_pLogic->VGetHumanView()->VGetRenderer()->VGetHeight();
    }

    IHumanView* Application::VGetHumanView(VOID)
    {
        return m_pLogic->VGetHumanView();
    }

    Application::~Application(VOID)
    {

    }
    
    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        INT wmId, wmEvent;
        PAINTSTRUCT ps;
        HDC hdc;

        switch (message)
        {
        case WM_SIZE:
            {
                if(s_initialized)
                {
                    UINT w = LOWORD(lParam);
                    UINT h = HIWORD(lParam);
                    s_mimnimized = wParam == SIZE_MINIMIZED;
                    if(!s_mimnimized)
                    {
                        CmGetApp()->VGetHumanView()->VOnResize(w, h);
                    }
                }
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
            chimera::CmGetApp()->VGetInputHandler()->VOnMessage(msg);
            break;
        case WM_SYSKEYDOWN:
            {
                break;
            }
        case WM_COMMAND:
            wmId    = LOWORD(wParam);
            wmEvent = HIWORD(wParam);
            return DefWindowProc(hWnd, message, wParam, lParam);
            break;
        case WM_PAINT:
            hdc = BeginPaint(hWnd, &ps);
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

/*
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
#include "FileSystem.h"

#ifdef _DEBUG
#pragma comment(lib, "Cudahx64Debug.lib")
#else
#pragma comment(lib, "Cudahx64Release.lib")
#endif

namespace chimera
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

            std::shared_ptr<chimera::WatchShaderFileModificationProcess> modProc = 
                std::shared_ptr<chimera::WatchShaderFileModificationProcess>(new chimera::WatchShaderFileModificationProcess(prog, finalName.c_str(), L"../Assets/shader/"));
            chimera::g_pApp->GetLogic()->AttachProcess(modProc);
        }
    }
#endif

    VOID PostInitMessage(LPCSTR message)
    {
        CONST FLOAT c[] = {0,0,0,0};
        chimera::g_pContext->ClearRenderTargetView(chimera::g_pBackBufferView, c);
        chimera::g_pApp->GetFontManager()->RenderText(message, 12.0f / (FLOAT)chimera::g_pApp->GetWindowWidth(), 12.0f / (FLOAT)chimera::g_pApp->GetWindowHeight());
        chimera::g_pSwapChain->Present(0, 0);
    }

    VOID LogToConsole(std::string message)
    {
        std::vector<std::string> line = util::split(message, '\n');
        TBD_FOR(line)
        {
            chimera::g_pApp->GetHumanView()->GetConsole()->AppendText(*it);
        }
    }

    VOID LogToIDEConsole(std::string message)
    {
        DEBUG_OUT(message.c_str());
    }

    class DI : public chimera::IInputHandler
    {
        BOOL VOnMessage(CONST MSG& msg) { return TRUE; }

        BOOL VInit(HINSTANCE hinstance, HWND hwnd, UINT width, UINT height) { return TRUE; }

        BOOL VOnRestore(VOID) { return TRUE; }

        BOOL VIsEscPressed(VOID) { return TRUE; }

        BOOL VGrabMouse(BOOL aq) { return TRUE; }

        BOOL VIsMouseGrabbed(VOID) { return TRUE; }

        VOID VOnUpdate(VOID) { }
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

    BOOL GameApp::VInit(CONST HINSTANCE hInstance, LPCWSTR title, std::string resCache, std::string logFile, ULONG ival) 
    {
        Logger::Init(logFile);
        g_hInstance = hInstance;
        m_updateInterval = ival;
        m_updateFreqMillis = 1000.0 / (DOUBLE)m_updateInterval;

        m_pFileSystem = new chimera::FileSystem();

        DI di; //to avoid crashes, this does nothing than not throwing a NULL pointer...
        m_pInput = &di;

        //Logger::s_pLogMgr->SetWriteCallBack(LogToIDEConsole);

        if(!m_pConfig->LoadConfigFile("config.ini"))
        {
            LOG_CRITICAL_ERROR("Failed to init config file");
            return FALSE;
        }

        BOOL fullscreen = m_config.GetBool("bFullscreen");
        UINT width = (UINT)m_config.GetInteger("iWidth");
        UINT height = (UINT)m_config.GetInteger("iHeight");

        if(!chimera::Init(WndProc, hInstance, title, width, height))
        {
            LOG_CRITICAL_ERROR("Failed to init Direct3D");
            return FALSE;
        }

        if(!cudah::Init(chimera::g_pDevice))
        {
            LOG_CRITICAL_ERROR("Failed to load Cuda device");
            return FALSE;
        }

        //clear screen to black TODO: some image?
        CONST FLOAT c[] = {0,0,0,0};
        chimera::g_pContext->ClearRenderTargetView(chimera::g_pBackBufferView, c);
        chimera::g_pContext->ClearDepthStencilView(chimera::g_pDepthStencilView, D3D11_CLEAR_STENCIL | D3D11_CLEAR_DEPTH, 1, 1);
        chimera::g_pSwapChain->Present(0, 0);

        m_pEventMgr = new chimera::EventManager(TRUE);
 
        m_pCache = new chimera::ResourceCache(1000, new chimera::ResourceFolder(resCache));
    
        if(!m_pCache->Init())
        {
            LOG_CRITICAL_ERROR("Failed to init Cache");
            return FALSE;
        }

        m_pVramManager = new chimera::VRamManager(500);

        m_pRenderer = new chimera::D3DRenderer();

        m_pCache->RegisterLoader(std::shared_ptr<chimera::IResourceLoader>(new chimera::ImageLoader("png", m_config.GetString("sTexturePath"))));
        m_pCache->RegisterLoader(std::shared_ptr<chimera::IResourceLoader>(new chimera::ImageLoader("jpg", m_config.GetString("sTexturePath"))));
        m_pCache->RegisterLoader(std::shared_ptr<chimera::IResourceLoader>(new chimera::ObjLoader(m_config.GetString("sMeshPath"))));
        m_pCache->RegisterLoader(std::shared_ptr<chimera::IResourceLoader>(new chimera::MaterialLoader(m_config.GetString("sMaterialPath"))));
        m_pCache->RegisterLoader(std::shared_ptr<chimera::IResourceLoader>(new chimera::WaveLoader(m_config.GetString("sSoundPath"))));

        if(FAILED(m_pRenderer->VOnRestore()))
        {
            LOG_CRITICAL_ERROR("Failed to create renderer");
            return FALSE;
        }

        chimera::Geometry::Create();

        chimera::ShaderProgram::Create();

        m_pFontManager = new chimera::FontManager();
        m_pFontRenderer = new chimera::D3DFontRenderer();
        m_pFontManager->VSetFontRenderer(m_pFontRenderer);

        chimera::IFont* f = new chimera::BMFont();

        if(!f->VCreate(m_config.GetString("sFontPath") + "font_16.fnt"))
        {
            LOG_CRITICAL_ERROR_A("Failed to create font %s", f->VGetStyle().name.c_str());
            return FALSE;
        }

        m_pFontManager->VAddFont("Arial", f);

        PostInitMessage("Creating Input ...");
        m_pInput = new chimera::DefaultWinInputHandler();
        if(!m_pInput->VInit(chimera::g_pApp->g_hInstance, chimera::g_hWnd, chimera::g_width, chimera::g_height))
        {
            LOG_CRITICAL_ERROR("failed to create input");
            return FALSE;
        }
   
        m_pInput->VSetCurserOffsets(fullscreen ? 0 : 8 , fullscreen ? 0 : 30); //todo, get systemmetrics

        m_pInput->VGrabMouse(fullscreen);

        chimera::PostInitMessage("Creating Script ...");
        m_pScript = new chimera::script::LuaScript();

        if(!m_pScript->VInit())
        {
            LOG_CRITICAL_ERROR("failed to init script system");
            return FALSE;
        }

        m_pScriptEventManager = new chimera::script::LuaScriptEventManager();

        chimera::script::internalexports::Register();

        //todo move this to view
        chimera::PointlightNode::Create();

        //todo move this to view
        chimera::SpotlightNode::Create();

        //todo move this to view
        chimera::EffectChain::StaticCreate();

        PostInitMessage("Loading Shader ...");
        std::shared_ptr<chimera::ShaderProgram> shader = chimera::ShaderProgram::CreateProgram("DefShader", L"DefShading.hlsl", "DefShading_VS", "DefShading_PS", NULL);
        shader->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shader->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shader->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        shader->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> shaderInstanced = chimera::ShaderProgram::CreateProgram("DefShaderInstanced", L"DefShading.hlsl", "DefShadingInstanced_VS", "DefShading_PS", NULL);
        shaderInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shaderInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        shaderInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        shaderInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        shaderInstanced->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> rttProgram = chimera::ShaderProgram::CreateProgram("ScreenQuad", L"ScreenQuad.hlsl", "RT_VS", "RT_PS", NULL);
        rttProgram->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        rttProgram->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        rttProgram->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> particles = chimera::ShaderProgram::CreateProgram("Particles", L"Particles.hlsl", "Particle_VS", "Particle_PS", NULL);
        particles->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        particles->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        particles->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        particles->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32A32_FLOAT);
        particles->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> boundingDebug = chimera::ShaderProgram::CreateProgram("BoundingGeo", L"BoundingGeo.hlsl", "Sphere_VS", "Sphere_PS", NULL);
        boundingDebug->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        boundingDebug->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        boundingDebug->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        boundingDebug->GenerateLayout();

#ifndef FAST_STARTUP

        std::shared_ptr<chimera::ShaderProgram> globalLighting = chimera::ShaderProgram::CreateProgram("GlobalLighting", L"Lighting.hlsl", "Lighting_VS", "GlobalLighting_PS", NULL);
        globalLighting->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        globalLighting->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        globalLighting->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> wireShader = chimera::ShaderProgram::CreateProgram("DefShaderWire", L"DefShading.hlsl", "DefShading_VS", "DefShadingWire_PS", NULL);
        wireShader->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShader->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShader->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        wireShader->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> wireShaderInstanced = chimera::ShaderProgram::CreateProgram("DefShaderWireInstanced", L"DefShading.hlsl", "DefShadingInstanced_VS", "DefShadingWire_PS", NULL);
        wireShaderInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShaderInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShaderInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        wireShaderInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        wireShaderInstanced->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> pointLight = chimera::ShaderProgram::CreateProgram("PointLight", L"Lighting.hlsl", "Lighting_VS", "PointLighting_PS", NULL);
        pointLight->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        pointLight->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        pointLight->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> spottLight = chimera::ShaderProgram::CreateProgram("SpotLight", L"Lighting.hlsl", "Lighting_VS", "SpotLighting_PS", NULL);
        spottLight->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spottLight->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        spottLight->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> sky = chimera::ShaderProgram::CreateProgram("Sky", L"Sky.hlsl", "Sky_VS", "Sky_PS", NULL);
        sky->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        sky->SetInputAttr("TEXCOORD", 1, 0, DXGI_FORMAT_R32G32_FLOAT);
        sky->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> render2cubemap = chimera::ShaderProgram::CreateProgram("PointLightShadowMap", L"PointLightShadowMap.hlsl", "RenderCubeMap_VS", "RenderCubeMap_PS", "RenderCubeMap_GS");
        render2cubemap->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemap->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemap->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        render2cubemap->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> spotLightShadow = chimera::ShaderProgram::CreateProgram("SpotLightShadowMap", L"SpotLightShadowMap.hlsl", "SpotLightShadow_VS", "SpotLightShadow_PS");
        spotLightShadow->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadow->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadow->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        spotLightShadow->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> render2cubemapInstanced = chimera::ShaderProgram::CreateProgram("PointLightShadowMapInstanced", L"PointLightShadowMap.hlsl", "RenderCubeMapInstanced_VS", "RenderCubeMap_PS", "RenderCubeMap_GS");
        render2cubemapInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        render2cubemapInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapInstanced->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> spotLightShadowInstanced = chimera::ShaderProgram::CreateProgram("SpotLightShadowMapInstanced", L"SpotLightShadowMap.hlsl", "SpotLightShadowInstanced_VS", "SpotLightShadow_PS");
        spotLightShadowInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadowInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadowInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        spotLightShadowInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        spotLightShadowInstanced->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> render2cubemapparticles = chimera::ShaderProgram::CreateProgram("PointLightShadowMap_Particles",  
            L"PointLightShadowMap.hlsl", "RenderCubeMapParticles_VS", "RenderCubeMap_PS", "RenderCubeMap_GS");

        render2cubemapparticles->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapparticles->SetInputAttrInstanced("TANGENT", 1, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        render2cubemapparticles->GenerateLayout();
    
        std::shared_ptr<chimera::ShaderProgram> transparancy = chimera::ShaderProgram::CreateProgram("Transparency", L"Transparency.hlsl", "Transparency_VS", "Transparency_PS", NULL);
        transparancy->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancy->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancy->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        transparancy->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> transparancyInstanced = chimera::ShaderProgram::CreateProgram("TransparencyInstanced", L"Transparency.hlsl", "TransparencyInstanced_VS", "Transparency_PS", NULL);
        transparancyInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancyInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancyInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        transparancyInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        transparancyInstanced->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> csm = chimera::ShaderProgram::CreateProgram("CSM", L"CascadedShadowMap.hlsl", "CSM_VS",  "CSM_PS", NULL);
        csm->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csm->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csm->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        csm->GenerateLayout();

        std::shared_ptr<chimera::ShaderProgram> csmInstanced = chimera::ShaderProgram::CreateProgram("CSM_Instanced", L"CascadedShadowMap.hlsl", "CSM_Instanced_VS", "CSM_PS", NULL);
        csmInstanced->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csmInstanced->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        csmInstanced->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        csmInstanced->SetInputAttrInstanced("TANGENT", 3, 1, DXGI_FORMAT_R32G32B32_FLOAT);
        csmInstanced->GenerateLayout();
#endif

        chimera::PostInitMessage("Creating Logic and View ...");
        VCreateLogicAndView();

#ifdef _DEBUG
        CreateShaderWatcher<chimera::VertexShader>();
        CreateShaderWatcher<chimera::PixelShader>();
        CreateShaderWatcher<chimera::GeometryShader>();

        ADD_EVENT_LISTENER(m_pCache, &chimera::ResourceCache::OnResourceChanged, chimera::ResourceChangedEvent::TYPE);
        ADD_EVENT_LISTENER(GetHumanView()->GetVRamManager(), &chimera::VRamManager::OnResourceChanged, chimera::ResourceChangedEvent::TYPE);
        ADD_EVENT_LISTENER((chimera::PhysX*)GetLogic()->GetPhysics(), &chimera::PhysX::OnResourceChanged, chimera::ResourceChangedEvent::TYPE);
#endif

        chimera::PostInitMessage("Loading Commands ...");
        chimera::commands::AddCommandsToInterpreter(*GetLogic()->GetCommandInterpreter());

        if(!chimera::CommandInterpreter::LoadCommands("controls.ini", *GetLogic()->GetCommandInterpreter()))
        {
            LOG_CRITICAL_ERROR("Failed to load controls");
        }

        m_timer0.Reset();
        m_updateTimer.Reset();
        m_renderingTimer.Reset();

        //Logger::s_pLogMgr->SetWriteCallBack(LogToConsole);

        chimera::SetFullscreenState(fullscreen, width, height);

        return TRUE;
    }

    HWND GameApp::GetWindowHandle(VOID)
    {
        return chimera::g_hWnd;
    }

    static BOOL s_mimnimized = FALSE;
    VOID GameApp::VRun(VOID)
    {
        MSG msg;

        g_pApp->GetUpdateTimer()->Reset();

        HACCEL hAccelTable;

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
                if(!s_mimnimized) //todo
                {
                    g_pApp->Render();

                    g_pApp->GetHumanView()->GetRenderer()->VPresent();   
                }
            }
        }
    }

    VOID GameApp::Update(VOID) 
    {
        m_timer0.Tick();
        ULONG time = m_timer0.GetTime();
        if(time < m_updateFreqMillis)
        {
            return;
        }
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
        return chimera::GetWindowWidth();
    }

    UINT GameApp::GetWindowHeight(VOID) CONST
    {
        return chimera::GetWindowHeight();
    }

    VOID GameApp::VClose(VOID)
    {
        SAFE_DELETE(m_pFileSystem);

        chimera::SpotlightNode::Destroy();

        chimera::PointlightNode::Destroy();

        chimera::EffectChain::StaticDestroy();

        GeometryFactory::Destroy();

        SAFE_DELETE(m_pScriptEventManager);

        SAFE_DELETE(m_pScript);

        SAFE_DELETE(m_pLogic);

        SAFE_DELETE(m_pVramManager);

        SAFE_DELETE(m_pCache);

        SAFE_DELETE(m_pEventMgr);

        chimera::ShaderProgram::Destroy();

        chimera::Geometry::Destroy();

        SAFE_DELETE(m_pFontRenderer);

        SAFE_DELETE(m_pFontManager);

        SAFE_DELETE(m_pInput);

        SAFE_DELETE(m_pRenderer);        

        cudah::Destroy();
        chimera::Release();
        Logger::Destroy();
    }

    GameApp::~GameApp(VOID)
    {
        Close();
    }

    VOID Close(VOID)
    {
        SAFE_DELETE(g_pApp);
    }
     */
}