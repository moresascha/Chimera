#include "api/ChimeraAPI.h"
#include "Application.h"
#include "D3DGraphics.h"
#include "APIError.h"

#include "GameLogic.h"
#include "EventManager.h"
#include "Event.h"
#include "GameView.h"
#include "Input.h"
#include "D3DGraphicsFactory.h"
#include "Cache.h"
#include "GraphicsSettings.h"
#include "ScreenElement.h"
#include "ResourceLoader.h"
#include "VRamManager.h"
#include "ActorFactory.h"
#include "Components.h"
#include "Effect.h"
#include "Camera.h"
#include "tbdFont.h"
#include "GuiComponent.h"
#include <fstream>

#define INIT_XML "../Assets/Actors/init.xml"

BOOL APIENTRY DllMain (HINSTANCE hInst, DWORD reason, LPVOID lpReserved)
{
    return TRUE;
}

namespace chimera
{
    class DefaultHumanViewFactory : public IHumanViewFactory
    {
    public:
        IHumanView* VCreateHumanView(VOID)
        {
            return new HumanGameView();
        }
    };

    class DefaultVRamManagerFactory : public IVRamManagerFactory
    {
    public:
        IVRamManager* VCreateVRamManager(VOID)
        {
            IVRamManager* vrm = new VRamManager(1024);
            vrm->VRegisterHandleCreator("obj", new GeometryCreator());
            vrm->VRegisterHandleCreator("jpg", new TextureCreator());
            vrm->VRegisterHandleCreator("png", new TextureCreator());
            return vrm;
        }
    };

    class DefaultInputFactroy : public IInputFactory
    {
    public:
        IInputHandler* VCreateInputHanlder(VOID)
        {
            return new DefaultWinInputHandler();
        }
    };

    class DefaultEffectFactoryFactory : public IEffectFactoryFactory
    {
    public:
        IEffectFactory* VCreateEffectFactroy(VOID)
        {
            return new EffectFactroy();
        }
    };

    class DefaultResourceCacheFactory : public IResourceFactory
    {
    public:
        IResourceCache* VCreateCache(VOID)
        {
            IResourceCache* cache = new ResourceCache();
            cache->VInit(1024, new ResourceFolder("../Assets/"));
            IConfig* cfg = CmGetApp()->VGetConfig();
            cache->VRegisterLoader(std::unique_ptr<IResourceLoader>(new ImageLoader("png", cfg->VGetString("sTexturePath"))));
            cache->VRegisterLoader(std::unique_ptr<IResourceLoader>(new ImageLoader("jpg", cfg->VGetString("sTexturePath"))));
            cache->VRegisterLoader(std::unique_ptr<IResourceLoader>(new ObjLoader(cfg->VGetString("sMeshPath"))));
            cache->VRegisterLoader(std::unique_ptr<IResourceLoader>(new MaterialLoader(cfg->VGetString("sMaterialPath"))));
            cache->VRegisterLoader(std::unique_ptr<IResourceLoader>(new WaveLoader(cfg->VGetString("sSoundPath"))));
            return cache;
        }
    };

    class DefaultEventFactroy : public IEventFactory
    {
    public:
        IEventManager* VCreateEventManager(VOID)
        {
            return new EventManager();
        }
    };

    class DefaultLogicFactory : public ILogicFactory
    {
    public:
        ILogic* VCreateLogic(VOID)
        {
            return new BaseGameLogic();
        }
    };

    class DefaultActorFactory : public IActorFactoryFactory
    {
    public:
        IActorFactory* VCreateActorFactroy(VOID)
        {
            return new ActorFactory();
        }
    };

    class DefaultFontFactroy : public IFontFactory
    {
    public:
        IFont* VCreateFont(VOID)
        {
            return new BMFont();
        }
        IFontRenderer* VCreateFontRenderer(VOID)
        {
            return new FontRenderer();
        }

        IFontManager* VCreateFontManager(VOID)
        {
            return new FontManager();
        }
    };

    class DefaultGuiFactory : public IGuiFactory
    {
    public:
        IGuiRectangle* VCreateRectangle(VOID) { return new GuiRectangle(); }

        IGuiTextureComponent* VCreateTextureComponent(VOID) { return new GuiTextComponent(); }

        IGuiTextComponent* VCreateTextComponent(VOID) { return new GuiTextComponent(); }

        IGuiTextInputComponent* VCreateTextInputComponent(VOID) { return new GuiTextInputComponent(); }

        IGuiLookAndFeel* VCreateLookAndFeel(VOID) { return new GuiLookAndFeel(); }

        IGui* VCreateGui(VOID) { return new Gui(); }
    };

    //factories end

    CM_APP_DESCRIPTION* pDescription = NULL;
    CONST CM_APP_DESCRIPTION* CM_API CmGetDescription(VOID)
    {
        return pDescription;
    }

    VOID CopyDescription(CM_APP_DESCRIPTION* desc)
    {
        pDescription = new CM_APP_DESCRIPTION();
        pDescription->cachePath = desc->cachePath;
        pDescription->logFile = desc->logFile;
        pDescription->facts = desc->facts;
        pDescription->ival = desc->ival;
        pDescription->hInstance = desc->hInstance;
        pDescription->titel = desc->titel;
    }

    IApplication* CmGetApp(VOID)
    {
        return g_pApp;
    }

    ErrorCode CmGetError(VOID)
    {
        return APIGetLastError();
    }

    IApplication* CmCreateApplication(CM_APP_DESCRIPTION* desc)
    {
        if(g_pApp)
        {
            return g_pApp;
        }

        if(!desc || !desc->hInstance)
        {
            APISetError(eErrorCode_InvalidValue);
            return NULL;
        }

        CopyDescription(desc);

        FactoryPtr* ff = desc->facts;

        DefaultHumanViewFactory viewFactory;
        DefaultInputFactroy inputFactory;
        d3d::D3DGraphicsFactory d3dGfxFactory;
        DefaultLogicFactory logicFactory;
        DefaultResourceCacheFactory resFactroy;
        DefaultEventFactroy efactroys;
        DefaultVRamManagerFactory vramFactory;
        DefaultActorFactory af;
        DefaultEffectFactoryFactory eff;
        DefaultFontFactroy fontFact;
        DefaultGuiFactory guifacto;

        FactoryPtr defaultFacts[] = 
        {
            CM_FACTORY_GFX,   (FactoryPtr)&d3dGfxFactory, sizeof(d3d::D3DGraphicsFactory),
            CM_FACTORY_VIEW,  (FactoryPtr)&viewFactory, sizeof(DefaultHumanViewFactory),
            CM_FACTORY_INPUT, (FactoryPtr)&inputFactory, sizeof(DefaultInputFactroy),
            CM_FACTORY_EVENT, (FactoryPtr)&efactroys, sizeof(DefaultEventFactroy),
            CM_FACTORY_LOGIC, (FactoryPtr)&logicFactory, sizeof(DefaultLogicFactory),
            CM_FACTROY_CACHE, (FactoryPtr)&resFactroy, sizeof(DefaultResourceCacheFactory),
            CM_FACTORY_VRAM,  (FactoryPtr)&vramFactory, sizeof(DefaultVRamManagerFactory),
            CM_FACTORY_ACTOR, (FactoryPtr)&af, sizeof(DefaultActorFactory),
            CM_FACTORY_EFFECT, (FactoryPtr)&eff, sizeof(DefaultEffectFactoryFactory),
            CM_FACTORY_FONT, (FactoryPtr)&fontFact, sizeof(DefaultFontFactroy),
            CM_FACTORY_GUI, (FactoryPtr)&guifacto, sizeof(DefaultGuiFactory),
            CM_FACTORY_END
        };

        if(!desc->facts)
        {
            ff = defaultFacts;
        }

        Application* app = new Application();

        Logger::Init(std::string(desc->logFile));

        if(desc->cachePath == std::string(""))
        {
            LOG_WARNING("Cachepath not set!");
        }

        if(desc->logFile == std::string(""))
        {
            desc->logFile = "log.log";
        }
        
        app->g_hInstance = desc->hInstance;

        if(!app->VInitialise(ff))
        {
            APISetError(eErrorCode_InvalidValue);
            return NULL;
        }

        if(!app->VGetHumanView()->VInitialise(ff))
        {
            APISetError(eErrorCode_InvalidValue);
            return NULL;
        }

        int w = app->VGetConfig()->VGetInteger("iWidth");
        int h = app->VGetConfig()->VGetInteger("iHeight");

        if(!app->VGetHumanView()->VGetRenderer()->VCreate(chimera::WndProc, desc->hInstance, desc->titel.c_str(), w, h))
        {
            APISetError(eErrorCode_InvalidValue);
            return NULL;
        }

        if(!app->VGetHumanView()->VOnRestore())
        {
            APISetError(eErrorCode_InvalidValue);
            return NULL;
        }

        //load init scene
        std::ifstream initXml(INIT_XML);
        if(initXml)
        {
            std::string line;
            while(initXml)
            {
                std::getline(initXml, line);
                app->VGetLogic()->VCreateActor(line.c_str(), 0);
            }
        }

        //defualt gfx settings
        std::unique_ptr<IGraphicsSettings> settings(new DefaultGraphicsSettings());
        IRenderScreen* screen = new RenderScreen(std::move(settings));
        screen->VSetName("main");
        app->VGetHumanView()->VAddScene(std::move(std::unique_ptr<IRenderScreen>(screen)));

        //console
        if(desc->args.find("-console") != std::string::npos)
        {
            IScreenElement* rect = new GuiConsole();
            rect->VSetName(VIEW_CONSOLE_NAME);
            CMDimension dim;
            dim.x = 0; dim.x = 0; dim.w = CmGetApp()->VGetWindowWidth(); dim.h = (UINT)(CmGetApp()->VGetWindowHeight() * 0.45f);
            rect->VSetDimension(dim);
            rect->VSetActive(FALSE);
            app->VGetHumanView()->VAddScreenElement(std::unique_ptr<IScreenElement>(rect));
        }

        CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VLoadCommands("controls.ini");

        if(!app->VGetInputHandler()->VInit(desc->hInstance, app->VGetHumanView()->VGetRenderer()->VGetWindowHandle(), app->VGetWindowWidth(), app->VGetWindowHeight()))
        {
            APISetError(eErrorCode_InvalidValue);
        }

        BOOL fullscreen = app->VGetConfig()->VGetBool("bFullscreen");

        if(fullscreen)
        {
            //app->VGetHumanView()->
        }
        app->VGetInputHandler()->VSetCurserOffsets(fullscreen ? 0 : 8 , fullscreen ? 0 : 30); //todo, get systemmetrics

        return app;
    }

    VOID CmLog(CONST std::string& tag, CONST std::string& message, CONST CHAR* funcName, CONST CHAR* file, CONST UINT line) 
    {
        Logger::s_pLogMgr->Log(tag, message, funcName, file, line);
    }

    VOID CmCriticalError(CONST std::string& tag, CONST std::string& message, CONST CHAR* funcName, CONST CHAR* file, CONST UINT line)
    {
        Logger::s_pLogMgr->CriticalError(tag, message, funcName, file, line);
    }

    VOID CmReleaseApplication(VOID)
    {
        SAFE_DELETE(pDescription);
        g_pApp->VClose();
        SAFE_DELETE(g_pApp);
        Logger::Destroy();
    }
}