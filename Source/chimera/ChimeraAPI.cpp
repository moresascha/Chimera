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

BOOL APIENTRY DllMain (HINSTANCE hInst, DWORD reason, LPVOID lpReserved)
{
    return TRUE;
}

namespace chimera
{
    CM_APP_DESCRIPTION* pDescription = NULL;
    CONST CM_APP_DESCRIPTION* CmGetDescription(VOID)
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
        }

        if(!app->VGetHumanView()->VInitialise(ff))
        {
            APISetError(eErrorCode_InvalidValue);
        }

        if(!app->VGetHumanView()->VGetRenderer()->VCreate(chimera::WndProc, desc->hInstance, desc->titel.c_str(), 800, 600))
        {
            APISetError(eErrorCode_InvalidValue);
        }

        if(!app->VGetHumanView()->VOnRestore())
        {
            APISetError(eErrorCode_InvalidValue);
        }
        
        //for testing

        std::unique_ptr<IGraphicsSettings> settings(new DefaultGraphicsSettings());
        IRenderScreen* screen = new RenderScreen(std::move(settings));
        screen->VSetName("main");
        app->VGetHumanView()->VAddScene(std::move(std::unique_ptr<IRenderScreen>(screen)));

        IScreenElement* normalScreen = new RenderTargetScreen(app->VGetHumanView()->VGetRenderer()->VGetAlbedoBuffer()->VGetRenderTarget(eDiff_NormalsTarget));
		CMDimension dim;
        dim.x = 0;
		dim.y = 0;
		dim.w = 200;
		dim.h = 150;
        normalScreen->VSetDimension(dim);
        normalScreen->VSetName("normals");
		app->VGetHumanView()->VAddScreenElement(std::move(std::unique_ptr<IScreenElement>(normalScreen)));


        std::unique_ptr<ActorDescription> _d = app->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
        CameraComponent* cmp = _d->AddComponent<CameraComponent>(CM_CMP_CAMERA);

		std::shared_ptr<ICamera> cam(new chimera::util::CharacterCamera(CmGetApp()->VGetWindowWidth(), CmGetApp()->VGetWindowHeight(), 1e-2f, 1e3));
		cmp->SetCamera(cam);

		TransformComponent* tcmp = _d->AddComponent<TransformComponent>(CM_CMP_TRANSFORM);
		tcmp->GetTransformation()->SetTranslate(util::Vec3(0,1,-2));
       
        IActor* actor = app->VGetLogic()->VCreateActor(std::move(_d));

		CmGetApp()->VGetHumanView()->VSetTarget(actor);

		CharacterController* cc = new CharacterController();
		std::unique_ptr<ActorController> ac(cc);

		cc->VActivate();

		CmGetApp()->VGetLogic()->VAttachView(std::move(ac), actor->GetId());

		//mesh actor

		_d = app->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
		_d->AddComponent<TransformComponent>(CM_CMP_TRANSFORM);
		RenderComponent* rcmp = _d->AddComponent<RenderComponent>(CM_CMP_RENDERING);
		rcmp->m_resource = "plane.obj";
		PhysicComponent* phxc = _d->AddComponent<PhysicComponent>(CM_CMP_PHX);
		phxc->m_shapeStyle = "plane";
		phxc->m_material = "static";

		app->VGetLogic()->VCreateActor(std::move(_d));


        TBD_FOR_INT(16)
        {
            _d = app->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
            tcmp = _d->AddComponent<TransformComponent>(CM_CMP_TRANSFORM);
            tcmp->GetTransformation()->Translate(util::Vec3(2.5f * (i % 4), 1, 2.5f * 4 * (i / 4 / 4.0f)));
            rcmp = _d->AddComponent<RenderComponent>(CM_CMP_RENDERING);
            rcmp->m_resource = "box.obj";
            app->VGetLogic()->VCreateActor(std::move(_d));
        }
        
        _d = app->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
        _d->AddComponent<TransformComponent>(CM_CMP_TRANSFORM);
        rcmp = _d->AddComponent<RenderComponent>(CM_CMP_RENDERING);
        rcmp->m_resource = "skydome.png";
        rcmp->m_type = "skydome";
        app->VGetLogic()->VCreateActor(std::move(_d));
        
        if(!app->VGetInputHandler()->VInit(desc->hInstance, app->VGetHumanView()->VGetRenderer()->VGetWindowHandle(), app->VGetWindowWidth(), app->VGetWindowHeight()))
        {
            APISetError(eErrorCode_InvalidValue);
        }

        BOOL fullscreen = FALSE;
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