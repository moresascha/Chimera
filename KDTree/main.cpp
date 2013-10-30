#pragma warning(disable: 4244)

#include <ChimeraAPI.h>
#include <../Source/chimera/Components.h>
#include "../Source/chimera/util.h"
#include "../Source/chimera/Timer.h"
#include "../Source/chimera/Logger.h"
#include "../Source/chimera/Components.h"
#include "../Source/chimera/AxisAlignedBB.h"
#include <fstream>


extern "C" void generate();
extern "C" void init();
extern "C" void release();

#ifdef _DEBUG
#pragma comment(lib, "Chimerax64Debug.lib")
#else
#pragma comment(lib, "Chimerax64Release.lib")
#endif

VOID startChimera(HINSTANCE hInstance)
{
    chimera::CM_APP_DESCRIPTION desc;
    desc.facts = NULL;
    desc.hInstance = hInstance;
    desc.titel = L"KD-TREE";
    desc.ival = 60;
    desc.cachePath = "../Assets/";
    desc.logFile = "log.log";

    chimera::CmCreateApplication(&desc);

    std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
    chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
    tcmp->GetTransformation()->Translate(chimera::util::Vec3(0, 1, 10));
    chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
    rcmp->m_resource = "box.obj";
    rcmp->m_drawType = "wire";
    chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

    chimera::CmGetApp()->VRun();

    chimera::CmReleaseApplication();
}

int APIENTRY _tWinMain(HINSTANCE hInstance,
                       HINSTANCE hPrevInstance,
                       LPTSTR    lpCmdLine,
                       int       nCmdShow)
{
    
#ifdef _DEBUG
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    init();

    generate();

    startChimera(hInstance);

    release();

    return 0;
}