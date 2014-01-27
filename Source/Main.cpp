#include "chimera/stdafx.h"
#include "chimera/Components.h"
#include "chimera/util.h"

static struct LeakDetecter
{
    LeakDetecter(void)
    {
//        _CrtSetBreakAlloc(217);
    }
} detecter;

void createSpheres(void)
{
    int c = 4;
    for(int i = 0; i < c; ++i)
    {
        for(int j = 0; j < c; ++j)
        {
            chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VCreateActor("sphere.xml");
            chimera::TransformComponent* cmp;
            actor->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
            cmp->GetTransformation()->SetTranslation(2.0f*i, 5.0f, 2.0f*j);
        }
    }
}

void callback(std::shared_ptr<chimera::IResHandle>& handle)
{
    std::shared_ptr<chimera::IMeshSet> meshes = std::static_pointer_cast<chimera::IMeshSet>(handle);
    for(auto it = meshes->VBegin(); it != meshes->VEnd(); ++it) 
    {
        std::string subMesh = it->first;

        std::unique_ptr<chimera::ActorDescription> desc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();

        desc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);

        chimera::RenderComponent* renderCmp = desc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);

        renderCmp->m_meshId = subMesh;
        renderCmp->m_resource = meshes->VGetResource();

        chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(desc));
    }
}

void createLights(void)
{
    int c = 2;
    chimera::util::cmRNG rng;
    for(int i = 0; i < c; ++i)
    {
        for(int j = 0; j < c; ++j)
        {
            chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VCreateActor("pointlight.xml");
            chimera::TransformComponent* cmp;
            actor->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
            cmp->GetTransformation()->SetTranslation(10*i, 5, 10*j);

            chimera::LightComponent* lightCmp;
            actor->VQueryComponent(CM_CMP_LIGHT, (chimera::IActorComponent**)&lightCmp);
            lightCmp->m_color.Set(rng.NextFloat(), rng.NextFloat(), rng.NextFloat(), 1);
        }
    }
}

void run(HINSTANCE hInstance)
{
    chimera::CM_APP_DESCRIPTION desc;
    ZeroMemory(&desc, sizeof(chimera::CM_APP_DESCRIPTION));
    desc.hInstance = hInstance;

    chimera::CmCreateApplication(&desc);

    //createSpheres();
    //createLights()
    chimera::CmGetApp()->VGetLogic()->VLoadLevel("testlevel.xml");
    //chimera::CmGetApp()->VGetLogic()->VCreateActor("ruin.xml");
    //chimera::CmGetApp()->VGetLogic()->VCreateActor("ton.xml");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind t spawnactor ton.xml");

    //chimera::CmGetApp()->VGetCache()->VGetHandleAsync(chimera::CMResource("twoObjTest.obj"), callback);

    chimera::CmGetApp()->VRun();

    chimera::CmReleaseApplication();
}

int APIENTRY _tWinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPTSTR    lpCmdLine,
                     int       nCmdShow)
{
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    run(hInstance);

    return 0;
}




