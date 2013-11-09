#pragma warning(disable: 4244)

#include <ChimeraAPI.h>
#include <../Source/chimera/Components.h>
#include "../Source/chimera/util.h"
#include "../Source/chimera/Timer.h"
#include "../Source/chimera/Logger.h"
#include "../Source/chimera/Components.h"
#include "../Source/chimera/AxisAlignedBB.h"
#include "../../Nutty/Nutty/HostBuffer.h"
#include "../../Nutty/Nutty/Fill.h"
#include "kdtree.cuh"
#include <fstream>


extern "C" void generate(nutty::HostBuffer<float3>& data, uint d, nutty::HostBuffer<float3>& aabbs, nutty::HostBuffer<SplitData>& splitData);
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
    desc.args = "-console";

    chimera::CmCreateApplication(&desc);
}

struct createElems
{
    float3 operator()(void)
    {
        float scale = 20;
        float3 f;
        f.x = scale * (-1.0f + 2.0f * rand() / (float)RAND_MAX);
        f.y = 1.0f + scale * 2.0f * rand() / (float)RAND_MAX;
        f.z = scale * (-1.0f + 2.0f * rand() / (float)RAND_MAX);
        return f;
    }
};

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
	
    startChimera(hInstance);
    /*
    chimera::CMResource res("cafe_bar.obj");
    std::shared_ptr<chimera::IMesh> h = std::static_pointer_cast<chimera::IMesh>(chimera::CmGetApp()->VGetCache()->VGetHandle(res));
    const float* v = h->VGetVertices();

    uint elements = h->VGetVertexCount();
    uint depth = 8;
    uint aabbsCount = (1 << (depth-1)) - 1;
    nutty::HostBuffer<float3> data(elements);
    uint index = 0;
    for(int i = 0; i < h->VGetVertexCount(); i +=h->VGetVertexStride())
    {
        float3 _v;
        _v.x = v[i + 0]; _v.y = v[i + 1]; _v.z = v[i + 2];
        data[index++] = _v;
    }
	
    //nutty::Fill(data.Begin(), data.End(), createElems());
    float3 zero = float3();
    nutty::HostBuffer<float3> aabbs(aabbsCount, zero);
    nutty::HostBuffer<SplitData> splitData(aabbsCount);
    generate(data, depth, aabbs, splitData);
	
    //root AABB
    std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
    chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
    chimera::util::Vec3 maxi(aabbs[0].x, aabbs[0].y, aabbs[0].z);
    chimera::util::Vec3 mini(aabbs[1].x, aabbs[1].y, aabbs[1].z);
    chimera::util::Vec3 scale = (maxi - mini)/2.0f;
    chimera::util::Vec3 mid = mini + (maxi - mini) * 0.5;
    tcmp->GetTransformation()->Scale(scale);
    tcmp->GetTransformation()->Translate(chimera::util::Vec3(mid.x, mid.y, mid.z));
    chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
    rcmp->m_resource = "wire_box.obj";
    rcmp->m_drawType = "wire";
    chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

    //split planes
    for(int i = 0; i < splitData.Size(); ++i)
    {
        SplitData sp = splitData[i];

        if(sp.axis == -1 || sp.split < -1000)
        {
            continue;
        }

        DEBUG_OUT_A("%d %f\n", sp.axis, sp.split);

        std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
        chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
        chimera::util::Vec3 maxi(aabbs[0].x, aabbs[0].y, aabbs[0].z);
        chimera::util::Vec3 mini(aabbs[1].x, aabbs[1].y, aabbs[1].z);
        chimera::util::Vec3 scale = (maxi - mini)/2;
        chimera::util::Vec3 mid = mini + (maxi - mini) * 0.5;
        
        if(sp.axis == 0)
        {
            scale.x = 0;
            mid.x = sp.split;
        }
        else if(sp.axis == 1)
        {
            scale.y = 0;
            mid.y = sp.split;
        }
        else
        {
            scale.z = 0;
            mid.z = sp.split;
        }

        tcmp->GetTransformation()->Scale(scale);
        tcmp->GetTransformation()->Translate(chimera::util::Vec3(mid.x, mid.y, mid.z));
        chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
        rcmp->m_resource = "wire_box.obj";
        rcmp->m_drawType = "wire";
        chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));
    }

    for(int i = 0; i < elements; ++i)
    {
        std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
        chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
        float3 pos = data[i];
        //tcmp->GetTransformation()->Scale(0.05f);
        tcmp->GetTransformation()->Translate(chimera::util::Vec3(pos.x, pos.y, pos.z));
        chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
        rcmp->m_resource = "box.obj";
        rcmp->m_drawType = "solid";
        chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));
    }
    */
    chimera::CmGetApp()->VRun();

    chimera::CmReleaseApplication();

    release();

    return 0;
}