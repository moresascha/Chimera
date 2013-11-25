#pragma warning(disable: 4244)

#include <ChimeraAPI.h>
#include <../Source/chimera/Components.h>
#include <../Source/chimera/Event.h>
#include "../Source/chimera/util.h"
#include "../Source/chimera/Timer.h"
#include "../Source/chimera/Logger.h"
#include "../Source/chimera/Components.h"
#include "../Source/chimera/AxisAlignedBB.h"
#include "../Source/chimera/Timer.h"

#include "kdtree.cuh"
#include "../../Nutty/Nutty/cuda/Module.h"
#include "../../Nutty/Nutty/cuda/Kernel.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"
#include "../../Nutty/Nutty/HostBuffer.h"
#include "../../Nutty/Nutty/Fill.h"
#include "../../Nutty/Nutty/Wrap.h"
#include "../../Nutty/Nutty/cuda/cuda_helper.h"

#include "WaitForActorCreated.h"

#include <fstream>


IKDTree* g_tree;

extern "C" IKDTree* generate(nutty::MappedPtr<float>* data, uint n, uint depth, uint maxDepth);
extern "C" void init();
extern "C" void release();

#ifdef _DEBUG
#pragma comment(lib, "Chimerax64Debug.lib")
#else
#pragma comment(lib, "Chimerax64Release.lib")
#endif

#ifdef _DEBUG
#pragma comment(lib, "Nuttyx64Debug.lib")
#else
#pragma comment(lib, "Nuttyx64Release.lib")
#endif

struct ID3D11Buffer;

UINT elems = 64;
float g_scale = 10;//30;
uint g_depth = 10;
uint g_maxDepth = g_depth;
uint g_parts = 1;
float g_timeScale = 1e-5f;

chimera::IGuiTextComponent* g_textInfo;

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

class AnimationProc : public chimera::IProcess
{
private:
    nutty::cuKernel* k;
    nutty::MappedPtr<float>* mappedPtr;
    nutty::cuModule* m;
    float time;
    std::string cc;
    chimera::util::HTimer m_timer;

public:
    AnimationProc(nutty::MappedPtr<float>* geo, uint lineCount) :  time(0)
    {
        mappedPtr = geo;
        m = new nutty::cuModule("./ptx/KD.ptx");
        k = new nutty::cuKernel(m->GetFunction("animateGeometry"));
        k->SetDimension(nutty::cuda::getCudaGrid(lineCount, 256U), 256);        
    }

    std::stringstream _ss;

    VOID AnimateGeo(ULONG deltaMillis)
    {
        time += g_timeScale * deltaMillis;
        nutty::DevicePtr<float> devptr = mappedPtr->Bind();
        k->SetKernelArg(0, devptr);
        k->SetKernelArg(1, time);
        k->SetKernelArg(2, g_scale);
        k->SetKernelArg(3, g_parts);
        k->SetKernelArg(4, elems);
        k->Call();
        mappedPtr->Unbind();
    }

    VOID VOnUpdate(ULONG deltaMillis)
    {
       // AnimateGeo(deltaMillis);
// 
        m_timer.Start();
        g_tree->Update();
        m_timer.Stop();

        g_tree->GetContentCountStr(cc);

        _ss.str("");
        _ss << "Rendering: " << chimera::CmGetApp()->VGetRenderingTimer()->VGetFPS() << "\n";
        _ss << "Construction (" << g_tree->GetCurrentDepth() << "): " << 1000.0f/m_timer.GetMillis() << "\n";
        _ss << cc;

        g_textInfo->VClearText();
        g_textInfo->VAppendText(_ss.str().c_str());
    }

    ~AnimationProc(VOID)
    {
        SAFE_DELETE(mappedPtr);
        SAFE_DELETE(k);
        SAFE_DELETE(m);
    }
};

class UpdateBBoxes : public chimera::IProcess
{
private:
    nutty::cuKernel* k;
    nutty::MappedPtr<float>* mappedPtr;
    nutty::cuModule* m;
    uint c;

public:
    UpdateBBoxes(nutty::MappedPtr<float>* geo, uint count)
    {
        mappedPtr = geo;
        m = new nutty::cuModule("./ptx/KD.ptx");
        k = new nutty::cuKernel(m->GetFunction("createBBox"));
        uint grid = 1;
        uint block = count;
        if(count > 32)
        {
            grid = nutty::cuda::getCudaGrid(count, 32U);
            block = 32;
        }
        k->SetDimension(grid, block);        
        c = count;
    }

    VOID VOnUpdate(ULONG deltaMillis)
    {
        nutty::DevicePtr<float> ptr = mappedPtr->Bind();
        nutty::DeviceBuffer<float3>* aabbs = (nutty::DeviceBuffer<float3>*)g_tree->GetBuffer(eAxisAlignedBB);
        k->SetKernelArg(0, *aabbs);
        k->SetKernelArg(1, *((nutty::DeviceBuffer<uint>*)g_tree->GetBuffer(eNodesContentCount)));
        k->SetKernelArg(2, ptr);
        k->SetKernelArg(3, c);
        k->SetKernelArg(4, g_depth);
        k->Call();
        mappedPtr->Unbind();
    }

    ~UpdateBBoxes(VOID)
    {
        SAFE_DELETE(mappedPtr);
        SAFE_DELETE(k);
        SAFE_DELETE(m);
    }
};

BOOL commandScale(chimera::ICommand& cmd)
{
    float d = cmd.VGetNextFloat();
    g_scale += d;
    return true;
}

BOOL commandModParts(chimera::ICommand& cmd)
{
    INT d = cmd.VGetNextInt();
    g_parts += (UINT)d;
    g_parts = CLAMP(g_parts, 1, 16);
    return true;
}

BOOL commandTimeScale(chimera::ICommand& cmd)
{
	FLOAT d = cmd.VGetNextFloat();
	g_timeScale *= d;
	return true;
}

BOOL commandIncDecDepth(chimera::ICommand& cmd)
{
    INT d = cmd.VGetNextInt();
    g_depth += d;
    g_tree->SetDepth(g_depth);
    g_depth = g_tree->GetCurrentDepth();
    return true;
}

void createBBoxGeo(uint aabbsCount)
{
    std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
    chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
    chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
    std::unique_ptr<chimera::IGeometry> geo = chimera::CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry();
    
    uint stride = 8;
    uint lines = (12 * aabbsCount);
    uint vertices = 2 * lines;
    FLOAT* v = new FLOAT[vertices * stride];

    for(UINT i = 0; i < vertices; ++i)
    {
        v[stride * i + 0] = i;
        v[stride * i + 1] = i%2;
        v[stride * i + 2] = 0;

        v[stride * i + 3] = 0;
        v[stride * i + 4] = 1;
        v[stride * i + 5] = 0;

        v[stride * i + 6] = 0;
        v[stride * i + 7] = 0;
    }

    geo->VSetTopology(chimera::eTopo_Lines);
    geo->VSetVertexBuffer(v, vertices, stride * sizeof(float));
    geo->VCreate();
    SAFE_DELETE(v);
   
     rcmp->m_geo = std::move(geo);
 
    ID3D11Buffer* buffer = (ID3D11Buffer*)rcmp->m_geo->VGetVertexBuffer()->VGetDevicePtr();
    nutty::MappedPtr<float>* mappedPtr = new nutty::MappedPtr<float>(std::move(nutty::Wrap<float>(buffer)));

    chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

    chimera::IProcess* proc = chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(new UpdateBBoxes(mappedPtr, aabbsCount)));
}

void createWorld(void)
{
    std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
    chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
    float meshScale = 0.1f;
    tcmp->GetTransformation()->SetScale(meshScale, meshScale, meshScale);
    chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
    rcmp->m_resource = "box.obj";
    std::shared_ptr<chimera::IVertexBuffer> geo = std::shared_ptr<chimera::IVertexBuffer>(chimera::CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateVertexBuffer());

    std::shared_ptr<chimera::IMesh> monkey = std::static_pointer_cast<chimera::IMesh>(chimera::CmGetApp()->VGetCache()->VGetHandle(chimera::CMResource("bunny.obj")));
    float scale = 10;
    CONST FLOAT* monkeyVertes = monkey->VGetVertices();
    elems = monkey->VGetVertexCount();
    FLOAT* v = new FLOAT[elems * 3];
    for(UINT i = 0; i < elems; ++i)
    {
        v[3*i+0] = scale *monkeyVertes[8 * i + 0];
        v[3*i+1] = scale + scale * monkeyVertes[8 * i + 1];
        v[3*i+2] = -scale * monkeyVertes[8 * i + 2];
    }
    
    geo->VInitParamater(elems, 3 * sizeof(float), v);
    geo->VCreate();
    SAFE_DELETE(v);

    ID3D11Buffer* buffer = (ID3D11Buffer*)geo->VGetDevicePtr();
    nutty::MappedPtr<float>* mappedPtr = new nutty::MappedPtr<float>(std::move(nutty::Wrap<float>(buffer)));
    AnimationProc* ap = new AnimationProc(mappedPtr, elems);
    //ap->AnimateGeo(123123);
    chimera::IProcess* proc = chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(ap));
    
    rcmp->m_vmemInstances = geo;
    chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

    chimera::CommandHandler h = commandScale;
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("crapscale", h);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind u down crapscale -0.1");
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind i down crapscale +0.1");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("crapparts", commandModParts);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind l crapparts -1");
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind k crapparts +1");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("craptime", commandTimeScale);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind t craptime 2");
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind z craptime 0.5");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("incDescdepth", commandIncDecDepth);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind + incDescdepth 1");
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind - incDescdepth -1");

    uint leafBBoxCount = (1 << (uint)(g_maxDepth-1));
    nutty::HostBuffer<float3> data(elems);

    g_tree = generate(mappedPtr, elems, g_depth, g_maxDepth);

    createBBoxGeo(leafBBoxCount);

    g_textInfo = chimera::CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextComponent();
    chimera::CMDimension dim;
    dim.x = 5;
    dim.y = 10;
    dim.w = chimera::CmGetApp()->VGetWindowWidth();
    dim.h = 60;
    g_textInfo->VSetDimension(dim);
    g_textInfo->VSetName("nodes_content");
    g_textInfo->VAppendText("");
    g_textInfo->VSetAlpha(0);
    g_textInfo->VSetTextColor(chimera::Color(0,1,0,0));
    chimera::CmGetApp()->VGetHumanView()->VAddScreenElement(std::unique_ptr<chimera::IGuiComponent>(g_textInfo));
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
	
    startChimera(hInstance);

    nutty::Init((ID3D11Device*)chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetDevice());
   
    createWorld();
    
    chimera::CmGetApp()->VRun();

    chimera::CmReleaseApplication();

    release();

    return 0;
}