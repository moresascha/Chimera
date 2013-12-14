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

#include "rtracer/tracer.cuh"

#include <fstream>
const char* c_home = "runproc nvcc -I\"E:\\Dropbox\\VisualStudio\\Chimera\\Include\" -I\"E:\\Dropbox\\VisualStudio\\Chimera\\Source\\chimera\\api\" -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.5\\include\" -ptx ./rtracer/tracer_kernel.cu -o ./ptx/tracer_kernel.ptx ";//-o .\ptx\tracer_kernel.ptx \"E:\Dropbox\VisualStudio\Chimera\KDTree\rtracer\tracer_kernel.cu";
const char* c_uni = "runproc nvcc -I../Include -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.5\\include\" -ptx ./rtracer/tracer_kernel.cu -o ./ptx/tracer_kernel.ptx ";//-o .\ptx\tracer_kernel.ptx \"E:\Dropbox\VisualStudio\Chimera\KDTree\rtracer\tracer_kernel.cu";
const char* c = c_home;

IKDTree* g_tree;
IRTracer* g_tracer;

extern "C" IKDTree* generate(void* data, uint n, uint depth, uint maxDepth);
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

UINT elems = 8;
float g_scale = 50; //30;
bool g_reconstructTree = true;
uint g_depth = 4;
uint g_maxDepth = 12;
uint g_parts = 4;
float g_timeScale = 1e-4f;
chimera::util::HTimer g_timer;

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
    nutty::DeviceBuffer<float3>* ptr;
    nutty::MappedBufferPtr<float>* _gfxGeo;
    nutty::cuModule* m;
    float time;
    float m_scale;

public:
    AnimationProc(nutty::DeviceBuffer<float3>* geo, uint elems, nutty::MappedBufferPtr<float>* gfxGeo = NULL) : time(0), _gfxGeo(gfxGeo)
    {
        ptr = geo;
        m = new nutty::cuModule("./ptx/animation_N_stuff.ptx");
        k = new nutty::cuKernel(m->GetFunction("animateGeometry"));
        k->SetDimension(nutty::cuda::GetCudaGrid(elems, 256U), 256);
        m_scale = g_scale;
    }

    std::stringstream _ss;

    VOID AnimateGeo(ULONG deltaMillis)
    {
        time += g_timeScale * deltaMillis;
        m_scale = g_scale*(UINT)(log(elems));
        k->SetKernelArg(0, *ptr);
        k->SetKernelArg(1, time);
        k->SetKernelArg(2, g_scale);
        k->SetKernelArg(3, g_parts);
        k->SetKernelArg(4, elems);
        k->Call(g_tree->GetDefaultStream().GetPointer());

        if(_gfxGeo)
        {
            nutty::DevicePtr<float> devptr = _gfxGeo->Bind();
            k->SetKernelArg(0, devptr);
            k->Call(g_tree->GetDefaultStream().GetPointer());
            _gfxGeo->Unbind();
        }
    }

    VOID VOnUpdate(ULONG deltaMillis)
    {
        if(chimera::CmGetApp()->VGetConfig()->VGetBool("bAnimate"))
        {
            AnimateGeo(deltaMillis);
        }

        if(g_reconstructTree)
        {
            cudaDeviceSynchronize();
            g_timer.Start();
            g_tree->Update();
            cudaDeviceSynchronize();
            g_timer.Stop();
        }
    }

    ~AnimationProc(VOID)
    {
        SAFE_DELETE(k);
        SAFE_DELETE(m);
        SAFE_DELETE(_gfxGeo);
    }
};

class UpdateBBoxes : public chimera::IProcess
{
private:
    nutty::cuKernel* k;
    nutty::MappedBufferPtr<float>* mappedPtr;
    nutty::cuModule* m;
    uint c;

public:
    UpdateBBoxes(nutty::MappedBufferPtr<float>* geo, uint count)
    {
        mappedPtr = geo;
        m = new nutty::cuModule("./ptx/animation_N_stuff.ptx");
        k = new nutty::cuKernel(m->GetFunction("createBBox"));
        uint grid = 1;
        uint block = count;
        if(count > 32)
        {
            grid = nutty::cuda::GetCudaGrid(count, 32U);
            block = 32;
        }
        k->SetDimension(grid, block);
        c = count;
    }

    VOID VOnUpdate(ULONG deltaMillis)
    {
        chimera::CmGetApp()->VGetRenderer()->VPresent();
        nutty::DevicePtr<float> ptr = mappedPtr->Bind(g_tree->GetDefaultStream().GetPointer());
        nutty::DeviceBuffer<AABB>* aabbs = g_tree->GetAABBs();
        k->SetKernelArg(0, *aabbs);
        k->SetKernelArg(1, *((nutty::DeviceBuffer<Node>*)g_tree->GetNodes()));
        k->SetKernelArg(2, ptr);
        k->SetKernelArg(3, c);
        k->SetKernelArg(4, g_depth);
        k->Call(g_tree->GetDefaultStream().GetPointer());
        mappedPtr->Unbind(g_tree->GetDefaultStream().GetPointer());
    }

    ~UpdateBBoxes(VOID)
    {
        SAFE_DELETE(mappedPtr);
        SAFE_DELETE(k);
        SAFE_DELETE(m);
    }
};

class Status : public chimera::IProcess
{
private:
    std::string cc;
    std::stringstream _ss;
public:
    VOID VOnUpdate(ULONG deltaMillis)
    {
        g_tree->GetContentCountStr(cc);
        chimera::IActor* player = chimera::CmGetApp()->VGetHumanView()->VGetTarget();
        chimera::util::Vec3 pos = chimera::GetActorCompnent<chimera::TransformComponent>(player, CM_CMP_TRANSFORM)->GetTransformation()->GetTranslation();
        _ss.str("");
        _ss << "Overall Rendering FPS: " << (float)chimera::CmGetApp()->VGetRenderingTimer()->VGetFPS() << "\n";
        if(g_tracer)
        {
            _ss << "Tracing: " << g_tracer->GetLastMillis() << " ms\n";
            _ss << "Rays: " << g_tracer->GetLastRayCount() << " (Shadow: " << g_tracer->GetLastShadowRayCount() << ")\n";
        }
        _ss << "Construction (d=" << g_tree->GetCurrentDepth() << "): " << g_timer.GetMillis() << " ms\n";
        _ss << "Prims: " << elems << "\n";
        _ss << "Player: " << pos.x << ", " << pos.y << ", " << pos.z;
        g_textInfo->VClearText();
        g_textInfo->VAppendText(_ss.str());
    }
};

void PreRestoreDelegate(chimera::IEventPtr event)
{
    if(g_tracer)
    {
        g_tracer->ReleaseSharedResources();
    }
}

BOOL commandScale(chimera::ICommand& cmd)
{
    float d = cmd.VGetNextFloat();
    g_scale += d;
    return true;
}

BOOL commandCompile(chimera::ICommand& cmd)
{
    
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(c);
    if(g_tracer)
    {
        g_tracer->Compile();
    }
    return true;
}

BOOL commandModParts(chimera::ICommand& cmd)
{
    INT d = cmd.VGetNextInt();
    g_parts += (UINT)16*d;
    g_parts = CLAMP(g_parts, 1, 512);
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

BOOL commandToggleRaytracer(chimera::ICommand& cmd)
{
    g_tracer->ToggleEnable();
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
    nutty::MappedBufferPtr<float>* mappedPtr = new nutty::MappedBufferPtr<float>(std::move(nutty::WrapBuffer<float>(buffer)));

    chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

    chimera::IProcess* proc = chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(new UpdateBBoxes(mappedPtr, aabbsCount)));
}

void createWorld(void)
{
    nutty::HostBuffer<float3> v;
    std::string m = chimera::CmGetApp()->VGetConfig()->VGetString("sMesh");
    if(m != "-")
    {
        std::shared_ptr<chimera::IMesh>& mesh = std::static_pointer_cast<chimera::IMesh>(chimera::CmGetApp()->VGetCache()->VGetHandle(chimera::CMResource(m)));
        CONST FLOAT* meshVertices = mesh->VGetVertices();

        elems = mesh->VGetVertexCount();
        UINT scale = 50*(UINT)(log(elems));

        v.Resize(elems);

        INT vi = 0;
        for(UINT i = 0; i < elems; ++i)
        {
            float3 pos;
            pos.x = scale * meshVertices[8 * i + 0];
            pos.y = scale * meshVertices[8 * i + 1];
            pos.z = scale * meshVertices[8 * i + 2];

            v[vi++] = pos;
        }
    }
    else
    {
        elems = chimera::CmGetApp()->VGetConfig()->VGetInteger("iPoints");
        UINT scale = 5*(UINT)(log(elems));

        v.Resize(elems);

        INT vi = 0;
        for(UINT i = 0; i < elems; ++i)
        {
            float3 pos;
            pos.x = 1.5f*i;//scale * (float)(rand()/(float)RAND_MAX);
            pos.y = 1;//scale * (float)(rand()/(float)RAND_MAX);
            pos.z = 1.5f * i;//scale * (float)(rand()/(float)RAND_MAX);
            pos.x = scale * (float)(rand()/(float)RAND_MAX);
            pos.y = scale * (float)(rand()/(float)RAND_MAX);
            pos.z = scale * (float)(rand()/(float)RAND_MAX);

            v[vi++] = pos;
        }
    }

    nutty::DeviceBuffer<float3>* data = new nutty::DeviceBuffer<float3>(elems);
    nutty::Copy(data->Begin(), v.Begin(), elems);

    nutty::MappedBufferPtr<float>* mappedPtr = NULL;
    
    if(chimera::CmGetApp()->VGetConfig()->VGetBool("bDrawRasterGeo"))
    {
        std::unique_ptr<chimera::ActorDescription> actorDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
        chimera::TransformComponent* tcmp = actorDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
        float meshScale = 1;
        tcmp->GetTransformation()->SetScale(meshScale, meshScale, meshScale);
        chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
        rcmp->m_resource = "sphere_low.obj";

        std::shared_ptr<chimera::IVertexBuffer> geo = std::shared_ptr<chimera::IVertexBuffer>(chimera::CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateVertexBuffer());

        geo->VInitParamater(elems, 3 * sizeof(float), *v.GetRawPointer());
        geo->VCreate();

        rcmp->m_vmemInstances = geo;
        chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

        ID3D11Buffer* buffer = (ID3D11Buffer*)geo->VGetDevicePtr();
        mappedPtr = new nutty::MappedBufferPtr<float>(std::move(nutty::WrapBuffer<float>(buffer)));
    }

    AnimationProc* ap = new AnimationProc(data, elems, mappedPtr);
    chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(ap));

    chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(new Status()));
    

    g_depth = chimera::CmGetApp()->VGetConfig()->VGetInteger("iTreeDepth");
    g_maxDepth = chimera::CmGetApp()->VGetConfig()->VGetInteger("iTreeMaxDepth");
    uint bboxCount = (1 << (uint)g_maxDepth) - 1;
    g_reconstructTree = chimera::CmGetApp()->VGetConfig()->VGetInteger("bReconstructTree") != 0;

    g_tree = generate((void*)data, elems, g_depth, g_maxDepth);

    g_tracer = createTracer(g_tree);
    chimera::IGraphicsSettings* settings = chimera::CmGetApp()->VGetHumanView()->VGetSceneByName("main")->VGetSettings();
    settings->VAddSetting(std::unique_ptr<chimera::IGraphicSetting>(g_tracer), chimera::eGraphicsSetting_Lighting);
    ADD_EVENT_LISTENER_STATIC(&PreRestoreDelegate, CM_EVENT_PRE_RESTORE);

    if(!chimera::CmGetApp()->VGetConfig()->VGetBool("bRayTrace"))
    {
        g_tracer->ToggleEnable();
    }

    if(chimera::CmGetApp()->VGetConfig()->VGetBool("bDrawBBox"))
    {
        createBBoxGeo(bboxCount);
    }

    g_textInfo = chimera::CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextComponent();
    chimera::CMDimension dim;
    dim.x = 5;
    dim.y = 10;
    dim.w = 200;
    dim.h = 82;
    g_textInfo->VSetDimension(dim);
    g_textInfo->VSetName("nodes_content");
    g_textInfo->VAppendText("");
    g_textInfo->VSetAlpha(0.85f);
    g_textInfo->VSetBackgroundColor(0.25f, 0.25f, 0.25f);
    g_textInfo->VSetTextColor(chimera::Color(0,1,0,0));
    chimera::CmGetApp()->VGetHumanView()->VAddScreenElement(std::unique_ptr<chimera::IGuiComponent>(g_textInfo));

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

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("toogleRT", commandToggleRaytracer);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind n toogleRT");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("cmprtkernel", commandCompile);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind c cmprtkernel");
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
   
    g_treeDebug = chimera::CmGetApp()->VGetConfig()->VGetBool("bDebug") == 1;

    createWorld();
    
    chimera::CmGetApp()->VRun();

    REMOVE_EVENT_LISTENER_STATIC(PreRestoreDelegate, CM_EVENT_PRE_RESTORE);

    chimera::CmReleaseApplication();

    release();

    return 0;
}