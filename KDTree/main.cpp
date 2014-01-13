#pragma warning(disable: 4244)

#include <ChimeraAPI.h>
#include "../Source/chimera/Components.h"
#include "../Source/chimera/Event.h"
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
#include "../../Nutty/Nutty/ForEach.h"
#include "../../Nutty/Nutty/Wrap.h"
#include "../../Nutty/Nutty/cuda/cuda_helper.h"

#include "rtracer/tracer.cuh"

#include "cuTimer.cuh"

#include <fstream>
const char* c_home = "runproc nvcc -I\"E:\\Dropbox\\VisualStudio\\Chimera\\Include\" -I\"E:\\Dropbox\\VisualStudio\\Chimera\\Source\\chimera\\api\" -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.5\\include\" -ptx ./rtracer/tracer_kernel.cu -o ./ptx/tracer_kernel.ptx ";//-o .\ptx\tracer_kernel.ptx \"E:\Dropbox\VisualStudio\Chimera\KDTree\rtracer\tracer_kernel.cu";
const char* c_uni = "runproc nvcc -I../Include -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.5\\include\" -ptx ./rtracer/tracer_kernel.cu -o ./ptx/tracer_kernel.ptx ";//-o .\ptx\tracer_kernel.ptx \"E:\Dropbox\VisualStudio\Chimera\KDTree\rtracer\tracer_kernel.cu";
const char* c = c_home;

IKDTree* g_tree;
IRTracer* g_tracer;

extern "C" IKDTree* generate(void* data, uint n, uint depth, uint maxDepth, PrimType type, float radius);
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
float g_scale = 5; //30;
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
    nutty::DeviceBuffer<float> transformRadius;
    nutty::DeviceBuffer<float3> orignalCopy;
    nutty::cuModule* m;
    float time;
    float m_scale;
    BOOL m_enable;

public:
    AnimationProc(nutty::DeviceBuffer<float3>* geo, uint elems,  BOOL enable, nutty::MappedBufferPtr<float>* gfxGeo = NULL) : time(0), _gfxGeo(gfxGeo), m_enable(enable)
    {
        ptr = geo;
        m = new nutty::cuModule("./ptx/animation_N_stuff.ptx");
        k = new nutty::cuKernel(m->GetFunction("animateGeometry"));
        k->SetDimension(nutty::cuda::GetCudaGrid(elems, 256U), 256);
        transformRadius.Resize(geo->Size());
        orignalCopy.Resize(geo->Size());

        nutty::Copy(orignalCopy.Begin(), geo->Begin(), geo->Size());
        nutty::Fill(transformRadius.Begin(), transformRadius.End(), nutty::unary::RandNorm<float>(30));
        g_timer.Start();
        g_timer.Stop();
        m_scale = g_scale;
    }

    std::stringstream _ss;

    VOID AnimateGeo(ULONG deltaMillis)
    {
        time += g_timeScale * deltaMillis;
        k->SetKernelArg(0, orignalCopy);
        k->SetKernelArg(1, *ptr);
        k->SetKernelArg(2, transformRadius);
        k->SetKernelArg(3, time);
        k->SetKernelArg(4, g_scale);
        k->SetKernelArg(5, g_parts);
        k->SetKernelArg(6, elems);
        k->Call(g_tree->GetDefaultStream().GetPointer());
    }

    VOID Toggle(VOID)
    {
        m_enable = !m_enable;
    }

    VOID Reset(VOID)
    {
        nutty::Copy(ptr->Begin(), orignalCopy.Begin(), orignalCopy.Size());
        time = 0;
    }

    VOID VOnUpdate(ULONG deltaMillis)
    {
        if(m_enable)
        {
            AnimateGeo(deltaMillis);
        }

        if(g_reconstructTree && m_enable)
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

AnimationProc* g_animationProc;

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
        nutty::DeviceBuffer<BBox>* aabbs = g_tree->GetAABBs();
        k->SetKernelArg(0, *aabbs);
        k->SetKernelArg(1, g_tree->GetNodes());
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
        //_ss << "Overall Rendering FPS: " << (float)chimera::CmGetApp()->VGetRenderingTimer()->VGetFPS() << "\n";
        if(g_tracer)
        {
            _ss << "Tracing: " << (g_tracer->GetLastMillis() + g_timer.GetMillis()) << " ms\n";
            _ss << "Rays: " << g_tracer->GetLastRayCount() << " (Shadow: " << g_tracer->GetLastShadowRayCount() << ")\n";
        }
        //_ss << "Construction (d=" << g_tree->GetCurrentDepth() << "): " << g_timer.GetMillis() << " ms\n";
        //_ss << "Prims: " << elems << "\n";
        //_ss << "Player: " << pos.x << ", " << pos.y << ", " << pos.z;
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
    g_parts += d;
    g_parts = CLAMP(g_parts, 1, 512);
    return true;
}

BOOL commandTimeScale(chimera::ICommand& cmd)
{
	FLOAT d = cmd.VGetNextFloat();
	g_timeScale *= d;
	return true;
}

BOOL commandToggleAni(chimera::ICommand& cmd)
{
    g_animationProc->Toggle();
    return true;
}

BOOL commandToggleResetAni(chimera::ICommand& cmd)
{
    g_animationProc->Reset();
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
        INT vi = 0;
        v.Resize(mesh->VGetIndexCount() / 3);
        FLOAT scale = chimera::CmGetApp()->VGetConfig()->VGetFloat("fgScale");
        float radius = chimera::CmGetApp()->VGetConfig()->VGetFloat("fSphereRadius");
 
        UINT index = 0;
        for(UINT i = 0; i < mesh->VGetIndexCount() / 3; ++i)
        {
            UINT i0 = mesh->VGetIndices()[3 * i + 0];
            UINT i1 = mesh->VGetIndices()[3 * i + 1];
            UINT i2 = mesh->VGetIndices()[3 * i + 2];
            float3 a, b, c;
            a.x = meshVertices[8 * i0 + 0];
            a.y = meshVertices[8 * i0 + 1];
            a.z = meshVertices[8 * i0 + 2];

            b.x = meshVertices[8 * i0 + 0];
            b.y = meshVertices[8 * i0 + 1];
            b.z = meshVertices[8 * i0 + 2];

            c.x = meshVertices[8 * i0 + 0];
            c.y = meshVertices[8 * i0 + 1];
            c.z = meshVertices[8 * i0 + 2];

            v[vi++] = scale * (a + b + c) / 3.0f;
        }

        elems = v.Size();
    }
    else
    {
        elems = chimera::CmGetApp()->VGetConfig()->VGetInteger("iPoints");
        FLOAT scale = chimera::CmGetApp()->VGetConfig()->VGetFloat("fScale");

        v.Resize(elems);

        INT vi = 0;
        for(UINT i = 0; i < elems; ++i)
        {
            float3 pos;
            pos.x = -scale + 2 * scale * (float)(rand()/(float)RAND_MAX);
            pos.y = scale * (float)(rand()/(float)RAND_MAX);
            pos.z = -scale + 2 * scale * (float)(rand()/(float)RAND_MAX);
            //DEBUG_OUT_A("%f %f %f\n", pos.x, pos.y, pos.z);
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
        float meshScale = chimera::CmGetApp()->VGetConfig()->VGetFloat("fSphereRadius");
        tcmp->GetTransformation()->SetScale(meshScale, meshScale, meshScale);
        chimera::RenderComponent* rcmp = actorDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
        rcmp->m_resource = "box.obj";

        std::shared_ptr<chimera::IVertexBuffer> geo = std::shared_ptr<chimera::IVertexBuffer>(chimera::CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateVertexBuffer());

        geo->VInitParamater(elems, 3 * sizeof(float), *v.GetRawPointer());
        geo->VCreate();

        rcmp->m_vmemInstances = geo;
        chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(actorDesc));

        ID3D11Buffer* buffer = (ID3D11Buffer*)geo->VGetDevicePtr();
        mappedPtr = new nutty::MappedBufferPtr<float>(std::move(nutty::WrapBuffer<float>(buffer)));
    }

    AnimationProc* ap = new AnimationProc(data, elems, chimera::CmGetApp()->VGetConfig()->VGetBool("bAnimate"), mappedPtr);
    g_animationProc = (AnimationProc*)chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(ap));

    chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::unique_ptr<chimera::IProcess>(new Status()));
    
    g_depth = chimera::CmGetApp()->VGetConfig()->VGetInteger("iTreeDepth");
    g_maxDepth = chimera::CmGetApp()->VGetConfig()->VGetInteger("iTreeMaxDepth");
    g_reconstructTree = chimera::CmGetApp()->VGetConfig()->VGetInteger("bReconstructTree") != 0;

    g_tree = generate((void*)data, elems, g_depth, g_maxDepth, eSphere, chimera::CmGetApp()->VGetConfig()->VGetFloat("fSphereRadius"));
    
    if(chimera::CmGetApp()->VGetConfig()->VGetBool("bProfile"))
    {
        cuTimer timer;
        std::ofstream out("profilingSum.txt");
        for(int i = 0; i < chimera::CmGetApp()->VGetConfig()->VGetInteger("iProfileSteps"); ++i)
        {
            timer.Tick();
            g_tree->Update();
            timer.Tock();
        }
        out << "profiling:\n" << chimera::CmGetApp()->VGetConfig()->VGetString("sMesh").c_str() << "\n" << "d=" << g_depth << "\n";
        out << timer.GetMillis() << ", " << timer.GetAverageMillis();
        exit(0);
        out.close();
    }

    g_tree->Update();

    g_tracer = createTracer(g_tree, chimera::CmGetApp()->VGetConfig()->VGetFloat("fSphereRadius"));
    chimera::IGraphicsSettings* settings = chimera::CmGetApp()->VGetHumanView()->VGetSceneByName("main")->VGetSettings();
    settings->VAddSetting(std::unique_ptr<chimera::IGraphicSetting>(g_tracer), chimera::eGraphicsSetting_Lighting);
    ADD_EVENT_LISTENER_STATIC(&PreRestoreDelegate, CM_EVENT_PRE_RESTORE);

    if(!chimera::CmGetApp()->VGetConfig()->VGetBool("bRayTrace"))
    {
        g_tracer->ToggleEnable();
    }

    if(chimera::CmGetApp()->VGetConfig()->VGetBool("bDrawBBox"))
    {
        createBBoxGeo(g_tree->GetNodesCount());
    }

    g_textInfo = chimera::CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextComponent();
    chimera::CMDimension dim;
    dim.x = 5;
    dim.y = 10;
    dim.w = 220;
    dim.h = 35;
    g_textInfo->VSetDimension(dim);
    g_textInfo->VSetName("nodes_content");
    g_textInfo->VAppendText("");
    g_textInfo->VSetAlpha(0.85f);
    g_textInfo->VSetBackgroundColor(0.25f, 0.25f, 0.25f);
    g_textInfo->VSetTextColor(chimera::Color(0,1,0,0));
    chimera::CmGetApp()->VGetHumanView()->VAddScreenElement(std::unique_ptr<chimera::IGuiComponent>(g_textInfo));

    chimera::CommandHandler h = commandScale;
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("crapscale", h);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind u down crapscale -1");
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind i down crapscale +1");

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

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("toogleAni", commandToggleAni);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind b toogleAni");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("toogleResetAni", commandToggleResetAni);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind v toogleResetAni");

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