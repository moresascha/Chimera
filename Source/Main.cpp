#include "chimera/stdafx.h"
#include <cutil_inline.h>
#ifdef _DEBUG
#include <cutil_math.h>
#endif
#include "chimera/Components.h"
#include "chimera/util.h"
#include "chimera/AxisAlignedBB.h"
#include "chimera/Event.h"
#include "chimera/Process.h"
#include "chimera/Camera.h"
#include "chimera/PhysicsSystem.h"

static struct LeakDetecter
{
    LeakDetecter(void)
    {
        //_CrtSetBreakAlloc(112920);
    }
} detecter;

void PostGameMessage(const char* msg)
{
    chimera::IGuiTextComponent* text = chimera::CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextComponent();
    text->VSetAlpha(0.0f);
    text->VSetBackgroundColor(0,0,0);
    chimera::CMDimension dim;
    dim.x = chimera::CmGetApp()->VGetWindowWidth() / 2 - 35;
    dim.y = 10;//chimera::CmGetApp()->VGetWindowHeight() / 2 - 8 - 100;
    dim.h = 16;
    dim.w = 70;
    //text->VSetAlignment(chimera::eGuiAlignCenter);
    text->VSetDimension(dim);
    text->VAppendText(msg);
    chimera::CmGetApp()->VGetHumanView()->VAddScreenElement(std::unique_ptr<chimera::IScreenElement>(text));
}

const static uint gridDim = 3;
static uint sphereCount = 0;
chimera::ActorId spheres[gridDim][gridDim];
bool killSphere(chimera::ICommand& cmd)
{
    if(sphereCount == gridDim * gridDim || sphereCount == 0)
    {
        return true;
    }
    chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VFindActor("HumanCameraController");
    chimera::CameraComponent* cmp;
    actor->VQueryComponent(CM_CMP_CAMERA, (chimera::IActorComponent**)&cmp);
    chimera::ActorId hitActor = chimera::CmGetApp()->VGetLogic()->VGetPhysics()->VRayCast(&cmp->GetCamera()->GetEyePos(), &cmp->GetCamera()->GetViewDir());
    if(hitActor != CM_INVALID_ACTOR_ID && hitActor > 5)
    {
        for(int r = 0; r < gridDim; ++r)
        {
            for(int c = 0; c < gridDim; ++c)
            {
                if(spheres[r][c] == hitActor)
                {
                    QUEUE_EVENT(new chimera::DeleteActorEvent(hitActor));
                    spheres[r][c] = CM_INVALID_ACTOR_ID;
                    sphereCount--;
                    if(sphereCount == 0)
                    {
                        PostGameMessage("You won!");
                    }
                }
            }
        }
    }
    return true;
}

void spawnSphereProc(uint deltaMillis)
{
    static uint time = 0;
    static chimera::util::cmRNG rng;
    static bool first = true;
    if(sphereCount == 0 && !first || sphereCount == gridDim * gridDim)
    {
        return;
    }
    if(time > 300 || first)
    {
        first = false;
        while(1)
        {
            uint r = rng.NextInt() % gridDim;
            uint c = rng.NextInt() % gridDim;
            if(spheres[r][c] == CM_INVALID_ACTOR_ID)
            {
                chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VCreateActor("sphere.xml");
                spheres[r][c] = actor->GetId();
                chimera::TransformComponent* cmp;
                actor->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
                cmp->GetTransformation()->SetTranslation(3.5f*r, 2, 3.5f*c);
                sphereCount++;
                if(sphereCount == gridDim * gridDim)
                {
                    PostGameMessage("You failed!");
                }
                break;
            }
        }

        time = 0;
    }

//     static uint time = 0;
//     time += deltaMillis;
//     int c = 4;
//     for(int i = 0; i < c; ++i)
//     {
//         for(int j = 0; j < c; ++j)
//         {
//             chimera::IActor* actor = spheres[i * 4 + j];
//             QUEUE_EVENT(new chimera::MoveActorEvent(actor->GetId(), chimera::util::Vec3(3.5f*i, 3 + sin(1e-3f * time), 3.5f*j), false));
//         }
//     }

    time += deltaMillis;
}

void createSpheres(void)
{
    int c = 4;
    for(int i = 0; i < c; ++i)
    {
        for(int j = 0; j < c; ++j)
        {
            chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VCreateActor("sphere.xml");
            //spheres[i * 4 + j] = actor->GetId();
            chimera::TransformComponent* cmp;
            actor->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
            cmp->GetTransformation()->SetTranslation(3.5f*i, 1, 3.5f*j);
        }
    }
}

bool compileCudakernel(chimera::ICommand& cmd)
{
    const char* c_home = "runproc nvcc -I\"C:\\Users\\Sascha\\Dropbox\\VisualStudio\\Chimera\\Include\" -I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v5.5\\include\" -ptx ./chimera/Particles.cu -o ../Assets/kernel/Particles.ptx";
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(c_home);
    return true;
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
    int c = 1;
    chimera::util::cmRNG rng;
    float scale = 5;
    for(int i = 0; i < c; ++i)
    {
        for(int j = 0; j < c; ++j)
        {
            chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VCreateActor("pointlight.xml");
            chimera::TransformComponent* cmp;
            actor->VQueryComponent(CM_CMP_TRANSFORM, (chimera::IActorComponent**)&cmp);
            cmp->GetTransformation()->SetTranslation(scale*i, 5, scale*j);

            chimera::LightComponent* lightCmp;
            actor->VQueryComponent(CM_CMP_LIGHT, (chimera::IActorComponent**)&lightCmp);
            lightCmp->m_color.Set(rng.NextFloat(), rng.NextFloat(), rng.NextFloat(), 1);
            lightCmp->m_intensity = 1;
        }
    }
}

void createGeometry(float* tri, uint count)
{
    std::unique_ptr<chimera::ActorDescription> triDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();

    triDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
    chimera::RenderComponent* rc = triDesc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
    rc->m_geo = std::move(chimera::CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry());

    rc->m_drawType = "wire";
    rc->m_geo->VSetTopology(chimera::eTopo_Triangles);
    rc->m_geo->VSetVertexBuffer(tri, count, 8 * sizeof(float));
    rc->m_geo->VCreate();

    chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(triDesc));
}

void run(HINSTANCE hInstance)
{
    chimera::CM_APP_DESCRIPTION desc;
    ZeroMemory(&desc, sizeof(chimera::CM_APP_DESCRIPTION));
    desc.hInstance = hInstance;

    chimera::CmCreateApplication(&desc);

    //createSpheres();
    //createLights();
    //chimera::CmGetApp()->VGetLogic()->VLoadLevel("testlevel.xml");
    //chimera::CmGetApp()->VGetLogic()->VCreateActor("ruin.xml");
    //chimera::CmGetApp()->VGetLogic()->VCreateActor("ton.xml");

    ZeroMemory(spheres, sizeof(spheres));

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind t spawnactor ton.xml");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("transformSpheres", &killSphere);
    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand("bind g transformSpheres");

    chimera::CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VRegisterCommand("compileCudaKernel", &compileCudakernel);

    chimera::IProcess* proc = new chimera::FunctionProcess(&spawnSphereProc);
    chimera::CmGetApp()->VGetLogic()->VGetProcessManager()->VAttach(std::shared_ptr<chimera::IProcess>(proc));

    //chimera::CmGetApp()->VGetCache()->VGetHandleAsync(chimera::CMResource("twoObjTest.obj"), callback);

    chimera::IGuiTextureComponent* text = chimera::CmGetApp()->VGetHumanView()->VGetGuiFactory()->VCreateTextureComponent();
    text->VSetTexture("crosshair_dot.png");
    //text->VSetAlpha(1.0f);
    //text->VSetBackgroundColor(0,0,0);
    chimera::CMDimension dim;
    dim.x = chimera::CmGetApp()->VGetWindowWidth() / 2 - 4;
    dim.y = chimera::CmGetApp()->VGetWindowHeight() / 2 - 4;
    dim.h = 8;
    dim.w = 8;
    //text->VSetAlignment(chimera::eGuiAlignCenter);
    text->VSetDimension(dim);
    chimera::CmGetApp()->VGetHumanView()->VAddScreenElement(std::unique_ptr<chimera::IScreenElement>(text));

    std::unique_ptr<chimera::ActorDescription> particleDesc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
    chimera::ParticleComponent* pc = particleDesc->AddComponent<chimera::ParticleComponent>(CM_CMP_PARTICLE);
    chimera::TransformComponent* tpc = particleDesc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
    //chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(particleDesc));

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




