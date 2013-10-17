#pragma warning(disable: 4244)

#include <ChimeraAPI.h>
#include "../Source/chimera/util.h"
#include "../Source/chimera/Timer.h"
#include "../Source/chimera/Logger.h"
#include "../Source/chimera/Components.h"
#include "../Source/chimera/AxisAlignedBB.h"
#include "Cudah.h"
#include "CUBuffer.h"
#include "CUModule.h"
#include "CUKernel.h"
#include "CUSort.h"

#include <fstream>

#ifdef _DEBUG
#pragma comment(lib, "Cudahx64Debug.lib")
#pragma comment(lib, "Chimerax64Debug.lib")
#else
#pragma comment(lib, "Cudahx64Release.lib")
#pragma comment(lib, "Chimerax64Release.lib")
#endif

typedef struct _SplitData
{
    UINT axis;
    float split;

} SplitData;

typedef struct _AABB
{
    float4 min;
    float4 max;
} AABB;

typedef struct _SAHSplit
{
    float split;
    float v;
} SAHSplit;

VOID GetSplitNAxis(UINT* axis, float* split, float3 mmin, float3 mmax)
{
    FLOAT w = mmax.x - mmin.x;
    FLOAT h = mmax.y - mmin.y;
    FLOAT d = mmax.z - mmin.z;
    *axis = (w >= d && w >= h) ? 0 : (h >= w && h >= d) ? 1 : 2;
    switch(*axis)
    {
    case 0: 
        {
            *split = mmin.x + (mmax.x - mmin.x) * 0.5f;
        } break;
    case 1: 
        {
            *split = mmin.y + (mmax.y - mmin.y) * 0.5f;
        } break;
    case 2: 
        {
            *split = mmin.z + (mmax.z - mmin.z) * 0.5f;
        } break;
    }
}

VOID startChimera(HINSTANCE hInstance)
{
	chimera::CM_APP_DESCRIPTION desc;
	desc.facts = NULL;
	desc.hInstance = hInstance;
	desc.titel = L"KD-TREE";
	desc.ival = 60;
	desc.cachePath = "../Assets/";
	desc.logFile = "log.log";

	chimera::IApplication* cm = chimera::CmCreateApplication(&desc);
}

/*
class BBNode : public tbd::SceneNode
{
public:
    util::AxisAlignedBB m_aabb;

    BBNode(util::AxisAlignedBB& aabb) : m_aabb(aabb)
    {
    }

    UINT VGetRenderPaths(VOID)
    {
        return tbd::eDRAW_TO_ALBEDO;
    } 

    VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        switch(path)
        {
        case tbd::eDRAW_TO_ALBEDO :
            {
                app::g_pApp->GetHumanView()->GetRenderer()->PushRasterizerState(d3d::g_pRasterizerStateWrireframe);
                app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                app::g_pApp->GetRenderer()->SetDefaultTexture();
                app::g_pApp->GetRenderer()->SetDefaultMaterial();
                app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                tbd::DrawBox(m_aabb);
                app::g_pApp->GetHumanView()->GetRenderer()->PopRasterizerState();
            } break;;
        }
    }

    ~BBNode(VOID)
    {
    }
}; */

CONST UINT depth = 5;
CONST UINT nodesCount = (1 << (depth)) - 1;
CONST UINT elememtCount = 8;
CONST UINT elementsPerBlock = elememtCount / 2;
CONST UINT resultCount = elememtCount / elementsPerBlock;
float4* points;

VOID shuffle(FLOAT* data, UINT l)
{
    chimera::util::cmRNG rng;
    TBD_FOR_INT(l)
    {
        data[i] = rng.NextFloat();
    }
}

VOID checkSort(FLOAT* data, UINT l)
{
    FLOAT s = 0;
    for(int j = 0; j < l; ++j)
    {
        FLOAT tmp = data[j];
        if(tmp<s)
        {
            DEBUG_OUT("error");
        }
        s = tmp;
    }
}

int APIENTRY tee(HINSTANCE hInstance,
                       HINSTANCE hPrevInstance,
                       LPTSTR    lpCmdLine,
                       int       nCmdShow)
{


#ifdef _DEBUG
    /*_CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR); */
#endif

    std::vector<chimera::util::AxisAlignedBB*> aas;
	/*
    startChimera(hInstance);

	std::unique_ptr<chimera::ActorDescription> desc = chimera::CmGetApp()->VGetLogic()->VGetActorFactory()->VCreateActorDescription();
	chimera::TransformComponent* tc = desc->AddComponent<chimera::TransformComponent>(CM_CMP_TRANSFORM);
	tc->GetTransformation()->Translate(chimera::util::Vec3(0,2,0));
	tc->GetTransformation()->SetScale(0.1f);
	chimera::RenderComponent* rc = desc->AddComponent<chimera::RenderComponent>(CM_CMP_RENDERING);
	rc->m_resource = "Box.obj";

	chimera::CmGetApp()->VGetLogic()->VCreateActor(std::move(desc));

	chimera::CmGetApp()->VRun();

	chimera::CmReleaseApplication();*/
    /*
    cudah::Init("../Cudah/ptx/Cudahu.ptx");

    UINT sortDataCnt = 1024;
    cudah::cuBuffer<FLOAT> csortData(sortDataCnt);
    
    for(cudah::cuIterator<FLOAT, BufferPtr, cudah::cuBuffer<FLOAT>> it = csortData.Begin(); it != csortData.End(); ++it)
    {
        it = (FLOAT)rand() / (FLOAT)RAND_MAX;
    }

    for(uint i = 0; i < sortDataCnt; ++i)
    {
        DEBUG_OUT_A("%f ", csortData[i]);
    }

    DEBUG_OUT("\n");

    bitonicSortPerGroup(csortData, 1024, 0, 0, 1024);

    for(uint i = 0; i < sortDataCnt; ++i)
    {
        DEBUG_OUT_A("%f ", csortData[i]);
    }

    DEBUG_OUT("\n");*/

    /*
    cudah::cuBuffer<FLOAT> b(50);
    cudah::Fill(b.Begin(), b.End(), 10.f);

    cudah::cuBuffer<FLOAT> c = b;

    cudah::Fill(c.Begin() + 5, c.Begin() + 10, 0.0f);

    //cudah::Copy(b.Begin(), b.End(), c.Begin(), c.End());
    cudah::cuModule m("./ptx/KD.ptx");

    cudah::cuKernel kernel(m.GetFunction("spreadContent"));

    TBD_FOR_INT(b.Size())
    {
        if(i < c.Size())
        {
            DEBUG_OUT_A("%f %f\n", b[i], c[i]);
        }
        else
        {
            DEBUG_OUT_A("%f\n", b[i]);
        }
    } */

    //cudah::Destroy();

    return 0;
}