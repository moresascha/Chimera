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

chimera::TransformComponent* cmp;

void run(HINSTANCE hInstance)
{
    chimera::CM_APP_DESCRIPTION desc;
    ZeroMemory(&desc, sizeof(chimera::CM_APP_DESCRIPTION));
    desc.hInstance = hInstance;

    chimera::CmCreateApplication(&desc);

    chimera::IActor* actor = chimera::CmGetApp()->VGetLogic()->VCreateActor("ton.xml");
    cmp = (chimera::TransformComponent*)actor->VGetComponent(CM_CMP_TRANSFORM);

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
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    run(hInstance);

    return 0;
}




