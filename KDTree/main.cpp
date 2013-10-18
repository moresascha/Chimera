#pragma warning(disable: 4244)

#include <ChimeraAPI.h>
#include "../Source/chimera/util.h"
#include "../Source/chimera/Timer.h"
#include "../Source/chimera/Logger.h"
#include "../Source/chimera/Components.h"
#include "../Source/chimera/AxisAlignedBB.h"
#include <fstream>

#ifdef _DEBUG
#pragma comment(lib, "Chimerax64Debug.lib")
#else
#pragma comment(lib, "Chimerax64Release.lib")
#endif

extern "C" void generate(void);

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

    generate();

    return 0;
}