// D3D.cpp : Defines the entry point for the application.
//
#include "stdafx.h"

/*static struct LeaksLocator
{
    LeaksLocator()
    {
        _CrtSetBreakAlloc(82565);
    }
} LeaksLocatorInst; */

#include "chimera/BasicApp.h"
#include "chimera/testgame/packman/Packman.h"
#include "chimera/script/Script.h"
#include "chimera/testgame/packman/Maze.h"
#include "chimera/util/math/Spline.h"

#define MAX_LOADSTRING 100
#undef FULL_SCREEN    
// Global Variables:
TCHAR szTitle[MAX_LOADSTRING];                         // The title bar text
TCHAR szWindowClass[MAX_LOADSTRING];               // the main window class name

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

    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    /*util::UniformBSpline s;
    
    s.AddPoint(util::Vec3(-10,10,0));
    s.AddPoint(util::Vec3(0,10,10));
    s.AddPoint(util::Vec3(+10,10,0));
    s.AddPoint(util::Vec3(0,10,-10));

    s.AddPoint(util::Vec3(0,10,10));
    s.AddPoint(util::Vec3(+10,10,0));
    s.AddPoint(util::Vec3(0,10,-10));

    s.Create(); */

    app::GameApp* g_pApp = new app::BasicApp(); //packman::Packman();//

#ifdef _DEBUG
    g_pApp->Init(hInstance, L"TBD (DEBUG BUILD)", "files", "log.log");
#else
    g_pApp->Init(hInstance, L"TBD (RELEASE BUILD)", "files", "log.log");
#endif


    //post init pre run code here
    //g_pApp->GetLogic()->VLoadLevel(new packman::Maze(100, 3));//

    g_pApp->Run();

    delete g_pApp;

    _CrtDumpMemoryLeaks();

     return 0;
}




