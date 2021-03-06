#include "stdafx.h"
static struct LeaksLocator
{
    LeaksLocator()
    {
        //_CrtSetBreakAlloc(2082);
    }
} LeaksLocatorInst;

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

    LPCWSTR APPNAME;
    
#ifdef _DEBUG
    APPNAME = L"Chimera (Debugx64)";
#else
    APPNAME = L"Chimera (Releasex64)";
#endif

    //chimera::d3d::Init(WndProc, hInstance, L"asd", 100, 100);

    //chimera::FactoryPtr facts[] = {CM_SOUND_FACTORY, (size_t)&s, CM_GFX_FACTORY, (size_t)&g, CM_VIEW_FACTORY, (size_t)&v, CM_FACTORY_END};
    
    chimera::CM_APP_DESCRIPTION desc;
    desc.facts = NULL;
    desc.hInstance = hInstance;
    desc.titel = APPNAME;
    desc.ival = 60;
    desc.cachePath = "../Assets/";
    desc.logFile = "log.log";
    desc.args = "-console";

    chimera::IApplication* cm = chimera::CmCreateApplication(&desc);

/*
    chimera::IGui* gui = cm->VGetHumanView()->VGetGuiFactory()->VCreateGui();
    gui->VSetName(VIEW_CONSOLE_NAME);
    chimera::IGuiTextInputComponent* input = cm->VGetHumanView()->VGetGuiFactory()->VCreateTextInputComponent();
    chimera::CMDimension dim;
    dim.x = 100;
    dim.y = 100;
    dim.w = 100;
    dim.h = 16;
    input->VSetDimension(dim);

    gui->VAddComponent(input);

    cm->VGetHumanView()->VAddScreenElement(std::unique_ptr<chimera::IGui>(gui));*/

    chimera::CmGetApp()->VRun();

    chimera::CmReleaseApplication();

    //_CrtDumpMemoryLeaks();
    return 0;
}