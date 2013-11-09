/************************************************************************/
/* Chimera API Entry                                                    */
/************************************************************************/
#pragma once
#include "CMDefines.h"
#include "CMTypes.h"

#include "../Logger.h"
#include "ApplicationAPI.h"

extern "C"
{
    BOOL APIENTRY DllMain(HINSTANCE hInst, DWORD reason, LPVOID lpReserved);

    namespace chimera
    {
        struct CM_APP_DESCRIPTION
        {
            UCHAR ival;
            std::wstring titel;
            std::string logFile;
            std::string cachePath;
            std::string args;
            HINSTANCE hInstance;
            FactoryPtr* facts;
        };

        CM_DLL_API 
        CONST CM_APP_DESCRIPTION* 
        CM_API 
        CmGetDescription(
            VOID
            );

        CM_DLL_API 
        ErrorCode 
        CM_API 
        CmGetError(
            VOID
            );

        CM_DLL_API 
        IApplication* 
        CM_API 
        CmCreateApplication(
            CM_APP_DESCRIPTION* desc
            );

        CM_DLL_API 
        VOID 
        CM_API 
        CmReleaseApplication(
        VOID
        );

        CM_DLL_API 
        IApplication* 
        CM_API 
        CmGetApp(
            VOID
            );

        CM_DLL_API 
        VOID 
        CM_API 
        CmLog(
            CONST std::string& tag, 
            CONST std::string& message, 
            CONST CHAR* funcName, 
            CONST CHAR* file, 
            CONST UINT line
            );

        CM_DLL_API
        VOID 
        CM_API 
        CmCriticalError(
            CONST std::string& tag, 
            CONST std::string& message, 
            CONST CHAR* funcName,
            CONST CHAR* file, 
            CONST UINT line
            );
    }
};


#include "EventAPI.h"
#include "FontAPI.h"
#include "GraphicsAPI.h"
#include "ViewAPI.h"
#include "SoundAPI.h"
#include "ScriptAPI.h"

#include "LogicAPI.h"
#include "ResourceAPI.h"
#include "InputAPI.h"
#include "ActorAPI.h"
#include "ScreenAPI.h"
#include "GuiAPI.h"

