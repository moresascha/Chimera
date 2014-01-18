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
    bool APIENTRY DllMain(HINSTANCE hInst, DWORD reason, LPVOID lpReserved);

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
        const CM_APP_DESCRIPTION* 
        CM_API 
        CmGetDescription(
            void
            );

        CM_DLL_API 
        ErrorCode 
        CM_API 
        CmGetError(
            void
            );

        CM_DLL_API 
        IApplication* 
        CM_API 
        CmCreateApplication(
            CM_APP_DESCRIPTION* desc
            );

        CM_DLL_API 
        void 
        CM_API 
        CmReleaseApplication(
        void
        );

        CM_DLL_API 
        IApplication* 
        CM_API 
        CmGetApp(
            void
            );

        CM_DLL_API 
        void 
        CM_API 
        CmLog(
            const std::string& tag, 
            const std::string& message, 
            const char* funcName, 
            const char* file, 
            const uint line
            );

        CM_DLL_API
        void 
        CM_API 
        CmCriticalError(
            const std::string& tag, 
            const std::string& message, 
            const char* funcName,
            const char* file, 
            const uint line
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
#include "ParticleSystemAPI.h"

