#pragma once
#include "CMTypes.h"
#include "CMCommon.h"

namespace chimera
{
    class IScreenElement
    {
    public:
        virtual CONST CMDimension& VGetDimension(VOID) = 0;

        virtual VOID VSetDimension(CONST CMDimension& dim) = 0;

        virtual UINT VGetPosX(VOID) CONST = 0;

        virtual UINT VGetPosY(VOID) CONST = 0;

        virtual UINT VGetWidth(VOID) CONST = 0;

        virtual UINT VGetHeight(VOID) CONST = 0;

        virtual LPCSTR VGetName(VOID) CONST = 0;

        virtual VOID VSetName(LPCSTR name) = 0;

        virtual VOID VDraw(VOID) = 0;

        virtual BOOL VOnRestore(VOID) = 0;

        virtual BOOL VIsEnable(VOID) CONST = 0;

        virtual VOID VSetEnable(BOOL enable) = 0;

        virtual BOOL VIsActive(VOID) CONST = 0;

        virtual VOID VSetActive(BOOL active) = 0;

        virtual VOID VUpdate(ULONG millis) = 0;

        virtual VOID VSetAlpha(FLOAT alpha) = 0;

        virtual VOID VSetBackgroundColor(FLOAT r, FLOAT g, FLOAT b) = 0;

        virtual CONST Color& VGetBackgroundColor(VOID) CONST = 0;

        virtual FLOAT VGetAlpha(VOID) CONST = 0;

        virtual ~IScreenElement(VOID) {}
    };

    class IRenderScreen
    {
    public:
        virtual BOOL VOnRestore(VOID) = 0;

        virtual IGraphicsSettings* VGetSettings(VOID) = 0;

        virtual VOID VDraw(VOID) = 0;

        virtual VOID VSetName(LPCSTR name) = 0;

        virtual CONST CMDimension& VGetDimension(VOID) = 0;

        virtual VOID VSetDimension(CONST CMDimension& dim) = 0;

        virtual LPCSTR VGetName(VOID) CONST = 0;

        virtual ~IRenderScreen(VOID) {}
    };

    /*class IRendertargetScreen
    {
    public:
        IRendertargetScreen(std::unique_ptr<IRenderTarget> target);

        VOID VDraw(VOID);

        ~IRendertargetScreen(VOID) {}
    };*/
}