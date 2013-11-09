#pragma once
#include "stdafx.h"

namespace chimera
{
    namespace geometryfactroy
    {
        CM_DLL_API IGeometry* CreateScreenQuad(BOOL cpuWrite = FALSE);

        CM_DLL_API IGeometry* GetGlobalScreenQuad(VOID);

        CM_DLL_API IGeometry* GetGlobalScreenQuadCPU(VOID);

        CM_DLL_API IGeometry* GetGlobalLineCPU(VOID);

        CM_DLL_API VOID Destroy(VOID);
/*

        CM_DLL_API FLOAT* GetSerpent(UINT xtiles, UINT ztiles, FLOAT scale = 1);

        CM_DLL_API IGeometry* GetGlobalScreenQuad(VOID);

        CM_DLL_API IGeometry* GetGlobalLineCPU(VOID);

        CM_DLL_API IGeometry* GetGlobalScreenQuadCPU(VOID);

        CM_DLL_API IGeometry* GetGlobalBoundingBoxCube(VOID);

        CM_DLL_API IGeometry* GetGlobalDefShadingCube(VOID);

        CM_DLL_API IGeometry* CreateScreenQuadCPUAcc(VOID);

        CM_DLL_API IGeometry* GetSphere(VOID);

        CM_DLL_API IGeometry* CreateNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale = 1, BOOL deleteRawData = TRUE);

        CM_DLL_API IGeometry* CreateAlternatingNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale = 1, BOOL deleteRawData = TRUE);

        CM_DLL_API IGeometry* CreateSphere(UINT segmentsX, UINT segmentsY, BOOL deleteRawData = TRUE);

        CM_DLL_API IGeometry* GetSkyDome(VOID);

        CM_DLL_API IGeometry* GetFrustumGeometry(VOID);

        CM_DLL_API FLOAT* GetSphereVertexBuffer(UINT segmentsX, UINT segmentsY);*/
    }
}