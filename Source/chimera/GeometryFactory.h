#pragma once
#include "stdafx.h"

namespace chimera
{
    namespace d3d
    {
        class Geometry;
        FLOAT* GetSerpent(UINT xtiles, UINT ztiles, FLOAT scale = 1);

        class GeometryFactory
        {
        private:
            static Geometry* m_sScreenQuad;
            static Geometry* m_sScreenQuadCPU;
            static Geometry* m_sSphere;
            static Geometry* m_sCube;
            static Geometry* m_sDefCube;
            static Geometry* m_sSkyDome;
            static Geometry* m_sFrustumGeomety;
            static Geometry* m_sLine;
        public:
            GeometryFactory(VOID);
            static Geometry* GetGlobalScreenQuad(VOID);
            static Geometry* GetGlobalLineCPU(VOID);
            static Geometry* GetGlobalScreenQuadCPU(VOID);
            static Geometry* CreateScreenQuad(VOID);
            static Geometry* GetGlobalBoundingBoxCube(VOID);
            static Geometry* GetGlobalDefShadingCube(VOID);
            static Geometry* CreateScreenQuadCPUAcc(VOID);
            static Geometry* GetSphere(VOID);
            static Geometry* CreateNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale = 1, BOOL deleteRawData = TRUE);
            static Geometry* CreateAlternatingNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale = 1, BOOL deleteRawData = TRUE);
            static Geometry* CreateSphere(UINT segmentsX, UINT segmentsY, BOOL deleteRawData = TRUE);
            static Geometry* GetSkyDome(VOID);
            static Geometry* GetFrustumGeometry(VOID);
            static FLOAT* GetSphereVertexBuffer(UINT segmentsX, UINT segmentsY);
            static VOID Destroy(VOID);
            ~GeometryFactory(VOID);
        };
    }
}