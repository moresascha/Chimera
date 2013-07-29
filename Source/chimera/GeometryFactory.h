#pragma once
#include "stdafx.h"
#include "Geometry.h"

FLOAT* GetSerpent(UINT xtiles, UINT ztiles, FLOAT scale = 1);

class GeometryFactory
{
private:
    static d3d::Geometry* m_sScreenQuad;
    static d3d::Geometry* m_sScreenQuadCPU;
    static d3d::Geometry* m_sSphere;
    static d3d::Geometry* m_sCube;
    static d3d::Geometry* m_sDefCube;
    static d3d::Geometry* m_sSkyDome;
    static d3d::Geometry* m_sFrustumGeomety;
    static d3d::Geometry* m_sLine;
public:
    GeometryFactory(VOID);
    static d3d::Geometry* GetGlobalScreenQuad(VOID);
    static d3d::Geometry* GetGlobalLineCPU(VOID);
    static d3d::Geometry* GetGlobalScreenQuadCPU(VOID);
    static d3d::Geometry* CreateScreenQuad(VOID);
    static d3d::Geometry* GetGlobalBoundingBoxCube(VOID);
    static d3d::Geometry* GetGlobalDefShadingCube(VOID);
    static d3d::Geometry* CreateScreenQuadCPUAcc(VOID);
    static d3d::Geometry* GetSphere(VOID);
    static d3d::Geometry* CreateNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale = 1, BOOL deleteRawData = TRUE);
    static d3d::Geometry* CreateAlternatingNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale = 1, BOOL deleteRawData = TRUE);
    static d3d::Geometry* CreateSphere(UINT segmentsX, UINT segmentsY, BOOL deleteRawData = TRUE);
    static d3d::Geometry* GetSkyDome(VOID);
    static d3d::Geometry* GetFrustumGeometry(VOID);
    static FLOAT* GetSphereVertexBuffer(UINT segmentsX, UINT segmentsY);
    static VOID Destroy(VOID);
    ~GeometryFactory(VOID);
};

