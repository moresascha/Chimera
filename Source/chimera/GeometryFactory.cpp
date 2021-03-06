#include "GeometryFactory.h"

namespace chimera
{
    namespace geometryfactroy
    {
        IGeometry* g_globalCPUWriteQuad = NULL;
        IGeometry* g_globalQuad = NULL;
        IGeometry* g_globalCPUWriteLine = NULL;

        IGeometry* CreateScreenQuad(bool cpuWrite)
        {
            IGeometry* geo = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry().release();

            const uint stride = 5;
            const uint count = 4;
            float localVertices[count * stride] = 
            {
                -1, -1, 0, 0, 1,
                +1, -1, 0, 1, 1,
                -1, +1, 0, 0, 0,
                +1, +1, 0, 1, 0
            };

            geo->VSetTopology(eTopo_TriangleStrip);
            geo->VSetVertexBuffer(localVertices, count, stride * sizeof(float), cpuWrite);
            geo->VCreate();
            return geo;
        }

        IGeometry* GetGlobalScreenQuad(void)
        {
            if(!g_globalQuad)
            {
                g_globalQuad = CreateScreenQuad(false);
            }
            return g_globalQuad;
        }

        IGeometry* GetGlobalScreenQuadCPU(void)
        {
            if(!g_globalCPUWriteQuad)
            {
                g_globalCPUWriteQuad = CreateScreenQuad(true);
            }
            return g_globalCPUWriteQuad;
        }

        IGeometry* GetGlobalLineCPU(void)
        {
            if(!g_globalCPUWriteLine)
            {
                g_globalCPUWriteLine = CmGetApp()->VGetHumanView()->VGetGraphicsFactory()->VCreateGeoemtry().release();

                const uint stride = 5;
                const uint count = 2;
                float* vertices = new float[stride * count];
                uint* indices = new uint[count];
                //3,2,0,1
                indices[0] = 0; indices[1] = 1;

                float localVertices[count * stride] = {
                    0, 0, 0, 0, 0,
                    1, 1, 0, 1, 1,
                };

                for(uint i = 0; i < count * stride; ++i)
                {
                    vertices[i] = localVertices[i];
                }

                g_globalCPUWriteLine->VSetIndexBuffer(indices, count);
                g_globalCPUWriteLine->VSetTopology(eTopo_Lines);
                g_globalCPUWriteLine->VSetVertexBuffer(vertices, count, stride * sizeof(float), true);
                g_globalCPUWriteLine->VCreate();

                delete[] indices;
                delete[] vertices;
                return g_globalCPUWriteLine;
            }

            return g_globalCPUWriteLine;
        }

        void SafeDestroy(IGeometry* geo)
        {
            if(geo)
            {
                geo->VDestroy();
            }
            SAFE_DELETE(geo);
        }

        void Destroy(void)
        {
            SafeDestroy(g_globalCPUWriteQuad);
            SafeDestroy(g_globalQuad);
            SafeDestroy(g_globalCPUWriteLine);
        }
    }

    
   /*
    IGeometry* GetGlobalLineCPU(VOID)
    {

        IGeometry* m_sLine = new Geometry();

        CONST UINT stride = 5;
        CONST UINT count = 2;
        FLOAT* vertices = new FLOAT[stride * count];
        UINT* indices = new UINT[count];
        //3,2,0,1
        indices[0] = 0; indices[1] = 1;

        FLOAT localVertices[count * stride] = {
            0, 0, 0, 0, 0,
            1, 1, 0, 1, 1,
        };

        for(UINT i = 0; i < count * stride; ++i)
        {
            vertices[i] = localVertices[i];
        }

        m_sLine->VSetIndexBuffer(indices, count);
        m_sLine->VSetTopology(eTopo_Lines);
        m_sLine->VSetVertexBuffer(vertices, count, stride * sizeof(FLOAT));
        m_sLine->GetVertexBuffer()->SetCPUAccess(D3D11_CPU_ACCESS_WRITE);
        m_sLine->GetVertexBuffer()->SetUsage(D3D11_USAGE_DYNAMIC);
        m_sLine->VCreate();

        delete[] indices;
        delete[] vertices;
        return m_sLine;
    }

    IGeometry* GetFrustumGeometry(VOID)
    {
        IGeometry* m_sFrustumGeomety = new Geometry();
        CONST UINT vertexCount = 24;
        CONST UINT vertexStride = 8;
        //pos,norm,color,tex
        FLOAT* vertices = new FLOAT[vertexStride * vertexCount];
        UINT* indices = new UINT[vertexCount + 6];

        FLOAT vertexTmp[vertexStride * vertexCount] = 
        {
            //vorne
            -1, -1, -1, / ** /0, 0, -1, / ** / 0, 1,
            -1, +1, -1, / ** /0, 0, -1, / ** / 0, 0,
            +1, +1, -1, / ** /0, 0, -1, / ** / 1, 0,
            +1, -1, -1, / ** /0, 0, -1, / ** / 1, 1,

            //hinten
            -1, -1, +1, / ** /0, 0, +1, / ** / 1, 0,
            -1, +1, +1, / ** /0, 0, +1, / ** / 1, 1,
            +1, +1, +1, / ** /0, 0, +1, / ** / 0, 1,
            +1, -1, +1, / ** /0, 0, +1, / ** / 0, 0,

            //links
            -1, -1, -1, / ** /-1, 0, 0, / ** / 1, 0, //8
            -1, -1, +1, / ** /-1, 0, 0, / ** / 0, 0, //9
            -1, +1, -1, / ** /-1, 0, 0, / ** / 1, 1, //10
            -1, +1, +1, / ** /-1, 0, 0, / ** / 0, 1, //11

            //rechts
            +1, -1, -1, / ** /+1, 0, 0, / ** / 0, 0, //12
            +1, -1, +1, / ** /+1, 0, 0, / ** / 1, 0, //13
            +1, +1, -1, / ** /+1, 0, 0, / ** / 0, 1, //14
            +1, +1, +1, / ** /+1, 0, 0, / ** / 1, 1, //15

            //oben
            +1, +1, +1, / ** /0, +1, 0, / ** / 1, 1, //16
            +1, +1, -1, / ** /0, +1, 0, / ** / 1, 0, //17
            -1, +1, +1, / ** /0, +1, 0, / ** / 0, 1, //18
            -1, +1, -1, / ** /0, +1, 0, / ** / 0, 0, //19

            //unten
            +1, -1, +1, / ** /0, -1, 0, / ** / 0, 1, //20
            +1, -1, -1, / ** /0, -1, 0, / ** / 0, 0, //21
            -1, -1, +1, / ** /0, -1, 0, / ** / 1, 1, //22
            -1, -1, -1, / ** /0, -1, 0, / ** / 1, 0, //23
        };

        UINT indexTmp[vertexCount + 6] = 
        {
            3,2,0,1,-1,
            5,6,4,7,-1,
            8,10,9,11,-1,
            13,15,12,14,-1,
            17,16,19,18,-1,
            20,21,22,23,-1
        };

        for(UINT i = 0; i < vertexCount * vertexStride; ++i) 
        {
            vertices[i] = vertexTmp[i];
            if(i < vertexCount + 6)
            {
                indices[i] = indexTmp[i];
            }
        }

        m_sFrustumGeomety->VSetIndexBuffer(indices, vertexCount + 6);
        m_sFrustumGeomety->VSetTopology(eTopo_TriangleStrip);
        m_sFrustumGeomety->VSetVertexBuffer(vertices, vertexCount, 4 * vertexStride);
        m_sFrustumGeomety->GetVertexBuffer()->SetCPUAccess(D3D11_CPU_ACCESS_WRITE);
        m_sFrustumGeomety->GetVertexBuffer()->SetUsage(D3D11_USAGE_DYNAMIC);
        m_sFrustumGeomety->VCreate();
        SAFE_ARRAY_DELETE(vertices);
        SAFE_ARRAY_DELETE(indices);

        return m_sFrustumGeomety;
    }

    IGeometry* GetGlobalScreenQuadCPU(VOID)
    {
        if(!m_sScreenQuadCPU)
        {
            m_sScreenQuadCPU = CreateScreenQuadCPUAcc();
        }
        return m_sScreenQuadCPU;
    }

    Geometry* CreateScreenQuad(VOID)
    {
        return _CreateScreenQuad((D3D11_CPU_ACCESS_FLAG)0);
    }

    Geometry* CreateScreenQuadCPUAcc(VOID)
    {
        return _CreateScreenQuad(D3D11_CPU_ACCESS_WRITE);
    }

    Geometry* GetGlobalDefShadingCube(VOID)
    {

        IGeometry* geo = new Geometry();

        CONST UINT vertexCount = 24;
        CONST UINT vertexStride = 8;
        //pos,norm,color,tex
        FLOAT* vertices = new FLOAT[vertexStride * vertexCount];
        UINT* indices = new UINT[vertexCount + 6];

        FLOAT vertexTmp[vertexStride * vertexCount] = 
        {
            //vorne
            -1, -1, -1, / ** /0, 0, -1, / ** / 0, 1,
            -1, +1, -1, / ** /0, 0, -1, / ** / 0, 0,
            +1, +1, -1, / ** /0, 0, -1, / ** / 1, 0,
            +1, -1, -1, / ** /0, 0, -1, / ** / 1, 1,

            //hinten
            -1, -1, +1, / ** /0, 0, +1, / ** / 1, 0,
            -1, +1, +1, / ** /0, 0, +1, / ** / 1, 1,
            +1, +1, +1, / ** /0, 0, +1, / ** / 0, 1,
            +1, -1, +1, / ** /0, 0, +1, / ** / 0, 0,

            //links
            -1, -1, -1, / ** /-1, 0, 0, / ** / 1, 0, //8
            -1, -1, +1, / ** /-1, 0, 0, / ** / 0, 0, //9
            -1, +1, -1, / ** /-1, 0, 0, / ** / 1, 1, //10
            -1, +1, +1, / ** /-1, 0, 0, / ** / 0, 1, //11

            //rechts
            +1, -1, -1, / ** /+1, 0, 0, / ** / 0, 0, //12
            +1, -1, +1, / ** /+1, 0, 0, / ** / 1, 0, //13
            +1, +1, -1, / ** /+1, 0, 0, / ** / 0, 1, //14
            +1, +1, +1, / ** /+1, 0, 0, / ** / 1, 1, //15

            //oben
            +1, +1, +1, / ** /0, +1, 0, / ** / 1, 1, //16
            +1, +1, -1, / ** /0, +1, 0, / ** / 1, 0, //17
            -1, +1, +1, / ** /0, +1, 0, / ** / 0, 1, //18
            -1, +1, -1, / ** /0, +1, 0, / ** / 0, 0, //19

            //unten
            +1, -1, +1, / ** /0, -1, 0, / ** / 0, 1, //20
            +1, -1, -1, / ** /0, -1, 0, / ** / 0, 0, //21
            -1, -1, +1, / ** /0, -1, 0, / ** / 1, 1, //22
            -1, -1, -1, / ** /0, -1, 0, / ** / 1, 0, //23
        };

        UINT indexTmp[vertexCount + 6] = 
        {
            3,2,0,1,-1,
            5,6,4,7,-1,
            8,10,9,11,-1,
            13,15,12,14,-1,
            17,16,19,18,-1,
            20,21,22,23,-1
        };

        for(UINT i = 0; i < vertexCount * vertexStride; ++i) 
        {
            vertices[i] = vertexTmp[i];
            if(i < vertexCount + 6)
            {
                indices[i] = indexTmp[i];
            }
        }

        m_sDefCube->VSetIndexBuffer(indices, vertexCount + 6);
        m_sDefCube->VSetTopology(eTopo_TriangleStrip);
        m_sDefCube->VSetVertexBuffer(vertices, vertexCount, 4 * vertexStride);
        m_sDefCube->VCreate();
        SAFE_ARRAY_DELETE(vertices);
        SAFE_ARRAY_DELETE(indices);
        return m_sDefCube;
    }

    Geometry* GetGlobalBoundingBoxCube(VOID)
    {

        m_sCube = new Geometry();

        CONST UINT vertexCount = 24;
        CONST UINT vertexStride = 12;
        //pos,norm,color,tex
        FLOAT* vertices = new FLOAT[vertexStride * vertexCount];
        UINT* indices = new UINT[vertexCount + 6];

        FLOAT vertexTmp[vertexStride * vertexCount] = 
        {
            //vorne
            -1, -1, -1,
            -1, +1, -1,
            +1, +1, -1,
            +1, -1, -1,

            //hinten
            -1, -1, +1,
            -1, +1, +1,
            +1, +1, +1,
            +1, -1, +1,

            //links
            -1, -1, -1,
            -1, -1, +1,
            -1, +1, -1,
            -1, +1, +1,

            //rechts
            +1, -1, -1,
            +1, -1, +1,
            +1, +1, -1,
            +1, +1, +1,

            //oben
            +1, +1, +1,
            +1, +1, -1,
            -1, +1, +1,
            -1, +1, -1,

            //unten
            +1, -1, +1,
            +1, -1, -1,
            -1, -1, +1,
            -1, -1, -1,
        };

        UINT indexTmp[vertexCount + 6] = 
        {
            3,2,0,1,-1,
            5,6,4,7,-1,
            8,10,9,11,-1,
            13,15,12,14,-1,
            17,16,19,18,-1,
            20,21,22,23,-1
        };

        for(UINT i = 0; i < vertexCount * vertexStride; ++i) 
        {
            vertices[i] = vertexTmp[i];
            if(i < vertexCount + 6)
            {
                indices[i] = indexTmp[i];
            }
        }

        m_sCube->VSetIndexBuffer(indices, vertexCount + 6);
        m_sCube->VSetTopology(eTopo_TriangleStrip);
        m_sCube->VSetVertexBuffer(vertices, vertexCount, vertexStride);
        m_sCube->VCreate();
        SAFE_ARRAY_DELETE(vertices);
        SAFE_ARRAY_DELETE(indices);
        return m_sCube;
    }

    FLOAT* GetSphereVertexBuffer(UINT segmentsX, UINT segmentsY)
    {
        UINT vertexCount = (segmentsX + 1) * (segmentsY + 1);

        FLOAT* vertexBuffer = new FLOAT[vertexCount * 8];

        FLOAT dphi = XM_2PI / (FLOAT)segmentsX;
        FLOAT dtheta = XM_PI / (FLOAT)segmentsY;
        FLOAT theta = -XM_PI;
        UINT vc = 0;

        for(UINT i = 0; i <= segmentsY; ++i)
        {
            FLOAT phi = 0;
            for(UINT j = 0; j <= segmentsX; ++j)
            {
                vertexBuffer[vc++] = sin(theta) * cos(phi);
                vertexBuffer[vc++] = cos(theta);
                vertexBuffer[vc++] = sin(theta) * sin(phi);

                vertexBuffer[vc++] = sin(theta) * cos(phi);
                vertexBuffer[vc++] = cos(theta);
                vertexBuffer[vc++] = sin(theta) * sin(phi);

                vertexBuffer[vc++] = phi / XM_2PI;
                vertexBuffer[vc++] = (XM_PI + theta) / XM_PI;
                phi += dphi;
            }
            theta += dtheta;
        }

        return vertexBuffer;
    }

    Geometry* CreateSphere(UINT segmentsX, UINT segmentsY, BOOL deleteRawData)
    {
        / *UINT segmentsX = 32;
        UINT segmentsY = 16; * /
        UINT indexCount = segmentsY * 2 * (segmentsX + 1) + segmentsY;
        UINT vertexCount = (segmentsX + 1) * (segmentsY + 1);

        UINT* indexBuffer = new UINT[indexCount];
        FLOAT* vertexBuffer = GetSphereVertexBuffer(segmentsX, segmentsY);//new FLOAT[vertexCount * 8];
        UINT ic = 0;
        / *
        FLOAT dphi = 2 * XM_PI / (FLOAT)segmentsX;
        FLOAT dtheta = XM_PI / (FLOAT)segmentsY;
        FLOAT phi = 0;
        FLOAT theta = -XM_PI;
        UINT vc = 0;

        for(UINT i = 0; i <= segmentsY; ++i)
        {
            for(UINT j = 0; j <= segmentsX; ++j)
            {
                vertexBuffer[vc++] = sin(theta) * cos(phi);
                vertexBuffer[vc++] = cos(theta);
                vertexBuffer[vc++] = sin(theta) * sin(phi);

                vertexBuffer[vc++] = sin(theta) * cos(phi);
                vertexBuffer[vc++] = cos(theta);
                vertexBuffer[vc++] = sin(theta) * sin(phi);

                vertexBuffer[vc++] = phi / XM_2PI;
                vertexBuffer[vc++] = (XM_PI + theta) / XM_PI;
                phi += dphi;
            }
            phi = 0;
            theta += dtheta;
        } * /

        for(UINT i = 0; i < segmentsY; ++i)
        {
            for(UINT j = 0; j <= segmentsX; ++j)
            {
                indexBuffer[ic++] = i * (segmentsX+1) + j + segmentsX + 1;
                indexBuffer[ic++] = i * (segmentsX+1) + j;
            }
            indexBuffer[ic++] = -1;
        }

        Geometry* geo = new Geometry(TRUE);
        geo->VSetIndexBuffer(indexBuffer, indexCount);
        geo->VSetTopology(eTopo_TriangleStrip);
        geo->VSetVertexBuffer(vertexBuffer, vertexCount, 8 * sizeof(FLOAT));
        geo->VCreate();

        if(deleteRawData)
        {
            geo->DeleteRawData();
        }

        return geo;
    }

    Geometry* GetSphere(VOID)
    {
        if(m_sSphere)
        {
            return m_sSphere;
        }
        m_sSphere = CreateSphere(32, 16);
        return m_sSphere;
    }

    Geometry* GetSkyDome(VOID)
    {

        UINT segmentsX = 64;
        UINT segmentsY = 8;
        UINT indexCount = segmentsY * 2 * (segmentsX + 1) + segmentsY;
        UINT vertexCount = (segmentsX + 1) * (segmentsY + 1);

        UINT* indexBuffer = new UINT[indexCount];
        FLOAT* vertexBuffer = new FLOAT[vertexCount * 5];

        FLOAT dphi = 2 * XM_PI / (FLOAT)segmentsX;
        FLOAT dtheta = XM_PI / (FLOAT)segmentsY;
        FLOAT phi = 0;
        FLOAT theta = 0;//-XM_PI;
        UINT ic = 0;
        UINT vc = 0;

        for(UINT i = 0; i <= segmentsY; ++i)
        {
            for(UINT j = 0; j <= segmentsX; ++j)
            {
                vertexBuffer[vc++] = sin(theta) * cos(phi);
                vertexBuffer[vc++] = cos(theta);
                vertexBuffer[vc++] = sin(theta) * sin(phi);

                vertexBuffer[vc++] = phi / XM_2PI;
                vertexBuffer[vc++] = 2 * (XM_PI + theta) / XM_PI;
                phi += dphi;
            }
            phi = 0;
            theta -= 0.5f * dtheta;
        }

        for(UINT i = 0; i < segmentsY; ++i)
        {
            for(UINT j = 0; j <= segmentsX; ++j)
            {
                indexBuffer[ic++] = i * (segmentsX+1) + j + segmentsX + 1;
                indexBuffer[ic++] = i * (segmentsX+1) + j;
            }
            indexBuffer[ic++] = -1;
        }

        m_sSkyDome = new Geometry();
        m_sSkyDome->VSetIndexBuffer(indexBuffer, indexCount);
        m_sSkyDome->VSetTopology(eTopo_TriangleStrip);
        m_sSkyDome->VSetVertexBuffer(vertexBuffer, vertexCount, 5 * sizeof(FLOAT));
        m_sSkyDome->VCreate();

        delete[] indexBuffer;
        delete[] vertexBuffer;

        return m_sSkyDome;
    }

    Geometry* CreateNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale, BOOL deleteRawData)
    {
        UINT indexCount = (4 + (xtiles-1) * 2) * ztiles + ztiles;
        UINT vertexCount = (xtiles + 1) * (ztiles + 1);
        UINT vertexStride = 8;

        UINT* indexBuffer = new UINT[indexCount];
        FLOAT* vertexBuffer = new FLOAT[vertexCount * vertexStride];

        for(UINT y = 0; y <= ztiles; ++y)
        {
            UINT yd = (xtiles+1) * y;
            for(UINT x = 0; x <= xtiles; ++x)
            {
                vertexBuffer[vertexStride * (yd + x) + 0] = (-1.0f + 2.0f * x / (FLOAT)xtiles) * scale; 
                vertexBuffer[vertexStride * (yd + x) + 1] = 0;
                vertexBuffer[vertexStride * (yd + x) + 2] = (-1.0f + 2.0f * y / (FLOAT)ztiles) * scale;
                vertexBuffer[vertexStride * (yd + x) + 3] = 0;
                vertexBuffer[vertexStride * (yd + x) + 4] = 1;
                vertexBuffer[vertexStride * (yd + x) + 5] = 0;
                vertexBuffer[vertexStride * (yd + x) + 6] = x / (FLOAT)xtiles;
                vertexBuffer[vertexStride * (yd + x) + 7] = y / (FLOAT)ztiles;
            }
        }

        UINT ic = 0;
        for(UINT i = 0; i < ztiles; ++i)
        {
            for(UINT j = 0; j <= xtiles; ++j)
            {
                indexBuffer[ic++] = i * (xtiles+1) + j + xtiles + 1;
                indexBuffer[ic++] = i * (xtiles+1) + j;

            }
            indexBuffer[ic++] = -1;
        }

        / *TBD_FOR_INT(ic)
        {
            DEBUG_OUT_A("%d ", indexBuffer[i]);
        } * /
        
        Geometry* geo = new Geometry();
        geo->VSetIndexBuffer(indexBuffer, indexCount);
        geo->VSetVertexBuffer(vertexBuffer, vertexCount, vertexStride * sizeof(FLOAT));
        geo->VSetTopology(eTopo_TriangleStrip);
        geo->VCreate();

        if(deleteRawData)
        {
            SAFE_ARRAY_DELETE(vertexBuffer);
            SAFE_ARRAY_DELETE(indexBuffer);
        }

        return geo;
    }

    Geometry* CreateAlternatingNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale, BOOL deleteRawData)
    {
        UINT indexCount = (4 + (xtiles-1) * 2) * ztiles + ztiles;
        UINT vertexCount = (xtiles + 1) * (ztiles + 1);
        UINT vertexStride = 8;

        UINT* indexBuffer = new UINT[indexCount];
        FLOAT* vertexBuffer = new FLOAT[vertexCount * vertexStride];

        FLOAT x = -1;
        float sx = 1;
        FLOAT y = 1;
        FLOAT dx = 2 / (float)xtiles;
        UINT index = 0;
        for(UINT j = 0; j <= ztiles; ++j)
        {
            for(UINT i = 0; i <= xtiles; ++i)
            {
                vertexBuffer[index++] = x * scale; 
                vertexBuffer[index++] = 0;
                vertexBuffer[index++] = y * scale;
                vertexBuffer[index++] = 0;
                vertexBuffer[index++] = 1;
                vertexBuffer[index++] = 0;
                vertexBuffer[index++] = i / (FLOAT)xtiles;
                vertexBuffer[index++] = j / (FLOAT)ztiles;
                / *util::Vec3 v(x,0,y);
                v.Print(); * /
                x += sx * dx;

                if(x > 1)
                {
                    sx *= -1;
                    y -= 2/(float)ztiles;
                    x = 1;
                } else if(x < -1)
                {
                    x = -1;
                    sx *= -1;
                    y -= 2/(float)xtiles;
                }
            }
        }

        UINT ic = 0;
        BOOL toggle = TRUE;
        for(UINT i = 0; i < ztiles; ++i)
        {
            for(UINT j = 0; j <= xtiles; ++j)
            {
                if(toggle)
                {
                        indexBuffer[ic++] = i * (xtiles+1) + j;
                    indexBuffer[ic++] = i * (xtiles+1) + 2 * xtiles - j + 1;
                }
                else
                {
                    indexBuffer[ic++] = i * (xtiles+1) + 2 * xtiles - j + 1; 
                    indexBuffer[ic++] = i * (xtiles+1) + j;
                }
            }
            indexBuffer[ic++] = -1;
            toggle = !toggle;
        }


        / *TBD_FOR_INT(ic)
        {
            DEBUG_OUT_A("%d ", indexBuffer[i]);
        } * /
        
        Geometry* geo = new Geometry();
        geo->VSetIndexBuffer(indexBuffer, indexCount);
        geo->VSetTopology(eTopo_TriangleStrip);
        geo->VSetVertexBuffer(vertexBuffer, vertexCount, vertexStride * sizeof(FLOAT));
        geo->VCreate();

        if(deleteRawData)
        {
            SAFE_ARRAY_DELETE(vertexBuffer);
            SAFE_ARRAY_DELETE(indexBuffer);
        }

        return geo;
    }

    FLOAT* GetSerpent(UINT xtiles, UINT ztiles, FLOAT scale)
    {
        UINT vertexCount = (xtiles+1) * (ztiles+1);
        FLOAT* vertexBuffer = new FLOAT[3 * vertexCount];

        FLOAT x = -1;
        float sx = 1;
        FLOAT y = 1;
        FLOAT dx = 2.0f / (float)xtiles;
        UINT index = 0;
        for(UINT j = 0; j < vertexCount; ++j)
        {
            vertexBuffer[index++] = x * scale; 
            vertexBuffer[index++] = 0;
            vertexBuffer[index++] = y * scale;

            x += sx * dx;

            if(x > 1)
            {
                sx *= -1;
                y -= 2.0f/(float)ztiles;
                x = 1;
            } else if(x < -1)
            {
                x = -1;
                sx *= -1;
                y -= 2.0f/(float)ztiles;
            }
        }
        return vertexBuffer;
    }*/
}