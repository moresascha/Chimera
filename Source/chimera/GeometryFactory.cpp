#include "GeometryFactory.h"

    d3d::Geometry* GeometryFactory::m_sScreenQuad = NULL;
    d3d::Geometry* GeometryFactory::m_sScreenQuadCPU = NULL;
    d3d::Geometry* GeometryFactory::m_sSphere = NULL;
    d3d::Geometry* GeometryFactory::m_sCube = NULL;
    d3d::Geometry* GeometryFactory::m_sDefCube = NULL;
    d3d::Geometry* GeometryFactory::m_sFrustumGeomety = NULL;
    d3d::Geometry* GeometryFactory::m_sSkyDome = NULL;
    d3d::Geometry* GeometryFactory::m_sLine = NULL;

    GeometryFactory::GeometryFactory(VOID) {
    }

    d3d::Geometry* GeometryFactory::GetGlobalLineCPU(VOID)
    {
        if(m_sLine)
        {
            return m_sLine;
        }

        m_sLine = new d3d::Geometry();

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

        m_sLine->SetIndexBuffer(indices, count, D3D11_PRIMITIVE_TOPOLOGY_LINELIST);
        m_sLine->SetVertexBuffer(vertices, count, stride * sizeof(FLOAT));
        m_sLine->GetVertexBuffer()->SetCPUAccess(D3D11_CPU_ACCESS_WRITE);
        m_sLine->GetVertexBuffer()->SetUsage(D3D11_USAGE_DYNAMIC);
        m_sLine->VCreate();

        delete[] indices;
        delete[] vertices;
        return m_sLine;
    }

    d3d::Geometry* GeometryFactory::GetFrustumGeometry(VOID)
    {
        if(m_sFrustumGeomety)
        {
            return m_sFrustumGeomety;
        }

        m_sFrustumGeomety = new d3d::Geometry();
        CONST UINT vertexCount = 24;
        CONST UINT vertexStride = 8;
        //pos,norm,color,tex
        FLOAT* vertices = new FLOAT[vertexStride * vertexCount];
        UINT* indices = new UINT[vertexCount + 6];

        FLOAT vertexTmp[vertexStride * vertexCount] = 
        {
            //vorne
            -1, -1, -1, /**/0, 0, -1, /**/ 0, 1,
            -1, +1, -1, /**/0, 0, -1, /**/ 0, 0,
            +1, +1, -1, /**/0, 0, -1, /**/ 1, 0,
            +1, -1, -1, /**/0, 0, -1, /**/ 1, 1,

            //hinten
            -1, -1, +1, /**/0, 0, +1, /**/ 1, 0,
            -1, +1, +1, /**/0, 0, +1, /**/ 1, 1,
            +1, +1, +1, /**/0, 0, +1, /**/ 0, 1,
            +1, -1, +1, /**/0, 0, +1, /**/ 0, 0,

            //links
            -1, -1, -1, /**/-1, 0, 0, /**/ 1, 0, //8
            -1, -1, +1, /**/-1, 0, 0, /**/ 0, 0, //9
            -1, +1, -1, /**/-1, 0, 0, /**/ 1, 1, //10
            -1, +1, +1, /**/-1, 0, 0, /**/ 0, 1, //11

            //rechts
            +1, -1, -1, /**/+1, 0, 0, /**/ 0, 0, //12
            +1, -1, +1, /**/+1, 0, 0, /**/ 1, 0, //13
            +1, +1, -1, /**/+1, 0, 0, /**/ 0, 1, //14
            +1, +1, +1, /**/+1, 0, 0, /**/ 1, 1, //15

            //oben
            +1, +1, +1, /**/0, +1, 0, /**/ 1, 1, //16
            +1, +1, -1, /**/0, +1, 0, /**/ 1, 0, //17
            -1, +1, +1, /**/0, +1, 0, /**/ 0, 1, //18
            -1, +1, -1, /**/0, +1, 0, /**/ 0, 0, //19

            //unten
            +1, -1, +1, /**/0, -1, 0, /**/ 0, 1, //20
            +1, -1, -1, /**/0, -1, 0, /**/ 0, 0, //21
            -1, -1, +1, /**/0, -1, 0, /**/ 1, 1, //22
            -1, -1, -1, /**/0, -1, 0, /**/ 1, 0, //23
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

        m_sFrustumGeomety->SetIndexBuffer(indices, vertexCount + 6, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        m_sFrustumGeomety->SetVertexBuffer(vertices, vertexCount, 4 * vertexStride);
        m_sFrustumGeomety->GetVertexBuffer()->SetCPUAccess(D3D11_CPU_ACCESS_WRITE);
        m_sFrustumGeomety->GetVertexBuffer()->SetUsage(D3D11_USAGE_DYNAMIC);
        m_sFrustumGeomety->VCreate();
        SAFE_ARRAY_DELETE(vertices);
        SAFE_ARRAY_DELETE(indices);

        return m_sFrustumGeomety;
    }

    d3d::Geometry* _CreateScreenQuad(D3D11_CPU_ACCESS_FLAG flags)
    {
        d3d::Geometry* geo = new d3d::Geometry();

        CONST UINT stride = 5;
        CONST UINT count = 4;
        FLOAT* vertices = new FLOAT[stride * count];
        UINT* indices = new UINT[count];
        //3,2,0,1
        indices[0] = 0; indices[1] = 1; indices[2] = 2; indices[3] = 3;

        FLOAT localVertices[count * stride] = {
            -1, -1, 0, 0, 1,
            +1, -1, 0, 1, 1,
            -1, +1, 0, 0, 0,
            +1, +1, 0, 1, 0,
        };

        for(UINT i = 0; i < count * stride; ++i)
        {
            vertices[i] = localVertices[i];
        }

        geo->SetIndexBuffer(indices, count, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        geo->SetVertexBuffer(vertices, count, stride * sizeof(FLOAT));
        geo->GetVertexBuffer()->SetCPUAccess(flags);
        if(flags & D3D11_CPU_ACCESS_WRITE)
        {
            geo->GetVertexBuffer()->SetUsage(D3D11_USAGE_DYNAMIC);
        }
        geo->VCreate();
        delete[] indices;
        delete[] vertices;
        return geo;
    }

    d3d::Geometry* GeometryFactory::GetGlobalScreenQuad(VOID) 
    {
        if(!m_sScreenQuad)
        {
            m_sScreenQuad = CreateScreenQuad();
        }
        return m_sScreenQuad;
    }

    d3d::Geometry* GeometryFactory::GetGlobalScreenQuadCPU(VOID)
    {
        if(!m_sScreenQuadCPU)
        {
            m_sScreenQuadCPU = CreateScreenQuadCPUAcc();
        }
        return m_sScreenQuadCPU;
    }

    d3d::Geometry* GeometryFactory::CreateScreenQuad(VOID)
    {
        return _CreateScreenQuad((D3D11_CPU_ACCESS_FLAG)0);
    }

    d3d::Geometry* GeometryFactory::CreateScreenQuadCPUAcc(VOID)
    {
        return _CreateScreenQuad(D3D11_CPU_ACCESS_WRITE);
    }

    d3d::Geometry* GeometryFactory::GetGlobalDefShadingCube(VOID)
    {
        if(m_sDefCube)
        {
            return m_sDefCube;
        }
        m_sDefCube = new d3d::Geometry();

        CONST UINT vertexCount = 24;
        CONST UINT vertexStride = 8;
        //pos,norm,color,tex
        FLOAT* vertices = new FLOAT[vertexStride * vertexCount];
        UINT* indices = new UINT[vertexCount + 6];

        FLOAT vertexTmp[vertexStride * vertexCount] = 
        {
            //vorne
            -1, -1, -1, /**/0, 0, -1, /**/ 0, 1,
            -1, +1, -1, /**/0, 0, -1, /**/ 0, 0,
            +1, +1, -1, /**/0, 0, -1, /**/ 1, 0,
            +1, -1, -1, /**/0, 0, -1, /**/ 1, 1,

            //hinten
            -1, -1, +1, /**/0, 0, +1, /**/ 1, 0,
            -1, +1, +1, /**/0, 0, +1, /**/ 1, 1,
            +1, +1, +1, /**/0, 0, +1, /**/ 0, 1,
            +1, -1, +1, /**/0, 0, +1, /**/ 0, 0,

            //links
            -1, -1, -1, /**/-1, 0, 0, /**/ 1, 0, //8
            -1, -1, +1, /**/-1, 0, 0, /**/ 0, 0, //9
            -1, +1, -1, /**/-1, 0, 0, /**/ 1, 1, //10
            -1, +1, +1, /**/-1, 0, 0, /**/ 0, 1, //11

            //rechts
            +1, -1, -1, /**/+1, 0, 0, /**/ 0, 0, //12
            +1, -1, +1, /**/+1, 0, 0, /**/ 1, 0, //13
            +1, +1, -1, /**/+1, 0, 0, /**/ 0, 1, //14
            +1, +1, +1, /**/+1, 0, 0, /**/ 1, 1, //15

            //oben
            +1, +1, +1, /**/0, +1, 0, /**/ 1, 1, //16
            +1, +1, -1, /**/0, +1, 0, /**/ 1, 0, //17
            -1, +1, +1, /**/0, +1, 0, /**/ 0, 1, //18
            -1, +1, -1, /**/0, +1, 0, /**/ 0, 0, //19

            //unten
            +1, -1, +1, /**/0, -1, 0, /**/ 0, 1, //20
            +1, -1, -1, /**/0, -1, 0, /**/ 0, 0, //21
            -1, -1, +1, /**/0, -1, 0, /**/ 1, 1, //22
            -1, -1, -1, /**/0, -1, 0, /**/ 1, 0, //23
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

        m_sDefCube->SetIndexBuffer(indices, vertexCount + 6, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        m_sDefCube->SetVertexBuffer(vertices, vertexCount, 4 * vertexStride);
        m_sDefCube->VCreate();
        SAFE_ARRAY_DELETE(vertices);
        SAFE_ARRAY_DELETE(indices);
        return m_sDefCube;
    }

    d3d::Geometry* GeometryFactory::GetGlobalBoundingBoxCube(VOID)
    {
        if(m_sCube)
        {
            return m_sCube;
        }
        m_sCube = new d3d::Geometry();

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

        m_sCube->SetIndexBuffer(indices, vertexCount + 6, D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        m_sCube->SetVertexBuffer(vertices, vertexCount, vertexStride);
        m_sCube->VCreate();
        SAFE_ARRAY_DELETE(vertices);
        SAFE_ARRAY_DELETE(indices);
        return m_sCube;
    }

    FLOAT* GeometryFactory::GetSphereVertexBuffer(UINT segmentsX, UINT segmentsY)
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

    d3d::Geometry* GeometryFactory::CreateSphere(UINT segmentsX, UINT segmentsY, BOOL deleteRawData)
    {
        /*UINT segmentsX = 32;
        UINT segmentsY = 16; */
        UINT indexCount = segmentsY * 2 * (segmentsX + 1) + segmentsY;
        UINT vertexCount = (segmentsX + 1) * (segmentsY + 1);

        UINT* indexBuffer = new UINT[indexCount];
        FLOAT* vertexBuffer = GetSphereVertexBuffer(segmentsX, segmentsY);//new FLOAT[vertexCount * 8];
        UINT ic = 0;
        /*
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
        } */

        for(UINT i = 0; i < segmentsY; ++i)
        {
            for(UINT j = 0; j <= segmentsX; ++j)
            {
                indexBuffer[ic++] = i * (segmentsX+1) + j + segmentsX + 1;
                indexBuffer[ic++] = i * (segmentsX+1) + j;
            }
            indexBuffer[ic++] = -1;
        }

        d3d::Geometry* geo = new d3d::Geometry(TRUE);
        geo->SetIndexBuffer(indexBuffer, indexCount, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        geo->SetVertexBuffer(vertexBuffer, vertexCount, 8 * sizeof(FLOAT));
        geo->VCreate();

        if(deleteRawData)
        {
            geo->DeleteRawData();
        }

        return geo;
    }

    d3d::Geometry* GeometryFactory::GetSphere(VOID)
    {
        if(m_sSphere)
        {
            return m_sSphere;
        }
        m_sSphere = CreateSphere(32, 16);
        return m_sSphere;
    }

    d3d::Geometry* GeometryFactory::GetSkyDome(VOID)
    {
        if(m_sSkyDome)
        {
            return m_sSkyDome;
        }
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

        m_sSkyDome = new d3d::Geometry();
        m_sSkyDome->SetIndexBuffer(indexBuffer, indexCount, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        m_sSkyDome->SetVertexBuffer(vertexBuffer, vertexCount, 5 * sizeof(FLOAT));
        m_sSkyDome->VCreate();

        delete[] indexBuffer;
        delete[] vertexBuffer;

        return m_sSkyDome;
    }

    d3d::Geometry* GeometryFactory::CreateNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale, BOOL deleteRawData)
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

        /*TBD_FOR_INT(ic)
        {
            DEBUG_OUT_A("%d ", indexBuffer[i]);
        } */
        
        d3d::Geometry* geo = new d3d::Geometry();
        geo->SetIndexBuffer(indexBuffer, indexCount, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        geo->SetVertexBuffer(vertexBuffer, vertexCount, vertexStride * sizeof(FLOAT));
        geo->VCreate();

        if(deleteRawData)
        {
            SAFE_ARRAY_DELETE(vertexBuffer);
            SAFE_ARRAY_DELETE(indexBuffer);
        }

        return geo;
    }

    d3d::Geometry* GeometryFactory::CreateAlternatingNormedGrid(UINT xtiles, UINT ztiles, FLOAT scale, BOOL deleteRawData)
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
                /*util::Vec3 v(x,0,y);
                v.Print(); */
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


        /*TBD_FOR_INT(ic)
        {
            DEBUG_OUT_A("%d ", indexBuffer[i]);
        } */
        
        d3d::Geometry* geo = new d3d::Geometry();
        geo->SetIndexBuffer(indexBuffer, indexCount, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        geo->SetVertexBuffer(vertexBuffer, vertexCount, vertexStride * sizeof(FLOAT));
        geo->VCreate();

        if(deleteRawData)
        {
            SAFE_ARRAY_DELETE(vertexBuffer);
            SAFE_ARRAY_DELETE(indexBuffer);
        }

        return geo;
    }

    VOID GeometryFactory::Destroy(VOID)
    {
        if(m_sSkyDome)
        {
            m_sSkyDome->VDestroy();
            delete m_sSkyDome;
        }
        if(m_sScreenQuad)
        {
            m_sScreenQuad->VDestroy();
            delete m_sScreenQuad;
        }
        if(m_sSphere)
        {
            m_sSphere->VDestroy();
            delete m_sSphere;
        }
        if(m_sCube)
        {
            m_sCube->VDestroy();
            delete m_sCube;
        }
        if(m_sDefCube)
        {
            m_sDefCube->VDestroy();
            delete m_sDefCube;
        }
        if(m_sFrustumGeomety)
        {
            m_sFrustumGeomety->VDestroy();
            delete m_sFrustumGeomety;
        }
        if(m_sScreenQuadCPU)
        {
            m_sScreenQuadCPU->VDestroy();
            delete m_sScreenQuadCPU;
        }
        if(m_sLine)
        {
            m_sLine->VDestroy();
            delete m_sLine;
        }
    }

    GeometryFactory::~GeometryFactory(VOID) {
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
    }
