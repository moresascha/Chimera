#include "ResourceLoader.h"
#include "Material.h"
#include "Mesh.h"
#include "util.h"
#include "D3DGeometry.h"
#include "D3DTexture.h"

namespace chimera
{
    struct SubMeshData 
    {
        std::list<Face> m_faces;
        UINT m_triplesCount;
        UINT m_trianglesCount;
        SubMeshData() : m_triplesCount(0), m_trianglesCount(0) {}
    };

    struct RawGeometryData 
    {
        std::vector<util::Vec3> m_positions;
        std::vector<util::Vec3> m_normals;
        std::vector<util::Vec3> m_texCoords;
        std::vector<util::Vec3> m_tangents;
    };

    VOID GetLine(CHAR* source, UINT& pos, UINT size)
    {
        while(source[pos] != '\n' && pos < size)
        {
            pos++;
        }
        pos++;
    }

    struct ShrdCharPtr
    {
        UINT m_length;
        CHAR* m_ptr;

        CHAR m_buffer[64];

        ShrdCharPtr(VOID) : m_ptr(NULL), m_length(0)
        {
            ZeroMemory(&m_buffer, sizeof(CHAR) * 64);
        }

        std::string GetString(VOID)
        {
            return std::string(m_ptr, m_length);
        }

        FLOAT GetFloat(VOID)
        {
            _memccpy(&m_buffer, m_ptr, '\0', m_length);
            return (FLOAT)atof(m_buffer);
        }

        INT GetInt(VOID)
        {
            _memccpy(&m_buffer, m_ptr, '\0', m_length);
            return atoi(m_buffer);
        }
    };

    BOOL ShrdPtrStrCmp(ShrdCharPtr& s0, CONST CHAR* str)
    {
        size_t l = std::strlen(str);
        if(s0.m_length != l)
        {
            return FALSE;
        }

        TBD_FOR_INT(s0.m_length)
        {
            if(s0.m_ptr[i] != str[i])
            {
                return FALSE;
            }
        }
        return TRUE;
    }

    class SharedMemoryStream
    {
    public:
        UINT m_start;
        UINT m_length;
        UINT m_end;
        CHAR* m_source;
        CHAR m_delim;

        SharedMemoryStream(UINT s, UINT l, CHAR* src, CHAR delim = ' ') : m_start(s), m_length(l), m_source(src), m_delim(delim)
        {
            m_end = m_start + m_length;
        }

        BOOL Good(VOID)
        {
            return m_start < m_end;
        }

        ShrdCharPtr& operator >> (ShrdCharPtr& ptr)
        {
            ptr.m_ptr = (m_source + m_start);
            if(!Good())
            {
                LOG_CRITICAL_ERROR("Erf wtf is going on!");
            }
            INT l = 0;
            CHAR c = m_source[m_start];
            while(Good() && c != m_delim && c != '\n' && c != '\0')
            {
                m_start++;
                c = m_source[m_start];
                l++;
            }
            ptr.m_length = l;
            m_start++;
            return ptr;
        }
    };

    UINT GetTriple(CHAR* s, Triple& triple) 
    {
        SharedMemoryStream ss(0, INFINITE, s, '/');
        ShrdCharPtr f[3];
        ss >> f[0];
        ss >> f[1];
        ss >> f[2];
        triple.position = f[0].GetInt() - 1;
        triple.texCoord = f[1].GetInt() - 1;
        triple.normal = f[2].GetInt() - 1;
        return 3;
    }

    INT ObjLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle) 
    {
        std::shared_ptr<chimera::Mesh> mesh = std::static_pointer_cast<chimera::Mesh>(handle);

        SubMeshData subMeshData;
        RawGeometryData rawData;
        std::string matName;
        UINT lastIndexStart = 0;
        UINT pos = 0;

        ShrdCharPtr flag;
        ShrdCharPtr x, y, z;
        ShrdCharPtr f[4];

        std::locale loc;
        CONST std::collate<CHAR>& coll = std::use_facet<std::collate<CHAR>>(loc);

        while(pos < size)//sss.good()) 
        {
            //std::getline(sss, line);
            INT start = pos;
            GetLine(source, pos, size);

            SharedMemoryStream ss(start, pos - start - 1, source);

            ss >> flag;
            if(ShrdPtrStrCmp(flag, "mtllib"))
            {
                ShrdCharPtr mtllib;
                ss >> mtllib;
                mesh->m_materials = mtllib.GetString();
            }
            if(ShrdPtrStrCmp(flag, "usemtl"))
            {
                if(subMeshData.m_triplesCount == 0) 
                {
                    matName.clear();
                    ShrdCharPtr mat;
                    ss >> mat;
                    matName = mat.GetString();
                    continue;
                }
                std::shared_ptr<MaterialSet> materials = std::static_pointer_cast<MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(mesh->VGetMaterials()));
                mesh->VAddIndexBufferInterval(lastIndexStart, subMeshData.m_trianglesCount * 3, materials->GetMaterialIndex(matName));
                lastIndexStart = subMeshData.m_trianglesCount * 3;
                matName.clear();
                ShrdCharPtr mat;
                ss >> mat;
                matName = mat.GetString();
            }
            if(ShrdPtrStrCmp(flag, "v"))
            {
                util::Vec3 vertex;
                ss >> x;
                ss >> y;
                ss >> z;

                vertex.Set(x.GetFloat(), y.GetFloat(), z.GetFloat());
                rawData.m_positions.push_back(vertex);
            }
            else if(ShrdPtrStrCmp(flag, "vt"))
            {
                util::Vec3 texCoord;
                ss >> x;
                ss >> y;
                texCoord.Set(x.GetFloat(), y.GetFloat(), 0);
                rawData.m_texCoords.push_back(texCoord);
            } 
            else if(ShrdPtrStrCmp(flag, "vn"))
            {
                util::Vec3 normal;
                ss >> x;
                ss >> y;
                ss >> z;
                normal.Set(x.GetFloat(), y.GetFloat(), z.GetFloat());
                rawData.m_normals.push_back(normal);
            }
            else if(ShrdPtrStrCmp(flag, "f"))
            {
                UINT triplesCount = 0;
                CHAR* s = source + ss.m_start;
                CHAR* e = source + ss.m_end;
                while(ss.Good())
                {
                    ss >> f[triplesCount];
                    triplesCount++;
                }
                Face face;
                for(UINT i = 0; i < triplesCount; ++i)
                {
                    Triple triple;
                    GetTriple(f[i].m_ptr, triple); 
                    triple.hash = (2729 * triple.position + triple.normal) * 3572 + triple.texCoord;
                    face.m_triples.push_back(triple);
                }
                subMeshData.m_triplesCount += triplesCount;
                subMeshData.m_trianglesCount += triplesCount / 2;
                subMeshData.m_faces.push_back(face);
                mesh->m_faces.push_back(face);
            }
        }

        //subMeshData.m_faces.resize(fc);

        std::shared_ptr<chimera::MaterialSet> materials = std::static_pointer_cast<chimera::MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(mesh->VGetMaterials()));
        mesh->VAddIndexBufferInterval(lastIndexStart, subMeshData.m_trianglesCount * 3, materials->GetMaterialIndex(matName));

        std::vector<UINT> lIndices;
        lIndices.reserve(mesh->VGetFaces().size() * 4);

        std::vector<FLOAT> lVertices;
        lVertices.reserve(rawData.m_positions.size() * 3);

        std::map<LONG, Triple> hashToTriple;

        util::Vec3 zero;

        for(auto it = subMeshData.m_faces.begin(); it != subMeshData.m_faces.end(); ++it)
        {
            /*
            util::Vec3 tangents[4];
            if(it->m_triples.size() == 4)
            {
                Triple t0 = it->m_triples[0];
                Triple t1 = it->m_triples[1];
                Triple t2 = it->m_triples[2];
                Triple t3 = it->m_triples[3];
                util::Vec3& p0 = rawData.m_positions[t0.position];
                util::Vec3& p1 = rawData.m_positions[t1.position];
                util::Vec3& p2 = rawData.m_positions[t2.position];
                util::Vec3& p3 = rawData.m_positions[t3.position];
                util::Vec3& tangent = p2 - p0 + p1 - p0 + p3 - p2 + p3 - p1;
                tangent.Normalize();
                tangents[0] = tangent;//(p1 - p0 + p2 - p0).Normalize();
                tangents[1] = tangent;//(p0 - p1 + p3 - p1).Normalize();
                tangents[2] = tangent;//(p0 - p2 + p3 - p2).Normalize();
                tangents[3] = tangent;//(p1 - p3 + p2 - p3).Normalize();
                p0.Print();
                p1.Print();
                p2.Print();
                p3.Print();
                tangent.Print(); 
                DEBUG_OUT("--");
            }
            else if(it->m_triples.size() == 3)
            {
                Triple t0 = it->m_triples[0];
                Triple t1 = it->m_triples[1];
                Triple t2 = it->m_triples[2];
                util::Vec3& p0 = rawData.m_positions[t0.position];
                util::Vec3& p1 = rawData.m_positions[t1.position];
                util::Vec3& p2 = rawData.m_positions[t2.position];
                util::Vec3& tangent = p2 - p0 + p1 - p0;
                tangent.Normalize();
                tangents[0] = tangent;
                tangents[1] = tangent;
                tangents[2] = tangent;
            }
            else
            {
                LOG_ERROR("unkown triples count");
            } */
            for(UINT i = 0; i < it->m_triples.size(); ++i)
            {
                Triple t = it->m_triples[i];

                auto it = hashToTriple.find(t.hash);

                if(it == hashToTriple.end())
                {
                    util::Vec3& pos0 = rawData.m_positions[t.position];

                    mesh->m_aabb.AddPoint(pos0);

                    util::Vec3* norm0 = &zero;
                    if(!rawData.m_normals.empty())
                    {
                        norm0 = &rawData.m_normals[t.normal];
                    }
                    
                    util::Vec3* texCoord0 = &zero;
                    if(!rawData.m_texCoords.empty())
                    {
                        texCoord0 = &rawData.m_texCoords[t.texCoord];
                    }

                    t.index = (UINT)lVertices.size() / 8;
                    
                    lVertices.push_back(pos0.x);
                    lVertices.push_back(pos0.y);
                    lVertices.push_back(pos0.z);

                    lVertices.push_back(norm0->x);
                    lVertices.push_back(norm0->y);
                    lVertices.push_back(norm0->z);

                    lVertices.push_back(texCoord0->x);
                    lVertices.push_back(texCoord0->y); //Note: textures used by obj.files are y-inverted, might change in the future since I use old textures
                    /*
                    std::stringstream ss;
                    ss << "Created triple= p=" << t.position << ", tx=" << t.texCoord << ", n=" << t.normal;
                    LOG_INFO(ss.str()); */
                    /*lVertices.push_back(tangents[i].x);
                    lVertices.push_back(tangents[i].y);
                    lVertices.push_back(tangents[i].z); */
                    
                    hashToTriple.insert(std::pair<LONG, Triple>(t.hash, t));
                }
            }

            /*2,0,1,2,0,3*/
            auto i0 = hashToTriple.find(it->m_triples[0].hash);
            auto i1 = hashToTriple.find(it->m_triples[1].hash);
            auto i2 = hashToTriple.find(it->m_triples[2].hash);

            __int64 index0 = i0->second.index;
            __int64 index1 = i1->second.index;
            __int64 index2 = i2->second.index;

            lIndices.push_back((UINT)index2);

            lIndices.push_back((UINT)index1);

            lIndices.push_back((UINT)index0);

            if(it->m_triples.size() == 4)
            {
                auto i3 = hashToTriple.find(it->m_triples[3].hash);;
                __int64 index3 = i3->second.index;
                lIndices.push_back((UINT)index2);

                lIndices.push_back((UINT)index0);

                lIndices.push_back((UINT)index3);
            }
        }
        UINT vertexSize = 8;
        UINT indexCount = (UINT)lIndices.size();
        UINT vertexCount = (UINT)lVertices.size() / vertexSize;

        FLOAT* vertices = new FLOAT[lVertices.size()];
        UINT* indices = new UINT[indexCount];
        UINT stride = vertexSize * sizeof(FLOAT);

        for(UINT i = 0; i < lVertices.size(); ++i)
        {
            vertices[i] = lVertices[i];
        }

        for(UINT i = 0; i < indexCount; ++i)
        {
            indices[i] = lIndices[i];
        }

        mesh->VSetIndices(indices, indexCount);
        mesh->VSetVertices(vertices, vertexCount, stride);
        mesh->VGetAABB().Construct();

        delete[] source;
        return indexCount * sizeof(UINT) + vertexCount * stride * sizeof(FLOAT);
    }
	
    INT MaterialLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle)
    {
        std::shared_ptr<chimera::MaterialSet> materials = std::static_pointer_cast<chimera::MaterialSet>(handle);

        std::stringstream ss;
        std::string desc(source);
        ss << desc;
        std::string line;
        std::shared_ptr<chimera::Material> current = NULL;
        UINT index = 0;

        while(ss.good())
        {
            std::getline(ss, line);
            std::stringstream streamLine;
            streamLine << line;

            std::string prefix;
            streamLine >> prefix;

            if(prefix == "newmtl")
            {
                current = std::shared_ptr<chimera::Material>(new chimera::Material());
                std::string name;
                streamLine >> name;
                materials->AddMaterial(current, name, index);
                index++;
            }
            else if(prefix == "Ns")
            {
                std::string ns;
                streamLine >> ns;
                DOUBLE d = atof(ns.c_str());
                current->m_specCoef = (FLOAT)d; 
            }
            else if(prefix == "Ka")
            {
                std::string val;
                streamLine >> val;
                DOUBLE r = atof(val.c_str());
                streamLine >> val;
                DOUBLE g = atof(val.c_str());
                streamLine >> val;
                DOUBLE b = atof(val.c_str());
                current->m_ambient.Set((FLOAT)r, (FLOAT)g, (FLOAT)b, 1.f);
            }
            else if(prefix == "Kd")
            {
                std::string val;
                streamLine >> val;
                DOUBLE r = atof(val.c_str());
                streamLine >> val;
                DOUBLE g = atof(val.c_str());
                streamLine >> val;
                DOUBLE b = atof(val.c_str());
                current->m_diffuse.Set((FLOAT)r, (FLOAT)g, (FLOAT)b, 1.f);
            }
            else if(prefix == "Ks")
            {
                std::string val;
                streamLine >> val;
                DOUBLE r = atof(val.c_str());
                streamLine >> val;
                DOUBLE g = atof(val.c_str());
                streamLine >> val;
                DOUBLE b = atof(val.c_str());
                current->m_specular.Set((FLOAT)r, (FLOAT)g, (FLOAT)b, 1.f);
            }
            else if(prefix == "map_Kd")
            {
                std::string val;
                streamLine >> val;
                current->m_textureDiffuse = chimera::CMResource(val);
                materials->VGetResourceCache()->VGetHandle(current->m_textureDiffuse); //preload texture
            }
            else if(prefix == "map_Kn")
            {
                std::string val;
                streamLine >> val;
                current->m_hasNormal = TRUE;
                current->m_textureNormal = chimera::CMResource(val);
                materials->VGetResourceCache()->VGetHandle(current->m_textureNormal); //preload texture
            }
            else if(prefix == "illum")
            {
                std::string val;
                streamLine >> val;
                DOUBLE illum = atof(val.c_str());
                current->m_reflectance = (FLOAT)illum;
            }
            else if(prefix == "scale")
            {
                std::string val;
                streamLine >> val;
                DOUBLE scale = atof(val.c_str());
                current->m_texScale = (FLOAT)scale;
            }
        }
        delete[] source;
        return size;
    }

    INT ImageLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle) 
    {
        Gdiplus::Bitmap* b = util::GetBitmapFromBytes(source, size);

        std::unique_ptr<ImageExtraData> extraData(new ImageExtraData(b->GetWidth(), b->GetHeight(), b->GetPixelFormat()));

        handle->VSetExtraData(std::move(extraData));
        
        handle->VSetBuffer(util::GetTextureData(b));

        size = b->GetWidth() * b->GetWidth() * 4;

        delete b;
        delete[] source;
        
        return size;
    }


    std::unique_ptr<IResHandle> ObjLoader::VCreateHandle(VOID) 
    { 
        return std::unique_ptr<Mesh>(new Mesh()); 
    }

    std::unique_ptr<IResHandle>  MaterialLoader::VCreateHandle(VOID) 
    { 
        return std::unique_ptr<MaterialSet>(new MaterialSet()); 
    }

    BOOL WaveLoader::ParseWaveFile(CHAR* wavStream, std::shared_ptr<IResHandle> handle, UINT& size)
    {
        WaveSoundExtraDatra* data = (WaveSoundExtraDatra*)(handle->VGetExtraData());

        DWORD file = 0;
        DWORD fileEnd = 0;
        DWORD length = 0;
        DWORD type = 0;
        DWORD pos = 0;

        type = *((DWORD*)(wavStream+pos));
        pos += sizeof(DWORD);

        if(type != mmioFOURCC('R', 'I', 'F', 'F'))
        {
            return FALSE;
        }

        length = *((DWORD*)(wavStream+pos));
        pos += sizeof(DWORD);

        type = *((DWORD*)(wavStream+pos));
        pos += sizeof(DWORD);

        if(type != mmioFOURCC('W', 'A', 'V', 'E'))
        {
            return FALSE;
        }

        fileEnd = length - 4;

        ZeroMemory(&data->m_format, sizeof(WAVEFORMATEX));

        BOOL bufferCopied = FALSE;

        while(file < fileEnd)
        {
            type = *((DWORD*)(wavStream+pos));
            pos += sizeof(DWORD);
            file += sizeof(DWORD);

            length = *((DWORD*)(wavStream+pos));
            pos += sizeof(DWORD);
            file += sizeof(DWORD);

            /*std::stringstream ss;
            ss << (CHAR)((type >> 0) &  0xFF);
            ss << (CHAR)((type >> 8) &  0xFF);
            ss << (CHAR)((type >> 16) &  0xFF);
            ss << (CHAR)((type >> 24) &  0xFF);
            ss << ", l=";
            ss << length;
            DEBUG_OUT(ss.str()); */

            switch(type)
            {
            case mmioFOURCC('I', 'N', 'F', 'O') :
                {
                    LOG_CRITICAL_ERROR("INFO CHUNK creates errors");
                } break;
            case mmioFOURCC('f', 'a', 'c', 't') :
                {
                    LOG_CRITICAL_ERROR("compressed wave file not supported");
                } break;
            case mmioFOURCC('f', 'm', 't', ' ') :
                {
                    memcpy(&data->m_format, wavStream+pos, length);
                    pos += length;
                    data->m_format.cbSize = (WORD)length;
                } break;
            case mmioFOURCC('d', 'a', 't', 'a') : 
                {
                    bufferCopied = TRUE;
                    /*if(length != size)
                    {
                        LOG_ERROR("strange");
                    } */
                    CHAR* buffer = new CHAR[length];
                    memcpy(buffer, wavStream+pos, length);
                    size = length;
                    handle->VSetBuffer(buffer);
                    pos += length;
                } break;
            }

            file += length;

            if(bufferCopied)
            {
                data->m_lengthMillis = (size * 1000) / data-> m_format.nAvgBytesPerSec;
                
                SAFE_ARRAY_DELETE(wavStream);

                return TRUE;
            }

            if(length & 1)
            {
                ++pos;
                ++file;
            }
        }
        return FALSE;
    }

    INT WaveLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<IResHandle> handle)
    {
        //DefaultRessourceLoader::VLoadRessource(source, size, handle);

        std::unique_ptr<WaveSoundExtraDatra> data(new WaveSoundExtraDatra());
        handle->VSetExtraData(std::move(data));

        if(!ParseWaveFile(source, handle, size))
        {
            LOG_CRITICAL_ERROR("invalid wave format");
        }

        return size;
    }

    //vram

    IVRamHandle* GeometryCreator::VGetHandle(VOID)
    {
        return new chimera::d3d::Geometry();
    }

    VOID GeometryCreator::VCreateHandle(IVRamHandle* handle)
    {
        IGeometry* geo = (IGeometry*)handle;
        std::shared_ptr<Mesh> mesh = std::static_pointer_cast<Mesh>(CmGetApp()->VGetCache()->VGetHandle(CMResource(handle->VGetResource())));
        geo->VSetIndexBuffer(mesh->VGetIndices(), mesh->VGetIndexCount());
        geo->VSetVertexBuffer(mesh->VGetVertices(), mesh->VGetVertexCount(), mesh->VGetVertexStride());
        geo->VSetTopology(eTopo_Triangles);
    }

    IVRamHandle* TextureCreator::VGetHandle(VOID)
    {
        return new chimera::d3d::Texture2D();
    }

    VOID TextureCreator::VCreateHandle(IVRamHandle* handle)
    {
        chimera::d3d::Texture2D* texture = (chimera::d3d::Texture2D*)handle;
        CMResource res = handle->VGetResource();
        std::shared_ptr<IResHandle> xtraHandle = CmGetApp()->VGetCache()->VGetHandle(res);
        ImageExtraData* data = (ImageExtraData*)(xtraHandle->VGetExtraData());

        texture->SetBindflags(D3D11_BIND_SHADER_RESOURCE);
        //texture->GetDescription().BindFlags = D3D11_BIND_SHADER_RESOURCE;

        //texture->GetDescription().Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture->SetFormat(DXGI_FORMAT_R8G8B8A8_UNORM);

        //texture->GetDescription().Width = data->m_width;
        texture->SetWidth(data->m_width);

        //texture->GetDescription().Height = data->m_height;
        texture->SetHeight(data->m_height);

        //texture->GetDescription().MipLevels = 0;
        texture->SetMipMapLevels(0);
    
        //texture->GetDescription().SampleDesc.Count = 1;
        texture->SetSamplerCount(1);
    
        //texture->GetDescription().SampleDesc.Quality = 0;
        texture->SetSamplerQuality(0);

        //texture->GetDescription().ArraySize = 1;
        texture->SetArraySize(1);

        //texture->GetDescription().MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
        texture->SetMicsFlags(D3D11_RESOURCE_MISC_GENERATE_MIPS);

        texture->SetData(xtraHandle->VBuffer());
    }
}