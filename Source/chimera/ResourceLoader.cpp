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
        uint m_triplesCount;
        uint m_trianglesCount;
        SubMeshData() : m_triplesCount(0), m_trianglesCount(0) {}
    };

    struct RawGeometryData 
    {
        std::vector<util::Vec3> m_positions;
        std::vector<util::Vec3> m_normals;
        std::vector<util::Vec3> m_texCoords;
        std::vector<util::Vec3> m_tangents;
    };

    void GetLine(char* source, uint& pos, uint size)
    {
        while(source[pos] != '\n' && pos < size)
        {
            pos++;
        }
        pos++;
    }

    struct ShrdCharPtr
    {
        uint m_length;
        char* m_ptr;

        char m_buffer[64];

        ShrdCharPtr(void) : m_ptr(NULL), m_length(0)
        {
            ZeroMemory(&m_buffer, sizeof(char) * 64);
        }

        std::string GetString(void)
        {
            return std::string(m_ptr, m_length);
        }

        float GetFloat(void)
        {
            _memccpy(&m_buffer, m_ptr, '\0', m_length);
            return (float)atof(m_buffer);
        }

        int GetInt(void)
        {
            _memccpy(&m_buffer, m_ptr, '\0', m_length);
            return atoi(m_buffer);
        }
    };

    bool ShrdPtrStrCmp(ShrdCharPtr& s0, const char* str)
    {
        size_t l = std::strlen(str);
        if(s0.m_length != l)
        {
            return false;
        }

        TBD_FOR_INT(s0.m_length)
        {
            if(s0.m_ptr[i] != str[i])
            {
                return false;
            }
        }
        return true;
    }

    class SharedMemoryStream
    {
    public:
        uint m_start;
        uint m_length;
        uint m_end;
        char* m_source;
        char m_delim;

        SharedMemoryStream(uint s, uint l, char* src, char delim = ' ') : m_start(s), m_length(l), m_source(src), m_delim(delim)
        {
            m_end = m_start + m_length;
        }

        bool Good(void)
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
            int l = 0;
            char c = m_source[m_start];
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

    uint GetTriple(char* s, Triple& triple) 
    {
        SharedMemoryStream ss(0, INFINITE, s, '/');
        ShrdCharPtr f[3];
        ss >> f[0];
        ss >> f[1];
        ss >> f[2];
        triple.position = max(0, f[0].GetInt() - 1);
        triple.texCoord = max(0, f[1].GetInt() - 1);
        triple.normal = max(0, f[2].GetInt() - 1);
        return 3;
    }

    void InitMesh(Mesh* currentMesh, RawGeometryData& rawData, uint lastIndexStart, SubMeshData& subMeshData, uint* _out_indexCount, uint* _out_vertexCount, GeometryTopology currentTopo, std::string& matName)
    {

        std::shared_ptr<chimera::MaterialSet> materials = std::static_pointer_cast<chimera::MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(currentMesh->VGetMaterials()));
        currentMesh->VAddIndexBufferInterval(lastIndexStart, subMeshData.m_trianglesCount * 3, materials->GetMaterialIndex(matName), currentTopo);

        std::vector<uint> lIndices;
        lIndices.reserve(currentMesh->VGetFaces().size() * 4);

        std::vector<float> lVertices;
        lVertices.reserve(rawData.m_positions.size() * 3);

        std::map<LONG, Triple> hashToTriple;

        util::Vec3 zero;

        for(auto it = subMeshData.m_faces.begin(); it != subMeshData.m_faces.end(); ++it)
        {
            for(uint i = 0; i < it->m_triples.size(); ++i)
            {
                Triple t = it->m_triples[i];

                auto it = hashToTriple.find(t.hash);

                if(it == hashToTriple.end())
                {
                    util::Vec3& pos0 = rawData.m_positions[t.position];

                    currentMesh->VGetAABB().AddPoint(pos0);

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

                    t.index = (uint)lVertices.size() / 8;

                    lVertices.push_back(pos0.x);
                    lVertices.push_back(pos0.y);
                    lVertices.push_back(pos0.z);

                    lVertices.push_back(norm0->x);
                    lVertices.push_back(norm0->y);
                    lVertices.push_back(norm0->z);

                    lVertices.push_back(texCoord0->x);
                    lVertices.push_back(texCoord0->y);

                    hashToTriple.insert(std::pair<LONG, Triple>(t.hash, t));
                }
            }

            /*2,0,1,2,0,3*/

            auto i0 = hashToTriple.find(it->m_triples[0].hash);
            auto i1 = hashToTriple.find(it->m_triples[1].hash);

            __int64 index0 = i0->second.index;
            __int64 index1 = i1->second.index;

            if(it->m_triples.size() == 2)
            {
                lIndices.push_back((uint)index1);

                lIndices.push_back((uint)index0);
            }
            else
            {
                auto i2 = hashToTriple.find(it->m_triples[2].hash);

                __int64 index2 = i2->second.index;

                lIndices.push_back((uint)index2);

                lIndices.push_back((uint)index1);

                lIndices.push_back((uint)index0);

                if(it->m_triples.size() == 4)
                {
                    auto i3 = hashToTriple.find(it->m_triples[3].hash);;
                    __int64 index3 = i3->second.index;
                    lIndices.push_back((uint)index2);

                    lIndices.push_back((uint)index0);

                    lIndices.push_back((uint)index3);
                }
            }
        }
        uint vertexSize = 8;
        uint indexCount = (uint)lIndices.size();
        uint vertexCount = (uint)lVertices.size() / vertexSize;

        float* vertices = new float[lVertices.size()];
        uint* indices = new uint[indexCount];
        uint stride = vertexSize * sizeof(float);

        for(uint i = 0; i < lVertices.size(); ++i)
        {
            vertices[i] = lVertices[i];
        }

        for(uint i = 0; i < indexCount; ++i)
        {
            indices[i] = lIndices[i];
        }

        currentMesh->VSetIndices(indices, indexCount);
        currentMesh->VSetVertices(vertices, vertexCount, stride);
        currentMesh->VGetAABB().Construct();

        *_out_indexCount += indexCount;
        *_out_vertexCount += vertexCount;
    }

    int ObjLoader::VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle) 
    {
        //std::shared_ptr<Mesh> mesh = std::static_pointer_cast<Mesh>(handle);
        std::shared_ptr<MeshSet> meshSet = std::static_pointer_cast<MeshSet>(handle);

        SubMeshData subMeshData;
        RawGeometryData rawData;
        std::string matName;
        std::string currentMatLib;
        uint lastIndexStart = 0;
        uint pos = 0;
        uint indexCount = 0;
        uint vertexCount = 0;

        ShrdCharPtr flag;
        ShrdCharPtr x, y, z;
        ShrdCharPtr f[4];

        //std::locale loc;
        //CONST std::collate<CHAR>& coll = std::use_facet<std::collate<CHAR>>(loc);

        GeometryTopology currentTopo;
        Mesh* currentMesh = NULL;

        while(pos < size)
        {
            int start = pos;
            GetLine(source, pos, size);

            SharedMemoryStream ss(start, pos - start - 1, source);

            ss >> flag;
            if(ShrdPtrStrCmp(flag, "mtllib"))
            {
                ShrdCharPtr mtllib;
                ss >> mtllib;
                currentMatLib = mtllib.GetString();
            }
            else if(ShrdPtrStrCmp(flag, "o"))
            {
                Mesh* m = new Mesh();

                ShrdCharPtr objName;
                ss >> objName;

                m->m_materials = currentMatLib;

                CMResource res(objName.GetString());
                meshSet->m_meshes[res.m_name] = std::shared_ptr<IMesh>(m);

                if(currentMesh)
                {
                    InitMesh(currentMesh, rawData, lastIndexStart, subMeshData, &indexCount, &vertexCount, currentTopo, matName);

                    /*rawData.m_normals.clear();
                    rawData.m_positions.clear();
                    rawData.m_tangents.clear();
                    rawData.m_texCoords.clear();*/
                    subMeshData.m_faces.clear();
                    subMeshData.m_trianglesCount = 0;
                    subMeshData.m_triplesCount = 0;
                }

                lastIndexStart = 0;

                currentMesh = m;
            }
            else if(ShrdPtrStrCmp(flag, "usemtl"))
            {
                if(subMeshData.m_triplesCount == 0) 
                {
                    matName.clear();
                    ShrdCharPtr mat;
                    ss >> mat;
                    matName = mat.GetString();
                    continue;
                }
                std::shared_ptr<MaterialSet> materials = std::static_pointer_cast<MaterialSet>(CmGetApp()->VGetCache()->VGetHandle(currentMesh->VGetMaterials()));
                currentMesh->VAddIndexBufferInterval(lastIndexStart, subMeshData.m_trianglesCount * 3, materials->GetMaterialIndex(matName), currentTopo);
                lastIndexStart += subMeshData.m_trianglesCount * 3;
                matName.clear();
                ShrdCharPtr mat;
                ss >> mat;
                matName = mat.GetString();
            }
            else if(ShrdPtrStrCmp(flag, "v"))
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
            else if(ShrdPtrStrCmp(flag, "f") || ShrdPtrStrCmp(flag, "l"))
            {
                currentTopo = ShrdPtrStrCmp(flag, "f") ? eTopo_Triangles : eTopo_Lines;
                uint triplesCount = 0;
                char* s = source + ss.m_start;
                char* e = source + ss.m_end;
                while(ss.Good())
                {
                    ss >> f[triplesCount];
                    triplesCount++;
                }
                Face face;
                for(uint i = 0; i < triplesCount; ++i)
                {
                    Triple triple;
                    GetTriple(f[i].m_ptr, triple); 
                    triple.hash = (2729 * triple.position + triple.normal) * 3572 + triple.texCoord;
                    face.m_triples.push_back(triple);
                }
                subMeshData.m_triplesCount += triplesCount;
                subMeshData.m_trianglesCount += triplesCount / 2;
                subMeshData.m_faces.push_back(face);
                currentMesh->m_faces.push_back(face);
            }
        }

        uint stride = 8;
        if(currentMesh == NULL)
        {
            LOG_CRITICAL_ERROR_A("%s", "Can't parse obj file!");
        }
        InitMesh(currentMesh, rawData, lastIndexStart, subMeshData, &indexCount, &vertexCount, currentTopo, matName);

        delete[] source;
        return indexCount * sizeof(uint) + vertexCount * stride * sizeof(float);
    }
    
    int MaterialLoader::VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle)
    {
        std::shared_ptr<chimera::MaterialSet> materials = std::static_pointer_cast<chimera::MaterialSet>(handle);

        std::stringstream ss;
        std::string desc(source);
        ss << desc;
        std::string line;
        std::shared_ptr<chimera::Material> current = NULL;
        uint index = 0;

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
                current->m_specCoef = (float)d; 
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
                current->m_ambient.Set((float)r, (float)g, (float)b, 1.f);
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
                current->m_diffuse.Set((float)r, (float)g, (float)b, 1.f);
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
                current->m_specular.Set((float)r, (float)g, (float)b, 1.f);
            }
            else if(prefix == "map_Kd")
            {
                std::string val;
                streamLine >> val;
                if(materials->VGetResourceCache()->VGetFile().VHasFile("textures/"+val))
                {
                    current->m_textureDiffuse = chimera::CMResource(val);
                    materials->VGetResourceCache()->VGetHandle(current->m_textureDiffuse); //preload texture
                }
                else
                {
                    DEBUG_OUT_A("Texture '%s' not found.\n", val.c_str());
                }
            }
            else if(prefix == "map_Kn")
            {
                std::string val;
                streamLine >> val;

                if(materials->VGetResourceCache()->VGetFile().VHasFile("textures/"+val))
                {
                    current->m_hasNormal = true;
                    current->m_textureNormal = chimera::CMResource(val);
                    materials->VGetResourceCache()->VGetHandle(current->m_textureNormal); //preload texture
                }
            }
            else if(prefix == "illum")
            {
                std::string val;
                streamLine >> val;
                DOUBLE illum = atof(val.c_str());
                current->m_reflectance = (float)(illum == 1.0f);
            }
            else if(prefix == "scale")
            {
                std::string val;
                streamLine >> val;
                DOUBLE scale = atof(val.c_str());
                current->m_texScale = (float)scale;
            }
        }
        delete[] source;
        return size;
    }

    int ImageLoader::VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle) 
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


    std::unique_ptr<IResHandle> ObjLoader::VCreateHandle(void) 
    { 
        return std::unique_ptr<IMeshSet>(new MeshSet());
    }

    std::unique_ptr<IResHandle>  MaterialLoader::VCreateHandle(void) 
    { 
        return std::unique_ptr<MaterialSet>(new MaterialSet()); 
    }

    bool WaveLoader::ParseWaveFile(char* wavStream, std::shared_ptr<IResHandle> handle, uint& size)
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
            return false;
        }

        length = *((DWORD*)(wavStream+pos));
        pos += sizeof(DWORD);

        type = *((DWORD*)(wavStream+pos));
        pos += sizeof(DWORD);

        if(type != mmioFOURCC('W', 'A', 'V', 'E'))
        {
            return false;
        }

        fileEnd = length - 4;

        ZeroMemory(&data->m_format, sizeof(WAVEFORMATEX));

        bool bufferCopied = false;

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
                    bufferCopied = true;
                    /*if(length != size)
                    {
                        LOG_ERROR("strange");
                    } */
                    char* buffer = new char[length];
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

                return true;
            }

            if(length & 1)
            {
                ++pos;
                ++file;
            }
        }
        return false;
    }

    int WaveLoader::VLoadRessource(char* source, uint size, std::shared_ptr<IResHandle>& handle)
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

    IVRamHandle* GeometryCreator::VGetHandle(void)
    {
        return new chimera::d3d::Geometry();
    }

    void GeometryCreator::VCreateHandle(IVRamHandle* handle)
    {
        IGeometry* geo = (IGeometry*)handle;

        std::string file = handle->VGetResource().m_name;
        std::string meshName = "";
        std::vector<std::string> split = util::split(file, '$');

        if(split.size() > 1)
        {
            meshName = split[0];
            file = split[1];
        }

        std::shared_ptr<IMeshSet> meshSet = std::static_pointer_cast<IMeshSet>(CmGetApp()->VGetCache()->VGetHandle(CMResource(file)));
        const IMesh* mesh;
        if(meshName != "")
        {
            mesh = meshSet->VGetMesh(meshName);
        }
        else
        {
            mesh = meshSet->VGetMesh(0);
        }

        geo->VSetIndexBuffer(mesh->VGetIndices(), mesh->VGetIndexCount());
        geo->VSetVertexBuffer(mesh->VGetVertices(), mesh->VGetVertexCount(), mesh->VGetVertexStride());
        geo->VSetTopology(eTopo_Triangles);
    }

    IVRamHandle* TextureCreator::VGetHandle(void)
    {
        return new chimera::d3d::Texture2D();
    }

    void TextureCreator::VCreateHandle(IVRamHandle* handle)
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