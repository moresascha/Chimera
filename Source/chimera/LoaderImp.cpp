#include "Resources.h"
#include "Material.h"
#include "GameApp.h"
#include "Material.h"
#include "Mesh.h"
#include <vector>
#include <sstream>
#include "util.h"

namespace tbd 
{

    struct SubMeshData 
    {
        std::vector<Face> m_faces;
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

    UINT GetTriple(std::string s, Triple& triple) 
    {
        CHAR* context = NULL;
        CHAR* elements = strtok_s(const_cast<CHAR*>(s.c_str()), "/", &context);
        std::string elems[3] = {"-1", "-1", "-1"};
        UINT count = 0;
        while(elements != NULL)
        {
            elems[count] = elements;
            elements = strtok_s(NULL, "/", &context);
            count++;
        }
        triple.position = atoi(elems[0].c_str()) - 1;
        triple.texCoord = atoi(elems[1].c_str()) - 1;
        triple.normal = atoi(elems[2].c_str()) - 1;
        return count;
    }

    INT ObjLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<tbd::ResHandle> handle) 
    {
        std::shared_ptr<tbd::Mesh> mesh = std::static_pointer_cast<tbd::Mesh>(handle);

        std::string desc(source);
        std::stringstream sss;
        sss << desc;

        clock_t start = clock();
        SubMeshData subMeshData;
        RawGeometryData rawData;
        std::string matName;
        UINT lastIndexStart = 0;
        while(sss.good()) 
        {
            std::string line;
            std::getline(sss, line);
            std::stringstream ss(line);

            std::string flag;
            ss >> flag;
            if(flag == "mtllib")
            {
                std::string mtllib;
                ss >> mtllib;
                mesh->m_materials = mtllib;
            }
            if(flag == "usemtl")
            {
                if(subMeshData.m_triplesCount == 0) 
                {
                    matName.clear();
                    ss >> matName;
                    continue;
                }
                std::shared_ptr<tbd::MaterialSet> materials = std::static_pointer_cast<tbd::MaterialSet>(app::g_pApp->GetCache()->GetHandle(mesh->GetMaterials()));
                mesh->AddIndexBufferInterval(lastIndexStart, subMeshData.m_trianglesCount * 3, materials->GetMaterialIndex(matName));
                lastIndexStart = subMeshData.m_trianglesCount * 3;
                matName.clear();
                ss >> matName;
            }
            if(flag == "v")
            {
                util::Vec3 vertex;
                std::string x, y, z;
                ss >> x;
                ss >> y;
                ss >> z;
                vertex.Set((FLOAT)atof(x.c_str()), (FLOAT)atof(y.c_str()), (FLOAT)atof(z.c_str()));
                rawData.m_positions.push_back(vertex);
            }
            else if(flag == "vt")
            {
                util::Vec3 texCoord;
                std::string x, y;
                ss >> x;
                ss >> y;
                texCoord.Set((FLOAT)atof(x.c_str()), (FLOAT)atof(y.c_str()), 0);
                rawData.m_texCoords.push_back(texCoord);
            } 
            else if(flag == "vn") 
            {
                util::Vec3 normal;
                std::string x, y, z;
                ss >> x;
                ss >> y;
                ss >> z;
                normal.Set((FLOAT)atof(x.c_str()), (FLOAT)atof(y.c_str()), (FLOAT)atof(z.c_str()));
                rawData.m_normals.push_back(normal);
            }
            else if(flag == "f")
            {
                std::vector<std::string> elems;
                while(ss.good())
                {
                    std::string s;
                    ss >> s;
                    if(s != "")
                    {
                        elems.push_back(s);
                    }
                }
                Face face;
                UINT triplesCount = (UINT)elems.size();
                for(UINT i = 0; i < triplesCount; ++i)
                {
                    Triple triple;
                    GetTriple(elems[i], triple); 
                    face.m_triples.push_back(triple);
                }
                subMeshData.m_triplesCount += triplesCount;
                subMeshData.m_trianglesCount += triplesCount / 2;
                subMeshData.m_faces.push_back(face);
                mesh->m_faces.push_back(face);
            }
        }

        std::shared_ptr<tbd::MaterialSet> materials = std::static_pointer_cast<tbd::MaterialSet>(app::g_pApp->GetCache()->GetHandle(mesh->GetMaterials()));
        mesh->AddIndexBufferInterval(lastIndexStart, subMeshData.m_trianglesCount * 3, materials->GetMaterialIndex(matName));

        std::vector<UINT> lIndices;
        std::vector<FLOAT> lVertices;
        std::vector<Triple> existingTriples;

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

                auto it = std::find(existingTriples.begin(), existingTriples.end(), t);

                if(it == existingTriples.end())
                {
                    util::Vec3& pos0 = rawData.m_positions[t.position];

                    mesh->m_aabb.AddPoint(pos0);

                    util::Vec3 norm0;
                    if(!rawData.m_normals.empty())
                    {
                        norm0 = rawData.m_normals[t.normal];
                    }
                    
                    util::Vec3 texCoord0;
                    if(!rawData.m_texCoords.empty())
                    {
                        texCoord0 = rawData.m_texCoords[t.texCoord];
                    }
                    
                    lVertices.push_back(pos0.x);
                    lVertices.push_back(pos0.y);
                    lVertices.push_back(pos0.z);

                    lVertices.push_back(norm0.x);
                    lVertices.push_back(norm0.y);
                    lVertices.push_back(norm0.z);

                    lVertices.push_back(texCoord0.x);
                    lVertices.push_back(texCoord0.y); //Note: textures used by obj.files are y-inverted, might change in the future since I use old textures
                    /*
                    std::stringstream ss;
                    ss << "Created triple= p=" << t.position << ", tx=" << t.texCoord << ", n=" << t.normal;
                    LOG_INFO(ss.str()); */
                    /*lVertices.push_back(tangents[i].x);
                    lVertices.push_back(tangents[i].y);
                    lVertices.push_back(tangents[i].z); */

                    existingTriples.push_back(t);
                }
            }

            /*2,0,1,2,0,3*/
            auto i0 = std::find(existingTriples.begin(), existingTriples.end(), it->m_triples[0]);
            auto i1 = std::find(existingTriples.begin(), existingTriples.end(), it->m_triples[1]);
            auto i2 = std::find(existingTriples.begin(), existingTriples.end(), it->m_triples[2]);

            __int64 index0 = i0 - existingTriples.begin();
            __int64 index1 = i1 - existingTriples.begin();
            __int64 index2 = i2 - existingTriples.begin();

            lIndices.push_back((UINT)index2);

            lIndices.push_back((UINT)index1);

            lIndices.push_back((UINT)index0);

            if(it->m_triples.size() == 4)
            {
                auto i3 = std::find(existingTriples.begin(), existingTriples.end(), it->m_triples[3]);
                __int64 index3 = i3 - existingTriples.begin();
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

        mesh->SetIndices(indices, indexCount);
        mesh->SetVertices(vertices, vertexCount, stride);
        mesh->GetAABB().Construct();

        delete[] source;
        return indexCount * sizeof(UINT) + vertexCount * stride * sizeof(FLOAT);
    }

    INT MaterialLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle)
    {
        std::shared_ptr<tbd::MaterialSet> materials = std::static_pointer_cast<tbd::MaterialSet>(handle);

        std::stringstream ss;
        std::string desc(source);
        ss << desc;
        std::string line;
        std::shared_ptr<tbd::Material> current = NULL;
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
                current = std::shared_ptr<tbd::Material>(new tbd::Material());
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
                current->m_textureDiffuse = tbd::Resource(val);
                materials->GetRessourceCache()->GetHandle(current->m_textureDiffuse); //preload texture
            }
            else if(prefix == "map_Kn")
            {
                std::string val;
                streamLine >> val;
                current->m_hasNormal = TRUE;
                current->m_textureNormal = tbd::Resource(val);
                materials->GetRessourceCache()->GetHandle(current->m_textureNormal); //preload texture
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

    INT ImageLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle) 
    {
        Gdiplus::Bitmap* b = util::GetBitmapFromBytes(source, size);

        std::shared_ptr<ImageExtraData> extraData = 
            std::shared_ptr<ImageExtraData>(new ImageExtraData(b->GetWidth(), b->GetHeight(), b->GetPixelFormat()));

        handle->SetExtraData(extraData);
        
        handle->SetBuffer(util::GetTextureData(b));

        size = b->GetWidth() * b->GetWidth() * 4;

        delete b;
        delete[] source;
        
        return size;
    }


    ResHandle* ObjLoader::VCreateHandle(VOID)  { return new tbd::Mesh(); }
    ResHandle*  MaterialLoader::VCreateHandle(VOID) { return new tbd::MaterialSet(); }

    BOOL WaveLoader::ParseWaveFile(CHAR* wavStream, std::shared_ptr<ResHandle> handle, UINT& size)
    {
        std::shared_ptr<WaveSoundExtraDatra> data = std::static_pointer_cast<WaveSoundExtraDatra>(handle->GetExtraData());

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
                    handle->SetBuffer(buffer);
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

    INT WaveLoader::VLoadRessource(CHAR* source, UINT size, std::shared_ptr<ResHandle> handle)
    {
        //DefaultRessourceLoader::VLoadRessource(source, size, handle);

        std::shared_ptr<WaveSoundExtraDatra> data = std::shared_ptr<WaveSoundExtraDatra>(new WaveSoundExtraDatra());
        handle->SetExtraData(data);

        if(!ParseWaveFile(source, handle, size))
        {
            LOG_CRITICAL_ERROR("invalid wave format");
        }

        return size;
    }
}