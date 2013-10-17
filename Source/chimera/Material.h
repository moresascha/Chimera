#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Cache.h"
#include <vector>

namespace chimera 
{
    class Material : public IMaterial 
    {
        friend class Geometry;
    public:
        chimera::CMResource m_textureDiffuse;
        chimera::CMResource m_textureNormal;
        util::Vec4 m_specular;
        util::Vec4 m_diffuse;
        util::Vec4 m_ambient;
        FLOAT m_specCoef, m_reflectance, m_texScale;
        BOOL m_hasNormal; //TODO
    public:

        Material(VOID) : m_specular(0.0f,0.0f,0.0f,0), m_diffuse(0.0f, 0.0f, 0.0f,0.0f), m_ambient(0.5f,0.5f,0.5f,0), m_specCoef(1), m_reflectance(0), m_texScale(1), m_hasNormal(FALSE), m_textureDiffuse("default.png")
        {

        }

        Material(CONST Material& mat)
        {
            m_ambient = mat.m_ambient;
            m_diffuse = mat.m_diffuse;
            m_specular = mat.m_specular;
            m_reflectance = mat.m_reflectance;
            m_textureDiffuse = mat.m_textureDiffuse;
            m_textureNormal = mat.m_textureNormal;
            m_specCoef = mat.m_specCoef;
            m_texScale = mat.m_texScale;
        }

        CONST util::Vec4& VGetSpecular(VOID) CONST { return m_specular; }

        CONST util::Vec4& VGetDiffuse(VOID) CONST { return m_diffuse; }

        CONST util::Vec4& VGetAmbient(VOID) CONST { return m_ambient; }

        FLOAT VGetSpecularExpo(VOID) { return m_specCoef; }

        FLOAT VGetReflectance(VOID) { return m_reflectance; }

        FLOAT VGetTextureScale(VOID) { return m_texScale; }

        CONST chimera::CMResource& VGetTextureDiffuse(VOID) CONST { return m_textureDiffuse; }

        CONST chimera::CMResource& VGetTextureNormal(VOID) CONST { return m_textureNormal; }

        VOID Material::operator=(CONST Material& mat)
        {
            m_textureDiffuse = mat.m_textureDiffuse;
            m_textureNormal = mat.m_textureNormal;
            m_ambient = mat.m_ambient;
            m_diffuse = mat.m_diffuse;
            m_hasNormal = mat.m_hasNormal;
            m_reflectance = mat.m_reflectance;
            m_specCoef = mat.m_specCoef;
            m_specular = mat.m_specular;
            m_texScale = mat.m_texScale;
        }

        ~Material(VOID) { }
    };

    class MaterialSet : public ResHandle
    {
    private:
        std::map<UINT, std::shared_ptr<IMaterial>> m_indexToMaterials;
        std::map<std::string, UINT> m_stringToMaterial;

    public:
        std::map<UINT, std::shared_ptr<IMaterial>>* GetMaterials(VOID) { return &m_indexToMaterials; }
        VOID AddMaterial(std::shared_ptr<IMaterial> material, std::string& name, UINT index) 
        { 
            m_indexToMaterials[index] = material;
            m_stringToMaterial[name] = index;
        }
        UINT GetMaterialIndex(std::string& name) { return m_stringToMaterial[name]; }
        std::shared_ptr<IMaterial> GetMaterial(UINT pos) { return m_indexToMaterials[pos]; }
        CONST util::Vec4& GetSpecular(UINT pos) { return m_indexToMaterials[pos]->VGetSpecular(); }
        CONST util::Vec4& GetDiffuse(UINT pos) { return m_indexToMaterials[pos]->VGetDiffuse(); }
        CONST util::Vec4& GetAmbient(UINT pos) { return m_indexToMaterials[pos]->VGetAmbient(); }
        FLOAT VGetSpecularExpo(UINT pos) { return m_indexToMaterials[pos]->VGetSpecularExpo(); }
        FLOAT GetReflectance(UINT pos) { return m_indexToMaterials[pos]->VGetReflectance(); }
        FLOAT GetTextureScale(UINT pos) { return m_indexToMaterials[pos]->VGetTextureScale(); }
        CONST chimera::CMResource& GetTexture(UINT pos) { return m_indexToMaterials[pos]->VGetTextureDiffuse(); }
        CONST chimera::CMResource& GetTextureNormal(UINT pos) { return m_indexToMaterials[pos]->VGetTextureNormal(); }
    };
};
