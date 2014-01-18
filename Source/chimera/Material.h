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
        float m_specCoef, m_reflectance, m_texScale;
        bool m_hasNormal; //TODO
    public:

        Material(void) : m_specular(0.0f,0.0f,0.0f,0), m_diffuse(0.5f, 0.5f, .5f, 1.0f), m_ambient(0.5f,0.5f,0.5f,0), m_specCoef(1), m_reflectance(0), m_texScale(1), m_hasNormal(false), m_textureDiffuse("default.png")
        {

        }

        Material(const Material& mat)
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

        const util::Vec4& VGetSpecular(void) const { return m_specular; }

        const util::Vec4& VGetDiffuse(void) const { return m_diffuse; }

        const util::Vec4& VGetAmbient(void) const { return m_ambient; }

        float VGetSpecularExpo(void) { return m_specCoef; }

        float VGetReflectance(void) { return m_reflectance; }

        float VGetTextureScale(void) { return m_texScale; }

        const chimera::CMResource& VGetTextureDiffuse(void) const { return m_textureDiffuse; }

        const chimera::CMResource& VGetTextureNormal(void) const { return m_textureNormal; }

        void Material::operator=(const Material& mat)
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

        ~Material(void) { }
    };

    class MaterialSet : public ResHandle
    {
    private:
        std::map<uint, std::shared_ptr<IMaterial>> m_indexToMaterials;
        std::map<std::string, uint> m_stringToMaterial;

    public:
        std::map<uint, std::shared_ptr<IMaterial>>* GetMaterials(void) { return &m_indexToMaterials; }
        void AddMaterial(std::shared_ptr<IMaterial> material, std::string& name, uint index) 
        { 
            m_indexToMaterials[index] = material;
            m_stringToMaterial[name] = index;
        }
        uint GetMaterialIndex(std::string& name) { return m_stringToMaterial[name]; }
        std::shared_ptr<IMaterial> GetMaterial(uint pos) { return m_indexToMaterials[pos]; }
        const util::Vec4& GetSpecular(uint pos) { return m_indexToMaterials[pos]->VGetSpecular(); }
        const util::Vec4& GetDiffuse(uint pos) { return m_indexToMaterials[pos]->VGetDiffuse(); }
        const util::Vec4& GetAmbient(uint pos) { return m_indexToMaterials[pos]->VGetAmbient(); }
        float VGetSpecularExpo(uint pos) { return m_indexToMaterials[pos]->VGetSpecularExpo(); }
        float GetReflectance(uint pos) { return m_indexToMaterials[pos]->VGetReflectance(); }
        float GetTextureScale(uint pos) { return m_indexToMaterials[pos]->VGetTextureScale(); }
        const chimera::CMResource& GetTexture(uint pos) { return m_indexToMaterials[pos]->VGetTextureDiffuse(); }
        const chimera::CMResource& GetTextureNormal(uint pos) { return m_indexToMaterials[pos]->VGetTextureNormal(); }
    };
};
