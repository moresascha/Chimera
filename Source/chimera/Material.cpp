#include "Material.h"
#include "Resources.h"
#include "GameApp.h"

namespace tbd 
{
    Material::Material(VOID) : m_specular(0.0f,0.0f,0.0f,0), m_diffuse(0.0f, 0.0f, 0.0f,0.0f), m_ambient(0.5f,0.5f,0.5f,0), m_specCoef(1), m_reflectance(0), m_texScale(1), m_hasNormal(FALSE), m_textureDiffuse("default.png") 
    {

    }

    Material::Material(CONST Material& mat)
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
}