#include "Material.h"
#include "Resources.h"
#include "GameApp.h"

namespace tbd 
{
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