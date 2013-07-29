#pragma once
namespace tbd 
{
    typedef UINT RenderPath;
    CONST UINT RENDERPATH_CNT = 10;
    CONST RenderPath eDRAW_TO_ALBEDO = 1 << 0;
    CONST RenderPath eDRAW_TO_SHADOW_MAP = 1 << 1;
    CONST RenderPath eDRAW_LIGHTING = 1 << 2;
    CONST RenderPath eDRAW_EDIT_MODE = 1 << 3;
    CONST RenderPath eDRAW_PICKING = 1 << 4;
    CONST RenderPath eDRAW_BOUNDING_DEBUG = 1 << 5;
    CONST RenderPath eDRAW_PARTICLE_EFFECTS = 1 << 6;
    CONST RenderPath eDRAW_DEBUG_INFOS = 1 << 7;
    CONST RenderPath eDRAW_TO_ALBEDO_INSTANCED = 1 << 8;
    CONST RenderPath eDRAW_TO_SHADOW_MAP_INSTANCED = 1 << 9;
    CONST RenderPath eDRAW_SKY = 1 << 10;

    /*
    typedef enum RenderPath 
    {
        eDRAW_TO_ALBEDO,
        eDRAW_TO_SHADOW_MAPS,
        eDRAW_LIGHTING,
        eDRAW_EDIT_MODE,
        eDRAW_PICKING,
        eDRAW_BOUNDING_DEBUG,
        eDRAW_PARTICLE_EFFECTS,
        eDRAW_DEBUG_INFOS
    } RenderPath; */
};