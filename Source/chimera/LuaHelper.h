#pragma once
#include "Script.h"
#include "luaplus/LuaPlus.h"
#include "Vec3.h"
namespace chimera
{
    namespace script
    {
        BOOL CheckForFieldAsNumber(LuaPlus::LuaObject& obj, LPCSTR field);
        BOOL CheckForFieldsAsNumbers(LuaPlus::LuaObject& obj, INT fieldCnt, ...);
        BOOL ConvertTableToVec3(util::Vec3& vec, LuaPlus::LuaObject& obj);

        VOID ConvertAndCheckTableToVec3(util::Vec3& vec, LuaPlus::LuaObject& obj);

        LuaPlus::LuaObject GetIntegerObject(INT i);
        LuaPlus::LuaObject GetFloatObject(FLOAT f);
        LuaPlus::LuaObject GetStringObject(CONST std::string& s);
        LuaPlus::LuaObject GetVec3Object(CONST util::Vec3& vec);
    }
}

