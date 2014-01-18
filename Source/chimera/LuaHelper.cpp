#include "LuaHelper.h"
#include "Script.h"
#include "GameApp.h"
#include "luaplus/LuaPlus.h"

namespace chimera
{
    namespace script
    {
        bool CheckForFieldAsNumber(LuaPlus::LuaObject& obj, LPCSTR field)
        {
            return obj[field].IsConvertibleToNumber();
        }

        bool CheckForFieldsAsNumbers(LuaPlus::LuaObject& obj, int fieldCnt, ...)
        {
            va_list list;
            va_start(list, fieldCnt);
            TBD_FOR_INT((uint)fieldCnt)
            {
                if(!CheckForFieldAsNumber(obj, va_arg(list, LPCSTR)))
                {
                    va_end(list);
                    return false;
                }
            }
            va_end(list);
            return true;
        }

        bool ConvertTableToVec3(util::Vec3& vec, LuaPlus::LuaObject& obj)
        {
            if(CheckForFieldsAsNumbers(obj, 3, "x" ,"y", "z"))
            {
                vec.x = obj["x"].GetFloat();
                vec.y = obj["y"].GetFloat();
                vec.z = obj["z"].GetFloat();
                return true;
            }
            else 
            {
                return false;
            }
        }

        LuaPlus::LuaObject GetIntegerObject(int i)
        {
            LuaPlus::LuaObject obj;
            obj.AssignInteger(chimera::g_pApp->GetScript()->GetState(), i);
            return obj;
        }

        LuaPlus::LuaObject GetFloatObject(float f)
        {
            LuaPlus::LuaObject obj;
            obj.AssignNumber(chimera::g_pApp->GetScript()->GetState(), (DOUBLE)f);
            return obj;
        }

        LuaPlus::LuaObject GetStringObject(const std::string& s)
        {
            LuaPlus::LuaObject obj;
            obj.AssignString(chimera::g_pApp->GetScript()->GetState(), s.c_str());
            return obj;
        }

        LuaPlus::LuaObject GetVec3Object(const util::Vec3& vec)
        {
            LuaPlus::LuaObject obj;
            obj.AssignNewTable(chimera::g_pApp->GetScript()->GetState());
            obj.SetNumber("x", vec.x);
            obj.SetNumber("y", vec.y);
            obj.SetNumber("z", vec.z);
            return obj;
        }

        void ConvertAndCheckTableToVec3(util::Vec3& vec, LuaPlus::LuaObject& obj)
        {
            if(!ConvertTableToVec3(vec, obj))
            {
                LOG_ERROR_NR("Failed ConvertAndCheckTableToVec3");
            }
        }
    }
}
