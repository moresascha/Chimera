#include "LuaHelper.h"
#include "Script.h"
#include "GameApp.h"
#include "luaplus/LuaPlus.h"

namespace tbd
{
    namespace script
    {
        BOOL CheckForFieldAsNumber(LuaPlus::LuaObject& obj, LPCSTR field)
        {
            return obj[field].IsConvertibleToNumber();
        }

        BOOL CheckForFieldsAsNumbers(LuaPlus::LuaObject& obj, INT fieldCnt, ...)
        {
            va_list list;
            va_start(list, fieldCnt);
            TBD_FOR_INT((UINT)fieldCnt)
            {
                if(!CheckForFieldAsNumber(obj, va_arg(list, LPCSTR)))
                {
                    va_end(list);
                    return FALSE;
                }
            }
            va_end(list);
            return TRUE;
        }

        BOOL ConvertTableToVec3(util::Vec3& vec, LuaPlus::LuaObject& obj)
        {
            if(CheckForFieldsAsNumbers(obj, 3, "x" ,"y", "z"))
            {
                vec.x = obj["x"].GetFloat();
                vec.y = obj["y"].GetFloat();
                vec.z = obj["z"].GetFloat();
                return TRUE;
            }
            else 
            {
                return FALSE;
            }
        }

        LuaPlus::LuaObject GetIntegerObject(INT i)
        {
            LuaPlus::LuaObject obj;
            obj.AssignInteger(app::g_pApp->GetScript()->GetState(), i);
            return obj;
        }

        LuaPlus::LuaObject GetFloatObject(FLOAT f)
        {
            LuaPlus::LuaObject obj;
            obj.AssignNumber(app::g_pApp->GetScript()->GetState(), (DOUBLE)f);
            return obj;
        }

        LuaPlus::LuaObject GetStringObject(CONST std::string& s)
        {
            LuaPlus::LuaObject obj;
            obj.AssignString(app::g_pApp->GetScript()->GetState(), s.c_str());
            return obj;
        }

        LuaPlus::LuaObject GetVec3Object(CONST util::Vec3& vec)
        {
            LuaPlus::LuaObject obj;
            obj.AssignNewTable(app::g_pApp->GetScript()->GetState());
            obj.SetNumber("x", vec.x);
            obj.SetNumber("y", vec.y);
            obj.SetNumber("z", vec.z);
            return obj;
        }

        VOID ConvertAndCheckTableToVec3(util::Vec3& vec, LuaPlus::LuaObject& obj)
        {
            if(!ConvertTableToVec3(vec, obj))
            {
                LOG_ERROR_NR("Failed ConvertAndCheckTableToVec3");
            }
        }
    }
}
