#pragma once
#include "stdafx.h"

namespace chimera
{
    namespace util
    {
        class _Value
        {
        public:
            virtual ~_Value(void) { }
        };

        template<class T>
        class Value : public _Value
        {
        private:
            T m_value;

        public:

            Value(T value) : m_value(value)
            {

            }

            Value(const Value<T>& value)
            {
                m_value = value.GetValue();
            }

            T GetValue(void)
            {
                return m_value;
            }
        };

        class Config : public IConfig
        {
        private:
            std::map<std::string, _Value*> m_values;

        public:
            Config(void) { }

            bool VLoadConfigFile(LPCSTR file);

            template<class T>
            T GetValue(LPCSTR value)
            {
                auto it = m_values.find(value);
#ifdef _DEBUG
                if(it == m_values.end())
                {
                    LOG_CRITICAL_ERROR_A("Config value '%s' not found.", value);
                }
#endif
                Value<T>* v = static_cast<Value<T>*>(it->second);
                return v->GetValue();
            }

            bool VGetBool(LPCSTR value)
            {
                return GetValue<bool>(value);
            }

            std::string VGetString(LPCSTR value)
            {
                return GetValue<std::string>(value);
            }

            int VGetInteger(LPCSTR value)
            {
                return GetValue<int>(value);
            }

            float VGetFloat(LPCSTR value)
            {
                return GetValue<float>(value);
            }

            DOUBLE VGetDouble(LPCSTR value)
            {
                return GetValue<bool>(value);
            }

            ~Config(void);
        };
    }
}

