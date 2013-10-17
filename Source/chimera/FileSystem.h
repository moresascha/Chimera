#pragma once
#include "stdafx.h"
#include "API_Interfaces.h"

namespace chimera
{
    class IEvent;
}

namespace chimera
{
    struct _File
    {
        std::string m_name;
        std::string m_path;
        std::string m_fullPath;

        _File::_File(CONST std::string& name)
        {
            m_name = name;
        }

        _File::_File(VOID)
        {

        }

        bool operator==(CONST _File& o)
        {
            return m_name == o.m_name;
        }

        bool operator<(CONST _File o)
        {
            return m_name < o.m_name;
        }
    };

    bool operator<(CONST _File o, CONST _File o1);

	class DLL : public _File
	{
    protected:

		HINSTANCE m_instance;

    public:
        DLL(LPCSTR path);

        DLL::~DLL(VOID);

		template<typename FuncType>
		FuncType GetFunction(LPCSTR name)
		{
			FuncType add = (FuncType)GetProcAddress(m_instance, name);
			if(add == NULL)
			{
				LOG_CRITICAL_ERROR("Function not found");
			}
			return add;
		}
	};

	class FileSystem : public IFileSystem
	{
	private:
        std::map<_File, std::list<OnFileChangedCallback>*> m_callBackMap;
        VOID OnFileChanged(std::shared_ptr<chimera::IEvent> data);

	public:
		FileSystem(VOID);

		VOID VRegisterFile(LPCSTR name, LPCSTR path);
		
        VOID VRegisterCallback(LPCSTR dllName, LPCSTR path, OnFileChangedCallback cb);

        VOID VRemoveCallback(LPCSTR dllName, OnFileChangedCallback cb);
		
		~FileSystem(VOID);
	};
};

