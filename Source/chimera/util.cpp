#include "util.h"
#include "ByteStream.h"
#include <time.h>
#include <fstream>

namespace chimera
{
    namespace util 
    {
        uint getText(const char* pFileName, char** source) 
        {
            FILE *fp;
            char *content = NULL;

            uint count=0;

            if(pFileName != NULL)
            {
                 fopen_s(&fp, pFileName, "r");

                 if(fp != NULL)
                 {
                      fseek(fp, 0, SEEK_END);
                      count = ftell(fp);
                      rewind(fp);

                      if(count > 0)
                      {
                           content = new char[sizeof(char) * (count+1)];
                           count = (uint)fread(content,sizeof(char),count,fp);
                           content[count] = '\0';

                      }
                      fclose(fp);
                 }
                 else
                      printf("File '%s' not Found\n", pFileName);
             }
             *source = content;
             return count;
        }

        void InitGdiplus(void) 
        {
            Gdiplus::GdiplusStartupInput gdiplusStartupInput;
           // Initialize GDI+.
            Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
        }

        void DestroyGdiplus(void) 
        {
            Gdiplus::GdiplusShutdown(gdiplusToken);
        }

        Gdiplus::Bitmap* GetBitmapFromFile(const WCHAR* file) 
        {
            return Gdiplus::Bitmap::FromFile(file);
        }

        Gdiplus::Bitmap* GetBitmapFromBytes(char* bytes, uint size) 
        {
            //tbd::ByteStream bs(bytes, size);
            IStream* pStream;
            HGLOBAL m_hBuffer  = ::GlobalAlloc(GMEM_FIXED, size);
            void* pBuffer = ::GlobalLock(m_hBuffer);
            CopyMemory(pBuffer, bytes, size);
            HRESULT hr = ::CreateStreamOnHGlobal(m_hBuffer, true, &pStream);
            Gdiplus::Bitmap* map = Gdiplus::Bitmap::FromStream(pStream);
            pStream->Release();
            return map;
        }

        char* GetTextureData(Gdiplus::Bitmap* map) 
        {
            char* color = new char[map->GetWidth() * map->GetHeight() * 4];
            Gdiplus::ImageAttributes atts;
            Gdiplus::Color c;
            uint index = 0;
            for(int h = 0; h < (int)map->GetHeight(); ++h)
            {
                for(int w = 0; w < (int)map->GetWidth(); ++w)
                {
                    Gdiplus::PixelFormat f = map->GetPixel(w, map->GetHeight() - h - 1, &c);
                    Gdiplus::ARGB raw = c.GetValue();
                    int r = (raw >> 16) & 0xFF;
                    int g = (raw >> 8) & 0xFF;
                    int b = (raw & 0xFF);
                    int a = (raw >> 24) & 0xFF;

                    //gamma correction
                    //r = (INT)(255 * pow((DOUBLE)r / 255.0, 2.2));
                    //g = (INT)(255 * pow((DOUBLE)g / 255.0, 2.2));
                    //b = (INT)(255 * pow((DOUBLE)b / 255.0, 2.2));
                    //a = (INT)(255 * pow((DOUBLE)a / 255.0, 2.2));

                    color[index++] = (char)(r & 0xFF);//c.GetR();
                    color[index++] = (char)(g & 0xFF);//c.GetGreen();
                    color[index++] = (char)(b & 0xFF);//c.GetBlue();
                    color[index++] = (char)(a & 0xFF);//c.GetAlpha();
                }
            }
            return color;
        } 

        char* GetTextureData(const WCHAR* file) 
        {
            Gdiplus::Bitmap* b = GetBitmapFromFile(file);
            char* color = GetTextureData(b);
            delete b;
            return color;
        }

        std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) 
        {
            std::stringstream ss(s);
            std::string item;
            while(std::getline(ss, item, delim)) 
            {
                elems.push_back(item);
            }
            return elems;
        }

        std::vector<std::string> split(const std::string &s, char delim)
        {
            std::vector<std::string> elems;
            return split(s, delim, elems);
        }

        std::wstring string2wstring(const std::string& s)
        {
            return std::wstring(s.begin(), s.end());
        }

        bool CheckIfFileExists(LPCSTR file)
        {
            std::ifstream ifile(file);
            return ifile.good();
        }

        /*
        std::string2wstring s2ws(const std::string& s)
        {
            int len;
            int slength = (int)s.length() + 1;
            len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0); 
            wchar_t* buf = new wchar_t[len];
            MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
            std::wstring r(buf);
            delete[] buf;
            return r;
        }
        */
    };
}