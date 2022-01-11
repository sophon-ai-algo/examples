//
// Created by yuan on 3/25/21.
//
#include "bmutility_string.h"
#include <stdarg.h>

namespace bm {
    std::vector<std::string> split(std::string str, std::string pattern)
    {
        std::string::size_type pos;
        std::vector<std::string> result;
        str += pattern;
        size_t size = str.size();

        for (size_t i = 0; i < size; i++)
        {
            pos = str.find(pattern, i);
            if (pos < size)
            {
                std::string s = str.substr(i, pos - i);
                result.push_back(s);
                i = pos + pattern.size() - 1;
            }
        }
        return result;
    }

    bool start_with(const std::string &str, const std::string &head)
    {
        return str.compare(0, head.size(), head) == 0;
    }

    std::string file_name_from_path(const std::string& path, bool hasExt){
        int pos = path.find_last_of('/');
        std::string str = path.substr(pos+1, path.size());
        if (!hasExt) {
            pos = str.find_last_of('.');
            if (std::string::npos != pos) {
                str = str.substr(0, pos);
            }
        }
        return str;
    }

    std::string file_ext_from_path(const std::string& str){
        std::string ext;
        auto pos = str.find_last_of('.');
        if (std::string::npos != pos) {
            ext = str.substr(pos+1, str.size());
        }
        return ext;
    }

    std::string format(const char *pszFmt, ...)
    {
        std::string str;
        va_list args, args2;
        va_start(args, pszFmt);
        {
            std::vector<char> vectorChars;
            va_copy(args2, args);
            int nLength = vsnprintf(nullptr, 0, pszFmt, args2);
            nLength += 1;  //上面返回的长度是包含\0，这里加上
            vectorChars.resize(nLength);
            vsnprintf(vectorChars.data(), nLength, pszFmt, args);
            str.assign(vectorChars.data());
        }
        va_end(args);
        return str;
    }

    std::string base64_enc(const void *ptr, size_t sz)
    {
        const char EncodeTable[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string encoded_str;
        char *p = (char*)ptr;
        unsigned char Tmp[4]={0};
        int LineLength=0;
        for (int i = 0; i < (sz / 3); i++) {
            Tmp[1] = *p++;
            Tmp[2] = *p++;
            Tmp[3] = *p++;
            encoded_str += EncodeTable[Tmp[1] >> 2];
            encoded_str += EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
            encoded_str += EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
            encoded_str += EncodeTable[Tmp[3] & 0x3F];
            LineLength += 4;

            if (LineLength == 76) {
                encoded_str += "\r\n";
                LineLength = 0;
            }
        }

        int Mod = sz % 3;
        if (Mod == 1) {
            Tmp[1] = *p++;
            encoded_str += EncodeTable[(Tmp[1] & 0xFC) >> 2];
            encoded_str += EncodeTable[((Tmp[1] & 0x03) << 4)];
            encoded_str += "==";
        }
        else if (Mod == 2) {
            Tmp[1] = *p++;
            Tmp[2] = *p++;
            encoded_str += EncodeTable[(Tmp[1] & 0xFC) >> 2];
            encoded_str += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
            encoded_str += EncodeTable[((Tmp[2] & 0x0F) << 2)];
            encoded_str += "=";
        }

        return encoded_str;
    }

    std::string base64_dec(const void *data, size_t size)
    {
         const char DecodeTable[] =
         {
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             62, // '+'
             0, 0, 0,
             63, // '/'
             52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
             0, 0, 0, 0, 0, 0, 0,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
             0, 0, 0, 0, 0, 0,
             26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
             39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
         };

         int nValue;
         int i= 0;
         std::string decoded_str;
         const char *p = (char*)data;
         int DataByte = size;
         while (i < DataByte)
         {
             if (*p != '\r' && *p!='\n')
             {
                 nValue = DecodeTable[*p++] << 18;
                 nValue += DecodeTable[*p++] << 12;
                 decoded_str+=(nValue & 0x00FF0000) >> 16;
                 //out_bytes++;
                 if (*p != '=')
                 {
                     nValue += DecodeTable[*p++] << 6;
                     decoded_str+=(nValue & 0x0000FF00) >> 8;
                     //out_bytes++;
                     if (*p != '=')
                     {
                         nValue += DecodeTable[*p++];
                         decoded_str+=nValue & 0x000000FF;
                         //out_bytes++;
                         }
                     }
                 i += 4;
                 }
             else// CB CR
             {
                 p++;
                 i++;
             }
         }
         return decoded_str;
    }
}