//
// Created by yuan on 3/30/21.
//
#include <iostream>
#include "bmutility_string.h"

int main(int argc, char *argv[])
{
    int pdata[]={0,2,4,6,8,10,0,1,2,3,4,5};
    std::string base64_str = bm::base64_enc(pdata, sizeof(pdata));
    std::cout << base64_str << std::endl;

    int real_size;
    int length = base64_str.length();
    int size = base64_str.size();
    assert(length == size);

    std::string out = bm::base64_dec(base64_str.data(), base64_str.length());
    int *out_intp = (int*)out.data();
    assert(0 == memcmp(out.data(), pdata, sizeof(pdata)));
    for(int i = 0;i < sizeof(pdata)/sizeof(int); ++i) {
        printf("%d ", out_intp[i]);
    }
    printf("\n");

}