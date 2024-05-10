#include<stdio.h>
#include<string.h>

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("Option is invalid,use the -v option to get the version\n");
        return 0;
    }
    if(!strstr(argv[1],"v"))
    {
        printf("Invalid option\n");
        return 0;
    }
    return 0;
}