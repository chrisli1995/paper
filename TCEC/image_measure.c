#include <linux/module.h> 
#include <linux/kernel.h> 
#include <linux/string.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/kmod.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

#define SIZE 200
#define DIR_SIZE 300

static char *Project_path = "/home/TCEC/";
static char *Base_value_path = "/home/TCEC/base/";

module_param(Project_path, charp, S_IRUGO);
module_param(Base_value_path, charp, S_IRUGO);

extern int aes_encrypt(char *keyfile, char *raw_file, char *enc_file);
extern int aes_decrypt(char *keyfile, char *enc_file, char *dec_file);
extern int hash_directory(char *dir_name, char *hash_buf, char *path_to_record_file, unsigned char *digest, int flag);
extern int crypto_sha256(unsigned char *digest, char *filename, char *hash_buf);
extern int aes_decrypt_buffer(char *enc_file, char *keyfile, char *buf_dec);
extern int validate_module_self(char *kernel_module_path, char *check_path, char *kernel_name);

int tpm_hash(char *raw_file, char *out_file)
{
    int result = 0;
    char cmdPath[] = "/bin/tpm2_hash";
    char *cmdArgv[] = {cmdPath, "-g", "sha256", "-o", out_file, raw_file, "--hex", "-T", "device:/dev/tpmrm0", NULL}; //"/home/tmp_data/klm/hash_enc.c"
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    return result;
}


int tpm_encrypt(char *raw_file, char *out_file, char *key)
{
    int result = 0;
    char cmdPath[] = "/bin/tpm2_rsaencrypt";
    char *cmdArgv[] = {cmdPath, "-c", key, "-o", out_file, raw_file, "-T", "device:/dev/tpmrm0", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    return result;
}


int tpm_decrypt(char *raw_file, char *out_file, char *key)
{
    int result = 0;
    char cmdPath[] = "/bin/tpm2_rsadecrypt";
    char *cmdArgv[] = {cmdPath, "-c", key, "-o", out_file, raw_file, "-T", "device:/dev/tpmrm0", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);
    return result;
}


int con_rm(char *raw_file)
{
    int result = 0;
    char cmdPath[] = "/bin/rm";
    char *cmdArgv[] = {cmdPath, "-rf", raw_file, NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);
    return result;
}

int docker_images(char *file)
{
    int result = 0;
    char cmdPath[] = "/bin/bash";
    char *cmdArgv[] = {"/bin/bash", "-c", "docker images --format '{{.Repository}}:{{.Tag}}' > ", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    char *name = (char *)kmalloc(strlen("docker images --format '{{.Repository}}:{{.Tag}}' > ") + strlen(file), GFP_KERNEL);

    strcpy(name, "docker images --format '{{.Repository}}:{{.Tag}}' > ");
    strcat(name, file);
    cmdArgv[2] = name;

    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    kfree(name);
    name = NULL;

    return result;
}


static int query_loc(char *path, char *image_name, char *image_name_new)
{
    int result = 0;
    char cmdPath[] = "/bin/bash";
    char *cmdArgv[] = {"/bin/bash", "-c", "docker inspect ", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};

    char head[] = "docker inspect ";
    char tail[] = " --format='{{.GraphDriver.Data.LowerDir}}'  >> ";
    char tail1[] = " --format='{{.GraphDriver.Data.UpperDir}}'  >> ";
    char sep[] = ":";
    char sep1[] = "#####\n\nBase Vaule:";

    int base_file_path_len = strlen(path) + strlen(image_name_new);
    char *base_file_path = kmalloc(base_file_path_len, GFP_KERNEL);
    char *cmdString = NULL;
    struct file *fp = NULL;
    int cmdString_Len = 0;
    int fsize = 0;

    loff_t pos;
    mm_segment_t fs;

    strcpy(base_file_path, path);
    strcat(base_file_path, image_name_new);

    cmdString_Len = strlen(head) + strlen(image_name) + strlen(tail) + base_file_path_len;
    cmdString = (char *)kmalloc(cmdString_Len + 50, GFP_KERNEL);

    strncpy(cmdString, head, cmdString_Len);
    strcat(cmdString, image_name);
    strcat(cmdString, tail);
    strcat(cmdString, base_file_path);
    cmdArgv[2] = cmdString;

   
    fp = filp_open(base_file_path, O_RDWR | O_CREAT, 0);
    if (IS_ERR(fp))
    {
        printk("base_file_path open error/n");
        return -1;
    }
    fs = get_fs();
    set_fs(KERNEL_DS);
    pos = 0;
    kernel_write(fp, sep, strlen(sep), &pos);

  
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    
    fsize = fp->f_inode->i_size;

    fs = get_fs();
    set_fs(KERNEL_DS);
    pos = fsize - 1;
    kernel_write(fp, sep, strlen(sep), &pos);

 
    memset(cmdString, 0x00, cmdString_Len + 50);

    strcpy(cmdString, head);
    strcat(cmdString, image_name);
    strcat(cmdString, tail1);
    strcat(cmdString, base_file_path);
    cmdArgv[2] = cmdString;

    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    fsize = fp->f_inode->i_size;

    pos = fsize - 1;
    kernel_write(fp, sep1, strlen(sep1), &pos);

    filp_close(fp, NULL);
    set_fs(fs);
    kfree(base_file_path);
    kfree(cmdString);
    base_file_path = NULL;
    cmdString = NULL;

    return 0;
}

static void name_change(char *image_name)
{
    int len = 0, i;
    len = strlen(image_name);
    for (i = 0; i < len; i++)
    {
        if (image_name[i] == ':')
        {
            image_name[i] = '_';
        }
        if (image_name[i] == '/')
        {
            image_name[i] = '_';
        }
    }
}

int k_write(char *path_of_record_file_hash, char *hash_buf, int len)
{
    struct file *filp = NULL;
    int fsize;

    loff_t pos;
    mm_segment_t fs;

    filp = filp_open(path_of_record_file_hash, O_RDWR | O_CREAT, 0);
    if (IS_ERR(filp))
    {
        fsize = 0;
        return -1;
    }
    else
    {
        fsize = filp->f_inode->i_size;

        fs = get_fs();
        set_fs(KERNEL_DS);
        pos = fsize;

        kernel_write(filp, hash_buf, len, &pos);
        filp_close(filp, NULL);
        set_fs(fs);
    }
    return fsize;
}

int use_hash_directory(char *path, char *dir_name, char *image_name)
{
    char hash_buf[100] = {'\0'};
    char path_of_record_file_name[512] = {'\0'};
    char path_of_record_file_hash[512] = {'\0'};
    int flag = 0, len = 64;
    unsigned char *digest = NULL;

    strcpy(path_of_record_file_name, path);
    strcat(path_of_record_file_name, ".allfile.tmp");

    strcpy(path_of_record_file_hash, path);
    strcat(path_of_record_file_hash, image_name);
    strcat(path_of_record_file_hash, ".tmp");

    digest = (unsigned char *)kmalloc(32, GFP_KERNEL);
    flag = hash_directory(dir_name, hash_buf, path_of_record_file_name, digest, 1);
    if (flag != -1)
    {
        k_write(path_of_record_file_hash, hash_buf, len);
    }

    con_rm(path_of_record_file_name);
    kfree(digest);
    digest = NULL;

    return 0;
}


static int dir_parse(char *path, char *filename, char *image_name)
{
    struct file *fp;
    char buf[DIR_SIZE] = "\0";
    char dir_loc[DIR_SIZE] = "\0";
    int i = 0;
    int j = 0, k = 0;
    int count = 0;
    int index[4] = {0};
    int kernel_read_num = 20;
    int offset = 0;
    int fsize = 0;

    mm_segment_t fs;
    loff_t pos;

    fp = filp_open(filename, O_RDONLY, 0);

    if (IS_ERR(fp))
    {
        printk("create file error/n");
        return -1;
    }
    fsize = fp->f_inode->i_size;

    fs = get_fs();
    set_fs(KERNEL_DS);
    pos = 0;

    while (pos < fsize)
    {
        kernel_read_num = kernel_read(fp, buf, DIR_SIZE, &pos);
        for (i = 0; i < DIR_SIZE; i++)
        {

            if (buf[i] == ':')
            {
                index[count] = i;
                count = count + 1;
            }
            if (buf[i] == '#')
            {
                index[count] = i;
                count = count + 1;
                break;
            }
            if (count == 2)
            {
                break;
            }
        }

        count = 0;
        for (k = 0, j = index[0] + 1; j < index[1]; j++)
        {
            if (buf[j] != '\n')
            {
                dir_loc[k] = buf[j];
                k++;
            }
        }


        if (!strstr(dir_loc, "no value"))
        {
            use_hash_directory(path, dir_loc, image_name);
        }

        pos = offset + index[1] - 2;
        offset = pos;


        if (buf[i] == '#')
        {
            break;
        }

        for (i = 0; i < DIR_SIZE; i++)
        {
            dir_loc[i] = '\0';
        }
        for (i = 0; i < DIR_SIZE; i++)
        {
            buf[i] = '\0';
        }
    }

    filp_close(fp, NULL);
    set_fs(fs);

    return 0;
}

static char *hash_res_write(char *path, char *image_name, char *hash_res, int flag)
{
    char hash_file_name[500] = {'\0'};
    char base_file_name[500] = {'\0'};
    int len = 64;
    char Mkey_path[500] = {'\0'};
    char Dkey_path[500] = {'\0'};
    unsigned char *digest = NULL;

    strcpy(Mkey_path, path);
    strcat(Mkey_path, "base_key");

    strcpy(Dkey_path, path);
    strcat(Dkey_path, ".base_key_dec");

    digest = (unsigned char *)kmalloc(32, GFP_KERNEL);

    strcpy(hash_file_name, path);
    strcat(hash_file_name, image_name);
    strcat(hash_file_name, ".tmp");

    memset(digest, 0, 32);
    crypto_sha256(digest, hash_file_name, hash_res);

   
    if (flag == 0)
    {
        tpm_decrypt(Mkey_path, Dkey_path, "0x81010002");
        strcpy(base_file_name, path);
        strcat(base_file_name, image_name);

        k_write(base_file_name, hash_res, len);
        aes_encrypt(Dkey_path, base_file_name, base_file_name);
    }

    con_rm(Dkey_path);
    con_rm(hash_file_name);
    kfree(digest);
    digest = NULL;

    return hash_res;
}

static int base_generate(char *path, char *image_name_new)
{

    char single_image_base_value[500] = {'\0'};
    char hash_res[100] = {'\0'};

    strncpy(single_image_base_value, path, 500);
    strcat(single_image_base_value, image_name_new);

    dir_parse(path, single_image_base_value, image_name_new);
    hash_res_write(path, image_name_new, hash_res, 0);

    return 0;
}


static int k_read(char *filename, char buf[], int len)
{
    struct file *fp;
    mm_segment_t fs;
    loff_t pos;
    fp = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(fp))
    {
        printk("create file error/n");
        return -1;
    }
    fs = get_fs();
    set_fs(KERNEL_DS);
    pos = 0;
    kernel_read(fp, buf, len, &pos);
    filp_close(fp, NULL);
    set_fs(fs);
    return 0;
}

static int parse(char *str_word, char *filename, char *cmdline)
{

    char buf[200] = {"\0"};
    int count = 0, i, j;
    int loc_flag = 0, step_index = 0;
    int read_num = 10;
    struct file *fp;
    mm_segment_t fs;
    loff_t pos;

    fp = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(fp))
    {
        printk("func parse create file error/n");
        return -1;
    }
    fs = get_fs();
    set_fs(KERNEL_DS);
    pos = 0;

    while (read_num > 3)
    {
        kernel_read(fp, buf, SIZE, &pos);
        for (j = 0; j < SIZE; j++)
        {
            if (buf[j] == '\0')
            {
                break;
            }
        }
        if (j == SIZE)
        {
            buf[j - 1] = '\0';
        }
        read_num = strlen(buf);

        for (i = 0, j = 0; i < read_num; i++)
        {
            if (buf[i] == '"')
            {
                count = count + 1;
                i = i + 1;
            }
            if (count == 1)
            {
                str_word[j] = buf[i];
                j++;
            }
            if (buf[i] == '"')
            {
                count = count + 1;
            }
            if (count == 2)
            {
                step_index = i;


                if (strstr(str_word, ":") != NULL) 
                {
                    if (strstr(str_word, "sha256") == NULL)
                    {
                        if (strstr(cmdline, str_word) != NULL)
                        {
                            filp_close(fp, NULL);
                            set_fs(fs);
                            return 0;
                        }
                    }
                }

                for (j = 0; j < 50; j++) 
                {
                    str_word[j] = '\0';
                }
                count = 0;
                j = 0;
            }
        }
        for (j = 0; j < 50; j++)
        {
            str_word[j] = '\0';
        }

        count = 0;
        loc_flag = loc_flag + step_index + 1; 
        pos = loc_flag;

        for (j = 0; j < 200; j++)
        {
            buf[j] = '\0';
        }
    }
    filp_close(fp, NULL);
    set_fs(fs);
    strcpy(str_word, "none");
    return 0;
}

int read_base(char *buf, char *base_value, int hash_len)
{
    char str[] = "Base Vaule:";
    char *str_loc = NULL;
    int i = 0;

    str_loc = strstr(buf, str);
    if (str_loc != NULL)
    {
        for (i = 0; i < hash_len; i++)
        {
            base_value[i] = str_loc[i + 11];
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

static int file_size(char *filename)
{
    struct file *filp = NULL;
    int flag;

    filp = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(filp))
    {
        flag = 0;
    }
    else
    {
        flag = filp->f_inode->i_size;
        filp_close(filp, NULL);
    }
    return flag;
}

int dir_parse_meas(char *path, char *buf, int len, char *image_name)
{
    char dir_loc[512] = {'\0'};
    int count = 0;
    int i, j = 0;
    for (i = 0; i < len - 10; i++)
    {
        if (buf[i] == ':' || buf[i] == '#')
        {
            count = count + 1;
        }
        if (buf[i] != ':' && count == 1)
        {
            dir_loc[j] = buf[i];
            j++;
        }
        if (count == 2)
        {
            if (!strstr(dir_loc, "no value"))
            {
                use_hash_directory(path, dir_loc, image_name);
            }
            count--;
            memset(dir_loc, 0, 512);
            j = 0;
        }
        if (buf[i] == '#')
            break;
    }

    return 0;
}

static int measure(char *pro_path, char *base_value_path, char *image_name)
{
    char image_name_new[50] = {'\0'};
    char single_image_base_value[500] = {'\0'};
    char hash_res[100] = {'\0'};
    char buf1[200] = {'\0'};
    char Mkey_path[512] = {'\0'};
    char Dkey_path[512] = {'\0'};
    int flag = 1, i, len = 64, size = 0;
    char *content = NULL;

    strcpy(image_name_new, image_name);
    name_change(image_name_new);
    strcpy(single_image_base_value, base_value_path);
    strcat(single_image_base_value, image_name_new);

    strcpy(Mkey_path, base_value_path);
    strcat(Mkey_path, "base_key");

    strcpy(Dkey_path, base_value_path);
    strcat(Dkey_path, ".base_key_dec");

    
    size = file_size(single_image_base_value) + 10;
    content = (char *)kmalloc(size, GFP_KERNEL);

    memset(content, 0, size);

    if (0 != tpm_decrypt(Mkey_path, Dkey_path, "0x81010002"))
    {
        kfree(content);
        content = NULL;
        return -2;
    }
    if (aes_decrypt_buffer(single_image_base_value, Dkey_path, content) < 0)
    {
        kfree(content);
        content = NULL;
        return -2;
    }
    if (-1 == read_base(content, buf1, 64))
    {
        kfree(content);
        content = NULL;
        return -2;
    }

    dir_parse_meas(pro_path, content, size, image_name_new);
    hash_res_write(pro_path, image_name_new, hash_res, 1);

    printk("base hash:%s/n", buf1);
    printk("measure hash:%s/n", hash_res);
    for (i = 0; i < len; i++)
    {
        if (buf1[i] != hash_res[i])
        {
            flag = 0;
            break;
        }
    }

    kfree(content);
    content = NULL;
    con_rm(Dkey_path);

    return flag;
}

static int file_exist(char *layer_path, char *file_path)
{
    struct file *filp = NULL;
    int flag;
    char *end_path = (char *)kmalloc(500, GFP_KERNEL);
    strcpy(end_path, layer_path);
    strcat(end_path, file_path);
    filp = filp_open(end_path, O_RDONLY, 0);
    if (IS_ERR(filp))
    {
        flag = 0;

    }
    else
    {
        flag = 1;
        filp_close(filp, NULL);
    }

    kfree(end_path);
    end_path = NULL;
    return flag;
}

EXPORT_SYMBOL(name_change);
EXPORT_SYMBOL(docker_images);
EXPORT_SYMBOL(query_loc);
EXPORT_SYMBOL(dir_parse);
EXPORT_SYMBOL(base_generate);
EXPORT_SYMBOL(measure);
EXPORT_SYMBOL(parse);
EXPORT_SYMBOL(file_exist);
EXPORT_SYMBOL(con_rm);
EXPORT_SYMBOL(k_read);
MODULE_LICENSE("GPL");
