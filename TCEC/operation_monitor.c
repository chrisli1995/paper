#include <linux/module.h> 
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/kmod.h>
#include <linux/sched.h>
#include <linux/vmalloc.h>
#include <linux/uaccess.h>
#include <linux/fs.h>
#include <linux/kallsyms.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/tty.h>
#include <linux/hash.h>
#include <linux/crypto.h>
#include <crypto/hash.h>
#include <crypto/algapi.h>
#include <linux/random.h>

#define BUF_SIZE 300
#define MAX_FILE_NAME_LEN 500
#define image_record_path "/var/lib/docker/image/overlay2/repositories.json"
#define image_name_size 50
#define PATH_LEN 350
#define PATH_LEN_S 250

static char *Project_path = "/home/TCEC/";
static char *Base_value_path = "/home/TCEC/base/";

module_param(Project_path, charp, S_IRUGO);
module_param(Base_value_path, charp, S_IRUGO);

MODULE_LICENSE("GPL");

extern int docker_images(char *file);
extern int query_loc(char *path, char *image_name, char *image_name_new);
extern void name_change(char *image_name);
extern int dir_parse(char *path, char *filename, char *image_name);
extern int dir_hash(char *path, char *dir_name, char *image_name);
extern int base_generate(char *path, char *image_name_new);
extern int measure(char *pro_path, char *base_value_path, char *image_name);
extern int parse(char *str_word, char *filename, char *cmdline);
extern int file_exist(char *layer_path, char *file_path);
extern int get_aes_key(int byte, char *filename, int flag);
extern int aes_decrypt(char *keyfile, char *enc_file, char *dec_file);
extern int aes_encrypt(char *keyfile, char *raw_file, char *enc_file);
extern int validate_module_self(char *kernel_module_path, char *check_path, char *kernel_name);

typedef asmlinkage long (*sys_call_ptr_t)(const struct pt_regs *);
static sys_call_ptr_t *sys_call_table;
sys_call_ptr_t old_execve;

void printString(char *string)
{

    struct tty_struct *tty;
    tty = get_current_tty();

    if (tty != NULL)
    {
        (tty->driver->ops->write)(tty, string, strlen(string));
    }

    else
        printk("tty equals to zero");
}

int new_dir(char *dir_name)
{
    int result = 0;
    char cmdPath[] = "/bin/bash";
    char *cmdArgv[] = {cmdPath, "-c", "/bin/mkdir 111", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    char *string = NULL;
    int string_len = 0;
    string_len = strlen("/bin/mkdir ") + strlen(dir_name);
    string = kmalloc(string_len, GFP_KERNEL); 

    strcpy(string, "/bin/mkdir ");
    strcat(string, dir_name);

    cmdArgv[2] = string;
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    kfree(string);
    string = NULL;
    return result;
}


static int end_replace(char str1[], char rep[])
{
    int raw_len;
    int i;

    raw_len = strlen(str1);
    for (i = raw_len; i > 0; i--)
    {
        if (str1[i] == '/')
        {
            str1[i] = '\0';
            break;
        }
    }
    if (i == 0)
    {
        printk("end_replace() not find / ");
        return -1;
    }
    strcat(str1, rep);
    return 0;
}


int test_file_exist(char *file_path)
{
    struct file *filp = NULL;
    int fsize;
    filp = filp_open(file_path, O_RDWR, 0);
    if (IS_ERR(filp))
    {
        fsize = 0;
    }
    else
    {
        fsize = filp->f_inode->i_size;
        filp_close(filp, NULL);
    }

    return fsize;
}

bool get_user_cmdline(char **argv, char *cmdline, int cmd_len)
{
    int i = 0, offset = 0;
    char tmp[260] = {0};
    if (unlikely(argv == NULL || cmdline == NULL || cmd_len <= 0))
        return false;

    memset(cmdline, 0, cmd_len);

    if (argv != NULL)
    {
        for (; i < 0x7FFFFFFF;)
        {
            const char __user *p;
            int ret = get_user(p, argv + i);
            if (ret || !p || IS_ERR(p))
                break;

            ret = copy_from_user(tmp, p, 256);
            if (ret < 256)
            {
                int tmp_len = strlen(tmp);
                if (offset + 1 + tmp_len > cmd_len)
                {
                    

                    break;
                }
                strncpy(cmdline + offset, tmp, tmp_len);
         
                offset += tmp_len;
                cmdline[offset] = ' ';
                offset++;
            }
            else
            {
                printk("err %s. copy_from_user failed. ret:%d.\n", __func__, ret);
            }
           
            ++i;
        }
    }
    if (cmdline[offset - 1] == ' ')
        cmdline[offset - 1] = 0;


    return true;
}


static int write_log(char *str, int str_size, char *filename)
{
    struct file *f;
    mm_segment_t fs;
    loff_t pos = 0;
    f = filp_open(filename, O_CREAT | O_RDWR, 0);
    if (IS_ERR(f))
    {
        printk("file create error, error code %ld\n", PTR_ERR(f));
        return -1;
    }
    fs = get_fs();
    set_fs(KERNEL_DS);
    kernel_write(f, str, str_size, &pos);

    filp_close(f, NULL);
    set_fs(fs);

    return 0;
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

int generate_and_init_vtpm(char *file_path)
{
    struct file *filp = NULL;
    int fsize;
    loff_t pos = 0;
    char *loc = NULL;
    char *find_loc = NULL;
    int len = 0, i = 0;
    char merge_path[PATH_LEN_S] = {'\0'};
    char vtpm_path[PATH_LEN_S] = {'\0'};
    char key_file[BUF_SIZE] = {'\0'};
    char log_file[BUF_SIZE] = {'\0'};
    int vtpm_path_size = 0;

    char passwd_file[BUF_SIZE] = {'\0'};

    filp = filp_open(file_path, O_RDONLY, 0);
    if (IS_ERR(filp))
    {
        printk(KERN_DEBUG "config file open error!");
        return -1;
    }
    else
    {
        fsize = filp->f_inode->i_size;
        loc = (char *)kmalloc(fsize, GFP_KERNEL);
        len = kernel_read(filp, loc, fsize, &pos);
        find_loc = strstr(loc, "root\":{\"path\":\"");

        while (find_loc[i + 15] != '}')
        {
            merge_path[i] = find_loc[i + 15];
            i++;
            if (i + 14 >= strlen(find_loc))
            {
                break;
            }
        }
        strncpy(vtpm_path, merge_path, sizeof(vtpm_path));
        end_replace(vtpm_path, "/diff/vtpm");
        new_dir(vtpm_path);
        vtpm_path_size = strlen(vtpm_path);
        strncpy(key_file, vtpm_path, strlen(vtpm_path));
        strcat(key_file, "/container_key");
        if (test_file_exist(key_file) != 0)
        {
            printk("It's not the first time to the container!");
            return 0;
        }
        get_aes_key(32, key_file, 0);
        tpm_encrypt(key_file, key_file, "0x81010002");
        strncpy(log_file, vtpm_path, strlen(vtpm_path));
        strcat(log_file, "/log");
        write_log(vtpm_path, vtpm_path_size, log_file);
        strncpy(passwd_file, vtpm_path, strlen(vtpm_path));
        strcat(passwd_file, "/passwd_key");
        get_aes_key(8, passwd_file, 1);

        filp_close(filp, NULL);
        kfree(loc);
        loc = NULL;
    }
    return 1;
}

static asmlinkage long my_execve(const struct pt_regs *regs)
{
    char __user *filename = (char *)regs->di;
    char user_filename[MAX_FILE_NAME_LEN] = {0};
    int len = 0, flag = 1;
    long copied = 0;
    char **argv = (char **)regs->si;
    int i = 0, j = 0;
    char file[BUF_SIZE] = {'\0'};
    char *image_names = NULL;
    char cmp_str[image_name_size] = {'\0'};
    char dirn[PATH_LEN] = {'\0'};
    int num = 0;

    mm_segment_t fs;
    loff_t pos;
    struct file *fp;
    int fsize;
    int kernel_read_num;
    char single_image[image_name_size] = {'\0'};
    char new_name[image_name_size] = {'\0'};
    int image_flag;
    char note[] = "TCEC installed, version 1.0.1 ";
    int flag_vtpm = 0;

    copied = strncpy_from_user(user_filename, filename, len);

    strncpy(file, Project_path, BUF_SIZE);
    strcat(file, "images.log");

    len = strnlen_user(filename, MAX_FILE_NAME_LEN);
    if (unlikely(len >= MAX_FILE_NAME_LEN))
    {
        len = MAX_FILE_NAME_LEN - 1;
    }

    get_user_cmdline(argv, user_filename, MAX_FILE_NAME_LEN);

    if (strstr(user_filename, "docker pull"))
    {
        if (strstr(user_filename, "logger"))
        {

            docker_images(file);

            fp = filp_open(file, O_RDONLY, 0);
            if (IS_ERR(fp))
            {
                printk("create file error/n");
                return -1;
            }

            fsize = fp->f_inode->i_size;
            fs = get_fs();
            set_fs(KERNEL_DS);
            pos = 0;

            image_names = (char *)kmalloc(fsize, GFP_KERNEL);
            kernel_read_num = kernel_read(fp, image_names, BUF_SIZE, &pos);

            for (i = 0; i < kernel_read_num; i++)
            {
                if (image_names[i] != '\n')
                {
                    single_image[j] = image_names[i];
                    j++;
                    continue;
                }

                strcpy(new_name, single_image);
                name_change(new_name);
                image_flag = file_exist(Base_value_path, new_name);

                if (image_flag == 0 && (strstr(user_filename, single_image) != NULL))
                {
                    query_loc(Base_value_path, single_image, new_name);
                    base_generate(Base_value_path, new_name);
                }

                memset(single_image, 0x00, sizeof(single_image));
                memset(new_name, 0x00, sizeof(single_image));
                j = 0;
            }

            filp_close(fp, NULL);
            set_fs(fs);

            kfree(image_names);
            image_names = NULL;
            printk("TCEC Complete the base value calculation!");
            printk("OK!");
        }
    }

    if (strstr(user_filename, "docker run") || strstr(user_filename, "docker create"))
    {
       

        if (!strstr(user_filename, "logger"))
        {
          
            parse(cmp_str, image_record_path, user_filename);

            if (strcmp(cmp_str, "none") != 0)
            {

                flag = measure(Project_path, Base_value_path, cmp_str);

                if (flag == 1)
                {
                    printk("measure complete,It's ok! Create a new secure container ^_^ !");
                    printString("Measure OK !\n");
                    return old_execve(regs);
                }
                else if (flag == -2)
                {
                    printk("measure complete,It's wrong! Can't crate a new secure container!");
                    printString("The base value is attacked.\n");
                    return -1;
                }

                else
                {
                    printk("measure complete,It's wrong! Can't crate a new secure container! ");
                    printString("Measure error,images may have been attack.\n");
                    return -1;
                }
            }
            else
            {
                printString("Please use docker pull to download the image firstly.");
                return -1;
            }
        }
    }

    if (strstr(user_filename, "tcec -v"))
    {
        if (!strstr(user_filename, "logger"))
        {
            printString(note);

        }
    }

    if (strstr(user_filename, "/var/run/docker/runtime-runc/moby"))
    {
       
        if (strstr(user_filename, "start"))
        {
            j = 0; 
            for (i = 0; i < strlen(user_filename); i++)
            {
                if (user_filename[i] == ' ' && user_filename[i - 1] == 'g' && user_filename[i - 2] == 'o' && user_filename[i - 3] == 'l' && user_filename[i - 4] == '-')
                {
                    num = 1;
                }
                if (num == 1 && (user_filename[i + 1] != ' '))
                {
                    dirn[j] = user_filename[i + 1];
                    j++;
                }
                if (user_filename[i + 1] == ' ')
                {
                    num = 0;
                }
            }
            if (j == 0)
            {
                printk(KERN_DEBUG "create vtpm error!");
                return -1; 
            }
            flag_vtpm = end_replace(dirn, "/config.json");
            if (flag_vtpm == -1)
            {
                printk(KERN_DEBUG "create vtpm error!");
                return -1;
            }
            generate_and_init_vtpm(dirn);
        }
    }

    return old_execve(regs);
}

static int __init monitor_init(void)
{

    sys_call_table = (sys_call_ptr_t *)kallsyms_lookup_name("sys_call_table");
    old_execve = sys_call_table[__NR_execve];

    write_cr0(read_cr0() & (~0x10000));
    sys_call_table[__NR_execve] = my_execve;
    write_cr0(read_cr0() | 0x10000);
    pr_info("%s inserted.\n", __func__);
    printk("-----------------------------------------------------------");
    printk("start..\n");
    printk("-----------------------------------------------------------");

    return 0;
}
static void __exit monitor_exit(void)
{
    write_cr0(read_cr0() & (~0x10000));
    sys_call_table[__NR_execve] = old_execve;
    write_cr0(read_cr0() | 0x10000);
    printk("-----------------------------------------------------------");
    pr_info("%s removed.\n", __func__);
}
module_init(monitor_init);
module_exit(monitor_exit);