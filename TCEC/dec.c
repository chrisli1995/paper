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
#include <crypto/sha256_base.h>

MODULE_LICENSE("GPL");
static dev_t dev_number;
static struct cdev *my_cdev;
static struct class *my_device;
static int read_num = 0;


static int char_open(struct inode *, struct file *);
static int char_release(struct inode *, struct file *);
static ssize_t char_write(struct file *, const char *, size_t, loff_t *);



struct file_operations cdev_ops = {.open = char_open,
                                   .release = char_release,
                                   .write = char_write,
                                   .owner = THIS_MODULE};

static char * proj_path = "/home/TCEC/";

module_param(proj_path,charp,S_IRUGO);

static int module_init_function(void)
{
    dev_t dev;
    int error = alloc_chrdev_region(&dev, 3, 6, "ccdrive");

    if (error < 0)
    {
      printk("allocate device number fail");
    }
    else
    {
      dev_number = dev;
    }
   
    my_cdev = cdev_alloc();
    my_cdev->ops = &cdev_ops;
    my_cdev->owner = THIS_MODULE;
    cdev_init(my_cdev, &cdev_ops);
    cdev_add(my_cdev, dev, 6);
    
    my_device = class_create(THIS_MODULE, "ccdrive");
    device_create(my_device, NULL, dev, NULL, "dec_device");

  return 0;
}

static void module_exit_function(void)
{

  device_destroy(my_device, dev_number);
  class_unregister(my_device);
  class_destroy(my_device);
 
  cdev_del(my_cdev);

  unregister_chrdev_region(dev_number, 6);
}


static int char_open(struct inode *inode, struct file *file)
{
  
  return 0;
}

static int char_release(struct inode *inode, struct file *file)
{
  
  return 0;
}

char *my_strcpy(char *dst, char const *src)
{
  while (*src != '\0')
    *dst++ = *src++;
  *dst = '\0';
  return dst;
}

int parse_receive_string(char *buf, int flag, char *res)
{
  char *const delim = ":";
  int count = 0;
  char *token = NULL;
  while (count != flag)
  {
    token = strsep(&buf, delim);
    count = count + 1;
  }

  if (token != NULL)
    my_strcpy(res, token);
  else
    return -1;

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

// TPM decrypt
int tpm_decrypt(char *raw_file, char *out_file, char *key)
{
    int result = 0;
    char cmdPath[] = "/bin/tpm2_rsadecrypt";
    char *cmdArgv[] = {cmdPath, "-c", key, "-o", out_file, raw_file, "-T", "device:/dev/tpmrm0", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);
    // printk(KERN_DEBUG "decrypt complete! The result of call_usermodehelper is %d\n", result);
    return result;
}

int con_rm(char *raw_file)
{
    int result = 0;
    char cmdPath[] = "/bin/rm";
    char *cmdArgv[] = {cmdPath, "-rf", raw_file, NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);
    // printk(KERN_DEBUG"remove complete! The result of call_usermodehelper is %d\n",result);
    return result;
}

static int aes_encrypt(char *keyfile, char *raw_file, char *enc_file)
{
    struct crypto_cipher *tfm;
    unsigned char *key, *src, *dst;
    struct file *fp_key, *fp_raw, *fp_enc;
    int len;
    loff_t key_pos = 0, raw_pos = 0, enc_pos = 0;
    int fsize;
    mm_segment_t fs;

    tfm = crypto_alloc_cipher("aes", 0, 32);
    if (IS_ERR(tfm))
    {
        
        return PTR_ERR(tfm);
    }

    key = kmalloc(32, GFP_KERNEL);
    memset(key, 0, 32);
    if (key < 0)
    {
        printk("key error!");
        return -ENOMEM;
    }
    dst = kmalloc(16, GFP_KERNEL);
    if (dst < 0)
    {
        printk("malloc error!");
        return -ENOMEM;
    }
    src = kmalloc(16, GFP_KERNEL);
    if (src < 0)
    {
        printk("malloc error!");
        return -ENOMEM;
    }

    fp_key = filp_open(keyfile, O_RDONLY, 0);
    if (IS_ERR(fp_key))
    {
        printk("read key error %ld\n", PTR_ERR(fp_key));
        return -1;
    }

    fp_enc = filp_open(enc_file, O_CREAT | O_RDWR, 0);
    if (IS_ERR(fp_enc))
    {
        printk("enc_file error %ld\n", PTR_ERR(fp_enc));
        return -1;
    }

    fp_raw = filp_open(raw_file, O_RDWR, 0);
    if (IS_ERR(fp_raw))
    {
        printk("raw_file error %ld\n", PTR_ERR(fp_raw));
        return -1;
    }

    fsize = fp_raw->f_inode->i_size;

    fs = get_fs();
    set_fs(KERNEL_DS);

    len = kernel_read(fp_key, key, 32, &key_pos);
    crypto_cipher_setkey(tfm, key, 32);

    while (raw_pos < fsize)
    {
        if (fsize - raw_pos < 16)
        {
            memset(src, 16 - (fsize - raw_pos), 16);
        }
        else
        {
            memset(src, 0, 16);
        }
        len = kernel_read(fp_raw, src, 16, &raw_pos);
        crypto_cipher_encrypt_one(tfm, dst, src);
        kernel_write(fp_enc, dst, 16, &enc_pos);
    }
    crypto_free_cipher(tfm);
    filp_close(fp_key, NULL);
    filp_close(fp_raw, NULL);
    filp_close(fp_enc, NULL);
    set_fs(fs);

    return 0;
}

static int aes_decrypt(char *keyfile, char *enc_file, char *dec_file)
{
    struct crypto_cipher *tfm;
    unsigned char *key, *src, *dst;
    struct file *fp_key, *fp_dec, *fp_enc;
    int len;
    loff_t key_pos = 0, dec_pos = 0, enc_pos = 0;
    int fsize;
    mm_segment_t fs;
    int zero_count = 0;

    tfm = crypto_alloc_cipher("aes", 0, 32);
    if (IS_ERR(tfm))
    {
        return PTR_ERR(tfm);
    }

    key = kmalloc(32, GFP_KERNEL);
    if (key < 0)
    {
        printk("key error!");
        return -ENOMEM;
    }
    memset(key, 0, 32);

    dst = kmalloc(16, GFP_KERNEL);
    if (dst < 0)
    {
        printk("malloc error!");
        return -ENOMEM;
    }
    src = kmalloc(16, GFP_KERNEL);
    if (src < 0)
    {
        printk("malloc error!");
        return -ENOMEM;
    }

    fp_key = filp_open(keyfile, O_RDONLY, 0);
    if (IS_ERR(fp_key))
    {
        printk("read key error %ld\n", PTR_ERR(fp_key));
        return -1;
    }

    fp_enc = filp_open(enc_file, O_RDWR, 0);
    if (IS_ERR(fp_enc))
    {
        printk("enc_file error %ld\n", PTR_ERR(fp_enc));
        return -1;
    }

    fp_dec = filp_open(dec_file, O_CREAT | O_RDWR, 0);
    if (IS_ERR(fp_dec))
    {
        printk("dec_file error %ld\n", PTR_ERR(fp_enc));
        return -1;
    }

    fsize = fp_enc->f_inode->i_size;

    fs = get_fs();
    set_fs(KERNEL_DS);

    len = kernel_read(fp_key, key, 32, &key_pos);
    crypto_cipher_setkey(tfm, key, 32);

    while (enc_pos < fsize)
    {

        len = kernel_read(fp_enc, src, 16, &enc_pos);
        memset(dst, 0, 16);
        crypto_cipher_decrypt_one(tfm, dst, src);
        if (fsize - dec_pos <= 16)
        {
          zero_count = (int)dst[15];
        }
        kernel_write(fp_dec, dst, 16 - zero_count, &dec_pos);
    }
    crypto_free_cipher(tfm);
    filp_close(fp_key, NULL);
    filp_close(fp_dec, NULL);
    filp_close(fp_enc, NULL);
    set_fs(fs);

    return 0;
}


static ssize_t char_write(struct file *file, const char *str, size_t size, loff_t *offset)
{
 
  size_t maxdata = 400, copied;
  int index = 0;
  char receive_data[maxdata];
  char receive_data_cp[maxdata];
  char mode[5] = {'\0'};
  char input_path[512] = {'\0'};
  char output_path[512] = {'\0'};
  char key[400] = {'\0'};
  char tmp_key[400] = {'\0'};
  sprintf(tmp_key,"%s%s",proj_path,"kk");
  int meas_flag=0;
  
  if (size < maxdata)
  {
    maxdata = size;
  }

  copied = _copy_from_user(receive_data, str, maxdata);
  if (copied == 0)
  {

    receive_data[maxdata] = '\0';

  }

  
    strcpy(receive_data_cp, receive_data);
    parse_receive_string(receive_data_cp, 1, mode);

    strcpy(receive_data_cp, receive_data);
    parse_receive_string(receive_data_cp, 2, input_path);

    strcpy(receive_data_cp, receive_data);
    parse_receive_string(receive_data_cp, 3, output_path);

    strcpy(receive_data_cp, receive_data);
    parse_receive_string(receive_data_cp, 4, key);

    if(strstr(mode,"dec"))
    {
        tpm_decrypt(key, tmp_key, "0x81010002");
        aes_decrypt(tmp_key, input_path,output_path);
        con_rm(tmp_key);
    }
    if(strstr(mode,"enc"))
    {
        tpm_decrypt(key, tmp_key, "0x81010002");
        aes_encrypt(tmp_key, input_path,output_path);
        con_rm(tmp_key);
    }



  return meas_flag;
}


module_init(module_init_function);
module_exit(module_exit_function);
