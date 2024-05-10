#include <linux/module.h> 
#include <linux/kernel.h> 
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

static char *Project_path = "/home/TCEC/";
static char *Base_value_path = "/home/TCEC/base/";

#define PATH_MAX 512
#define MODULE_NAME_SIZE 100
#define MODULE_NUM 4
#define MAX_FILE_SIZE 3*1024 * 1024

module_param(Project_path, charp, S_IRUGO);
module_param(Base_value_path, charp, S_IRUGO);

extern int k_read(char *filename, char *buf, int len);

struct sdesc
{
    struct shash_desc shash;
    char ctx[];
};

static struct sdesc *init_sdesc(struct crypto_shash *alg)
{
    struct sdesc *sdesc;
    int size;

    size = sizeof(struct shash_desc) + crypto_shash_descsize(alg);
    sdesc = kmalloc(size, GFP_KERNEL);
    if (!sdesc)
        return ERR_PTR(-ENOMEM);
    sdesc->shash.tfm = alg;
    return sdesc;
}

static unsigned char *crypto_sha256_file(char *filename) 
{
    struct file *fp;
    int fsize;
    mm_segment_t fs;
    int size = 1024;
    unsigned char buf[size];
    loff_t pos = 0;
    int len;
    int ret;

    unsigned char *digest;
    struct crypto_shash *alg;
    char *hash_alg_name = "sha256";
    struct sdesc *sdesc;

    alg = crypto_alloc_shash(hash_alg_name, 0, 0);

    if (IS_ERR(alg))
    {
        pr_info("can't alloc alg %s\n", hash_alg_name);
        return NULL;
    }

    digest = kmalloc(256, GFP_KERNEL);
    if (digest < 0)
    {
        printk("digest malloc error!");
        return NULL;
    }

    sdesc = init_sdesc(alg);
    crypto_shash_init(&sdesc->shash);
    if (IS_ERR(sdesc))
    {
        pr_info("can't alloc sdesc\n");
        return NULL;
    }

    fp = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(fp))
    {
        printk("3 myread %ld\n", PTR_ERR(fp));
        printk("create file error/n");
        return "error";
    }

    fsize = fp->f_inode->i_size;

    fs = get_fs();
    set_fs(KERNEL_DS);
    while (pos < fsize)
    {

        len = kernel_read(fp, buf, size, &pos);
        ret = crypto_shash_update(&sdesc->shash, buf, len);
    }
    crypto_shash_final(&sdesc->shash, digest);
    filp_close(fp, NULL);
    set_fs(fs);

    crypto_free_shash(alg);
    return digest;
}

static int crypto_sha256(unsigned char *digest, char *filename, char *hash_buf) 
{
    struct file *fp;
    int fsize;
    mm_segment_t fs;
    int size = 1024;
    unsigned char buf[size];

    loff_t pos = 0;
    int len;
    int ret;

    struct crypto_shash *alg;
    char *hash_alg_name = "sha256";
    struct sdesc *sdesc;

    alg = crypto_alloc_shash(hash_alg_name, 0, 0);

    if (IS_ERR(alg))
    {
        pr_info("can't alloc alg %s\n", hash_alg_name);
        return PTR_ERR(alg);
    }

    sdesc = init_sdesc(alg);
    crypto_shash_init(&sdesc->shash);
    if (IS_ERR(sdesc))
    {
        pr_info("can't alloc sdesc\n");
        return PTR_ERR(sdesc);
    }

    fp = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(fp))
    {
        printk("3 myread %ld\n", PTR_ERR(fp));
        printk("create file error/n");
        return -1;
    }

    fsize = fp->f_inode->i_size;
    fs = get_fs();
    set_fs(KERNEL_DS);
    while (pos < fsize)
    {

        len = kernel_read(fp, buf, size, &pos);
        ret = crypto_shash_update(&sdesc->shash, buf, len);
    }
    crypto_shash_final(&sdesc->shash, digest);
    filp_close(fp, NULL);
    set_fs(fs);

    crypto_free_shash(alg);

    sprintf(hash_buf, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
            digest[0], digest[1], digest[2], digest[3], digest[4],
            digest[5], digest[6], digest[7], digest[8], digest[9], digest[10], digest[11], digest[12],
            digest[13], digest[14], digest[15], digest[16], digest[17], digest[18], digest[19], digest[20],
            digest[21], digest[22], digest[23], digest[24], digest[25], digest[26], digest[27], digest[28],
            digest[29], digest[30], digest[31]);

    return 0; 
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
        kfree(key);
        key=NULL;
        printk("malloc error!");
        return -ENOMEM;
    }
    src = kmalloc(16, GFP_KERNEL);
    if (src < 0)
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        printk("malloc error!");
        return -ENOMEM;
    }

    fp_key = filp_open(keyfile, O_RDONLY, 0);
    if (IS_ERR(fp_key))
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
        printk("read key error %ld\n", PTR_ERR(fp_key));
        return -1;
    }

    fp_enc = filp_open(enc_file, O_CREAT | O_RDWR, 0);
    if (IS_ERR(fp_enc))
    {
        printk("1 enc_file error %ld\n", PTR_ERR(fp_enc));
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
        return -1;
    }

    fp_raw = filp_open(raw_file, O_RDWR, 0);
    if (IS_ERR(fp_raw))
    {
        printk("raw_file error %ld\n", PTR_ERR(fp_raw));
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
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
        kfree(key);
        key=NULL;
        return -ENOMEM;
    }
    src = kmalloc(16, GFP_KERNEL);
    if (src < 0)
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        printk("malloc error!");
        return -ENOMEM;
    }

    fp_key = filp_open(keyfile, O_RDONLY, 0);
    if (IS_ERR(fp_key))
    {
        printk("read key error %ld\n", PTR_ERR(fp_key));
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
        return -1;
    }

    fp_enc = filp_open(enc_file, O_RDWR, 0);
    if (IS_ERR(fp_enc))
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
        printk("2 enc_file error %ld\n", PTR_ERR(fp_enc));
        return -1;
    }

    fp_dec = filp_open(dec_file, O_CREAT | O_RDWR, 0);
    if (IS_ERR(fp_dec))
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
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

static int get_aes_key(int byte, char *filename, int flag)
{
    struct file *f;
    unsigned char *pkey = NULL;
    unsigned char key_buf[100] = {'\0'};
    mm_segment_t fs;
    loff_t pos = 0;

    pkey = kmalloc(byte, GFP_KERNEL);
    if (pkey < 0)
    {
        printk("key allocate error!");
        return -ENOMEM;
    }
    get_random_bytes(pkey, byte);

    f = filp_open(filename, O_CREAT | O_RDWR, 0);
    if (IS_ERR(f))
    {
        printk("key create error, error code %ld\n", PTR_ERR(f));
        return -1;
    }
    fs = get_fs();
    set_fs(KERNEL_DS);
    if (flag == 0)
    {
        kernel_write(f, pkey, byte, &pos);
    }
    else
    {
        sprintf(key_buf, "%02x%02x%02x%02x%02x%02x%02x%02x", pkey[0], pkey[1], pkey[2], pkey[3], pkey[4], pkey[5], pkey[6], pkey[7]);
        kernel_write(f, key_buf, strlen(key_buf), &pos);
    }
    filp_close(f, NULL);
    set_fs(fs);
    kfree(pkey);
    pkey = NULL;

    return 0;
}

int buffer_read(char *raw_buf, char *src, int size, int pos)
{
    int i = 0, j = 0;
    while (i < pos)
    {
        raw_buf++;
        i++;
    }
    for (j = 0; j < size; j++)
    {
        *src = *raw_buf;
        src++;
        raw_buf++;
    }

    return pos + size;
}

int buffer_write(char *buf_enc, char *dst, int size, int pos)
{
    int i = 0, j = 0;
    while (i < pos)
    {
        buf_enc++;
        i++;
    }
    for (j = 0; j < size; j++)
    {
        *buf_enc = *dst;
        dst++;
        buf_enc++;
    }
    return pos + size;
}

static int aes_encrypt_buffer(char *enc_file, char *keyfile, char *raw_buf, int buf_size, int buf_len, int pos)
{
    struct crypto_cipher *tfm = NULL;
    unsigned char *key = NULL, *src = NULL, *dst = NULL;
    struct file *fp_key = NULL, *fp_enc = NULL;
    int len = 0;
    char *buf_enc = NULL;
    loff_t key_pos = 0, raw_pos = 0, enc_pos = 0, tmp_p = pos;
    mm_segment_t fs;
    //*

    fs = get_fs();
    set_fs(KERNEL_DS);

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
        kfree(key);
        key=NULL;
        return -ENOMEM;
    }
    src = kmalloc(16, GFP_KERNEL);
    if (src < 0)
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        printk("malloc error!");
        return -ENOMEM;
    }

    buf_enc = (char *)kmalloc(buf_size + 16, GFP_KERNEL);
    if (buf_enc < 0)
    {
        kfree(key);
        key=NULL;
        kfree(dst);
        dst=NULL;
        kfree(src);
        src=NULL;
        printk("malloc error!");
        return -ENOMEM;
    }
    memset(buf_enc, 0, buf_size + 16);

    fp_enc = filp_open(enc_file, O_CREAT | O_RDWR, 0);
    if (IS_ERR(fp_enc))
    {
        kfree(buf_enc);
        buf_enc = NULL;
        kfree(key);
        key = NULL;
        kfree(dst);
        dst = NULL;
        kfree(src);
        src = NULL;
        printk("3 enc_file error %ld\n", PTR_ERR(fp_enc));
        return -1;
    }

    fp_key = filp_open(keyfile, O_RDONLY, 0);

    if (IS_ERR(fp_key))
    {
        printk("read key error %ld\n", PTR_ERR(fp_key));
        kfree(buf_enc);
        buf_enc = NULL;
        kfree(key);
        key = NULL;
        kfree(dst);
        dst = NULL;
        kfree(src);
        src = NULL;
        return -1;
    }

    len = kernel_read(fp_key, key, 32, &key_pos);
    crypto_cipher_setkey(tfm, key, 32);

    while (raw_pos < buf_len)
    {
        if (buf_len - raw_pos < 16)
        {
            memset(src, 0, 16);
            raw_pos = buffer_read(raw_buf, src, buf_len - raw_pos, raw_pos);
        }
        else
        {
            memset(src, 0, 16);
            raw_pos = buffer_read(raw_buf, src, 16, raw_pos);
        }
        crypto_cipher_encrypt_one(tfm, dst, src);
        enc_pos = buffer_write(buf_enc, dst, 16, enc_pos);

    }

    if (fp_enc && buf_enc && enc_pos)
    {
        kernel_write(fp_enc, buf_enc, enc_pos, &tmp_p);
        filp_close(fp_enc, NULL);
    }
    if (tfm)
        crypto_free_cipher(tfm);
    if (fp_key)
        filp_close(fp_key, NULL);

    set_fs(fs);

    kfree(buf_enc);
    buf_enc = NULL;
    kfree(key);
    key = NULL;
    kfree(dst);
    dst = NULL;
    kfree(src);
    src = NULL;

    return 0;
}

static int aes_decrypt_buffer(char *enc_file, char *keyfile, char *buf_dec)
{
    struct crypto_cipher *tfm;
    unsigned char *key, *src, *dst;
    struct file *fp_key, *fp_enc;
    int len;
    loff_t key_pos = 0, enc_pos = 0;
    int dec_pos = 0;
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
        kfree(key);
        key = NULL;
        printk("malloc error!");
        return -ENOMEM;
    }
    src = kmalloc(16, GFP_KERNEL);
    if (src < 0)
    {
        kfree(key);
        key = NULL;
        kfree(dst);
        dst=NULL;
        printk("malloc error!");
        return -ENOMEM;
    }

    fp_key = filp_open(keyfile, O_RDONLY, 0);
    if (IS_ERR(fp_key))
    {
        kfree(key);
        key = NULL;
        kfree(dst);
        dst=NULL;
        printk("read key error %ld\n", PTR_ERR(fp_key));
        return -1;
    }

    fp_enc = filp_open(enc_file, O_RDWR, 0);
    if (IS_ERR(fp_enc))
    {
        kfree(key);
        key = NULL;
        kfree(dst);
        dst=NULL;
        printk("4 open %s error %ld\n",enc_file, PTR_ERR(fp_enc));
        return -1;
    }

    fsize = fp_enc->f_inode->i_size;

    if (fsize % 16 != 0)
    {
        printk("ciphertext has been changed!");
        return -1;
    }

    fs = get_fs();
    set_fs(KERNEL_DS);

    len = kernel_read(fp_key, key, 32, &key_pos);
    crypto_cipher_setkey(tfm, key, 32);

    while (enc_pos < fsize)
    {

        memset(src, 0, 16);
        len = kernel_read(fp_enc, src, 16, &enc_pos);

        memset(dst, 0, 16);
        crypto_cipher_decrypt_one(tfm, dst, src);
        // printk("decpos %d", dec_pos);
        if (fsize - dec_pos <= 16)
        {
            zero_count = (int)dst[15];
        }
 
        dec_pos = buffer_write(buf_dec, dst, 16, dec_pos); 
    }
    crypto_free_cipher(tfm);
    filp_close(fp_key, NULL);
    filp_close(fp_enc, NULL);
    set_fs(fs);

    kfree(key);
    key = NULL;
    kfree(dst);
    dst = NULL;
    kfree(src);
    src = NULL;

    return dec_pos;
}

int get_all_file_name(char *dir_name, char *tmp_file, int size)
{
    int result = 0;
    char cmdPath[] = "/bin/bash";
    char *cmdArgv[] = {cmdPath, "-c", "find /var/lib/docker/overlay2/bf32226e827d79b9b2a91add47ef371f5dbed652b62eda2595c80da018542f38/diff -type f > file.txt", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};

    char *cmdString = (char *)kmalloc(size, GFP_KERNEL);

    strcpy(cmdString, "find ");
    strcat(cmdString, dir_name);
    strcat(cmdString, " -type f > ");
    strcat(cmdString, tmp_file);

    cmdArgv[2] = cmdString;
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    return result;
}

struct sdesc *hash_crypto(struct sdesc *sdesc, char *filename)
{
    struct file *fp;
    int fsize;
    mm_segment_t fs;
    int size = 1024;
    unsigned char buf[size];

    loff_t pos = 0;
    int len;
    int ret;

    fp = filp_open(filename, O_RDONLY, 0);

    if (IS_ERR(fp))
    {
        printk("3 myread %ld\n", PTR_ERR(fp));
        printk("create file error/n");
        return NULL;
    }

    fsize = fp->f_inode->i_size;
    fs = get_fs();
    set_fs(KERNEL_DS);
    while (pos < fsize)
    {
        len = kernel_read(fp, buf, size, &pos);
        ret = crypto_shash_update(&sdesc->shash, buf, len);
    }

    filp_close(fp, NULL);
    set_fs(fs);
    return sdesc;
}

int deal_path_name(char *path_to_record_file, int len)
{
    char *loc = NULL;
    char tmp[PATH_MAX] = {'\0'};
    int i = 0;
    loc = strstr(path_to_record_file, "diff/");
    if (loc == NULL)
    {
        return -1;
    }
    while (loc[i] != '\0')
    {
        tmp[i] = loc[i + 5];
    }
    memset(path_to_record_file, '\0', len);
    strcpy(path_to_record_file, tmp);
    return 0;
}

static int hash_directory(char *dir_name, char *hash_buf, char *path_to_record_file_arg, unsigned char *digest, int flag)
{
    struct file *fp = NULL;
    mm_segment_t fs;
    loff_t pos = 0;
    char file_path[PATH_MAX] = {'\0'};
    int i = 0, j = 0, fsize = 0;
    char *file_content = NULL;
    char path_to_record_file[PATH_MAX] = {'\0'};

   
    struct sdesc *sdesc;
    struct crypto_shash *alg;
    char *hash_alg_name = "sha256";

    strcpy(path_to_record_file, path_to_record_file_arg);

    memset(digest, 0, 32); 

    if (flag == 1)
    {
        get_all_file_name(dir_name, path_to_record_file, PATH_MAX); 
    }

    fp = filp_open(path_to_record_file, O_RDONLY, 0);
    if (IS_ERR(fp))
    {
        printk("11111");
        printk("3 myread %ld\n", PTR_ERR(fp));
        printk("create file error/n");
        return -1;
    }

    alg = crypto_alloc_shash(hash_alg_name, 0, 0);

    if (IS_ERR(alg))
    {
        pr_info("can't alloc alg %s\n", hash_alg_name);
        return PTR_ERR(alg);
    }

    sdesc = init_sdesc(alg);
    crypto_shash_init(&sdesc->shash);
    if (IS_ERR(sdesc))
    {
        pr_info("can't alloc sdesc\n");
        return PTR_ERR(sdesc);
    }

    fsize = fp->f_inode->i_size;

    pos = fsize + 1;

    fs = get_fs();
    set_fs(KERNEL_DS);

    kernel_write(fp, "\n", 1, &pos);

    if (fsize < MAX_FILE_SIZE)
    {
        file_content = (char *)vmalloc(fsize);
        memset(file_content, 0, fsize);
        if (file_content == NULL)
        {
            printk("file_content vmalloc failed! \n");
            return -1;
        }
        pos = 0;
        kernel_read(fp, file_content, fsize, &pos);
        for (i = 0; i < fsize; i++)
        {
            if (file_content[i] != '\n')
            {
                file_path[j] = file_content[i];
                j++;
            }
            else
            {
                sdesc = hash_crypto(sdesc, file_path);
                j = 0;
                memset(file_path, '\0', PATH_MAX);
            }
        }

        crypto_shash_final(&sdesc->shash, digest);
        crypto_free_shash(alg);

        sprintf(hash_buf, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
                digest[0], digest[1], digest[2], digest[3], digest[4],
                digest[5], digest[6], digest[7], digest[8], digest[9], digest[10], digest[11], digest[12],
                digest[13], digest[14], digest[15], digest[16], digest[17], digest[18], digest[19], digest[20],
                digest[21], digest[22], digest[23], digest[24], digest[25], digest[26], digest[27], digest[28],
                digest[29], digest[30], digest[31]);
    }
    else
    {
        printk("dir is too large");
        return -1;
    }

    if (file_content)
    {
        vfree(file_content);
        file_content = NULL;
    }
    filp_close(fp, NULL);
    set_fs(fs);

    return 0;
}

int vpcr_extend(unsigned char *old_hash, unsigned char *new_hash, unsigned char *hash_buf, int size)
{

    unsigned char buf1[size];
    unsigned char buf2[size];

    int ret;

    struct crypto_shash *alg;
    char *hash_alg_name = "sha256";
    struct sdesc *sdesc;

    alg = crypto_alloc_shash(hash_alg_name, 0, 0);

    if (IS_ERR(alg))
    {
        pr_info("can't alloc alg %s\n", hash_alg_name);
        return PTR_ERR(alg);
    }

    sdesc = init_sdesc(alg);
    crypto_shash_init(&sdesc->shash);
    if (IS_ERR(sdesc))
    {
        pr_info("can't alloc sdesc\n");
        return PTR_ERR(sdesc);
    }
    memset(buf1, 0, size);
    memset(buf2, 0, size);
    memcpy(buf1, old_hash, size);
    memcpy(buf2, new_hash, size);

    ret = crypto_shash_update(&sdesc->shash, buf1, size);
    ret = crypto_shash_update(&sdesc->shash, buf2, size);

    memset(old_hash, 0, 32);

    crypto_shash_final(&sdesc->shash, old_hash);

    crypto_free_shash(alg);

    memset(hash_buf, 0, 65);

    sprintf(hash_buf, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
            old_hash[0], old_hash[1], old_hash[2], old_hash[3], old_hash[4],
            old_hash[5], old_hash[6], old_hash[7], old_hash[8], old_hash[9], old_hash[10], old_hash[11], old_hash[12],
            old_hash[13], old_hash[14], old_hash[15], old_hash[16], old_hash[17], old_hash[18], old_hash[19], old_hash[20],
            old_hash[21], old_hash[22], old_hash[23], old_hash[24], old_hash[25], old_hash[26], old_hash[27], old_hash[28],
            old_hash[29], old_hash[30], old_hash[31]);

    return 0; 
}

// rm
int con_rm(char *raw_file)
{
    int result = 0;
    char cmdPath[] = "/bin/rm";
    char *cmdArgv[] = {cmdPath, "-rf", raw_file, NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);

    return result;
}
static int parse_code_base(char *file, char *code_name, char *hash_value, int hash_len)
{
    struct file *fp;
    mm_segment_t fs;
    loff_t pos;
    char *str_loc = NULL;
    char *buf = NULL;
    int i = 0, fsize = 0, len = 0;

    fp = filp_open(file, O_RDONLY, 0);

    if (IS_ERR(fp))
    {
        printk("create file error/n");
        return -1;
    }

    fsize = fp->f_inode->i_size;
    buf = (char *)kmalloc(fsize, GFP_KERNEL);

    fs = get_fs();
    set_fs(KERNEL_DS);
    pos = 0;

    kernel_read(fp, buf, fsize, &pos);
    str_loc = strstr(buf, code_name);
    if (str_loc == NULL)
    {
        kfree(buf);
        buf = NULL;
        filp_close(fp, NULL);
        set_fs(fs);
        return -1;
    }
    len = strlen(code_name) + 1;
    for (i = 0; i < hash_len; i++)
    {
        hash_value[i] = str_loc[i + len];
    }

    kfree(buf);
    buf = NULL;
    filp_close(fp, NULL);
    set_fs(fs);
    return 0;
}

int hash_compare(char *hash1, char *hash2)
{
    int i = 0;
    for (i = 0; i < 64; i++)
    {
        if (hash1[i] != hash2[i])
        {
            return -1;
        }
    }
    return 0;
}

static int tpm_aes_decrypt(char *raw_file, char *out_file, char *key)
{
    int result = 0;
    char cmdPath[] = "/bin/tpm2_encryptdecrypt";
    char *cmdArgv[] = {cmdPath, "-d", "-c", key, "-o", out_file, raw_file, "-T", "device:/dev/tpmrm0", NULL};
    char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};
    result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);
    printk(KERN_DEBUG "aes decrypt complete! The result of call_usermodehelper is %d\n", result);
    return result;
}

static int validate_module_self(char *kernel_module_path, char *check_path, char *kernel_name)
{
    unsigned char *digest = (unsigned char *)kmalloc(32, GFP_KERNEL);
    char buf[100] = {'\0'};
    char hash_res[100] = {'\0'};

    memset(digest, 0, 32);
    crypto_sha256(digest, kernel_module_path, hash_res);
    if (-1 == parse_code_base(check_path, kernel_name, buf, 64))
        return -1;

    if (hash_compare(hash_res, buf) == 0)
        return 1;
    else
        return -1;
}

static int __init crypto_init(void)
{

    char check_path[PATH_MAX] = {'\0'};
    char kernel_module_path[PATH_MAX] = {'\0'};
    char kernel_module[MODULE_NUM][MODULE_NAME_SIZE] = {"crypto.ko", "image_measure.ko", "operation_monitor.ko", "communications.ko"};
    char check_path_dec[PATH_MAX] = {'\0'};
    int i = 0, module_meas_flag = 1, aes_decrypt_flag = 0;
    char check_bp[PATH_MAX] = {'\0'};

    sprintf(check_bp, "%s%s", Base_value_path, ".check1");
    sprintf(check_path, "%s%s", Base_value_path, "check");
    sprintf(check_path_dec, "%s.tmp", check_path);


    aes_decrypt_flag = tpm_aes_decrypt(check_path, check_path_dec, "0x81010003");
    if (aes_decrypt_flag != 0)
    {
        memset(check_path_dec, 0, PATH_MAX);
        strcpy(check_path_dec, check_bp);
    }
    for (i = 0; i < MODULE_NUM; i++)
    {
        sprintf(kernel_module_path, "%s/%s", Project_path, kernel_module[i]);
        if (validate_module_self(kernel_module_path, check_path_dec, kernel_module[i]) == -1)
        {
            module_meas_flag = -1;
            break;
        }
        memset(kernel_module_path, 0, PATH_MAX);
    }

    if (module_meas_flag == 1)
    {
        printk("All modules successfully completed the verification and passed the verification.");
        if (aes_decrypt_flag == 0)
            con_rm(check_path_dec);
        return 0;
    }
    else
    {
        printk("The %s module verificate error!", kernel_module[i]);
        if (aes_decrypt_flag == 0)
            con_rm(check_path_dec);
        return -1;
    }

    return 0;
}

static void __exit crypto_exit(void)
{
    pr_info("%s module removed.\n", __func__);
}

EXPORT_SYMBOL(hash_directory);
EXPORT_SYMBOL(get_aes_key);
EXPORT_SYMBOL(aes_decrypt_buffer);
EXPORT_SYMBOL(aes_encrypt_buffer);
EXPORT_SYMBOL(crypto_sha256);
EXPORT_SYMBOL(aes_encrypt);
EXPORT_SYMBOL(aes_decrypt);
EXPORT_SYMBOL(parse_code_base);
EXPORT_SYMBOL(crypto_sha256_file);
EXPORT_SYMBOL(validate_module_self);
EXPORT_SYMBOL(tpm_aes_decrypt);
module_init(crypto_init);
module_exit(crypto_exit);
