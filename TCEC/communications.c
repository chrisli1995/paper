#include <asm-generic/errno-base.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <stddef.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/module.h>
#include <linux/crypto.h>
#include <crypto/hash.h>
#include <crypto/skcipher.h>
#include <linux/cred.h>
#include <linux/scatterlist.h>
#include <crypto/internal/hash.h>
#include <linux/init.h>
#include <linux/types.h>
#include <crypto/sha256_base.h>
#include <linux/string.h>
#include <linux/time.h>
#include <linux/timer.h>
#include <linux/timex.h>
#include <linux/rtc.h>
#include <linux/sched/signal.h>
#include <linux/mm.h>

#define DIR_SIZE 200
#define vTPM_PATH "/vtpm/"
#define SIZE 40
#define HASH_SIZE 100
#define FILE_PATH_MAX 200
#define LINE_LEN 480
#define container_id_size 65
#define PATH_LEN 512
#define base_value_mode '1'
#define measure_mode '2'

static char *Project_path = "/home/TCEC/";
static char *Base_value_path = "/home/TCEC/base/";

module_param(Project_path, charp, S_IRUGO);
module_param(Base_value_path, charp, S_IRUGO);

MODULE_LICENSE("GPL");
static dev_t dev_number;
static struct cdev *my_cdev;
static struct class *my_device;
static int read_num = 0;
static char agent_base_hash[HASH_SIZE] = {'\0'};

extern int get_aes_key(int byte, char *filename, int flag);
extern int aes_decrypt_buffer(char *enc_file, char *keyfile, char *buf_dec);
extern int aes_encrypt_buffer(char *enc_file, char *keyfile, char *raw_buf, int buf_size, int buf_len, int pos);
extern int crypto_sha256(unsigned char *digest, char *filename, char *hash_buf);
extern int hash_directory(char *dir_name, char *hash_buf, char *path_to_record_file, unsigned char *digest, int flag);
extern int validate_module_self(char *kernel_module_path, char *check_path, char *kernel_name);
extern int parse_code_base(char *file, char *code_name, char *hash_value, int hash_len);
extern int tpm_aes_decrypt(char *raw_file, char *out_file, char *key);

static int char_open(struct inode *, struct file *);
static int char_release(struct inode *, struct file *);
static ssize_t char_read(struct file *, char *, size_t, loff_t *);
static ssize_t char_write(struct file *, const char *, size_t, loff_t *);
static int hash_compare(char *base, char *meas);


struct file_operations cdev_ops = {.open = char_open,
                                   .release = char_release,
                                   .read = char_read,
                                   .write = char_write,
                                   .owner = THIS_MODULE};

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

int log_file_exist(char *file_path)
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

char *my_strcpy(char *dst, char const *src)
{
  while (*src != '\0')
    *dst++ = *src++;
  *dst = '\0';
  return dst;
}


int parse_receive_string(char *buf, int flag, char *res)
{
  char *const delim = "-";
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

static int module_init_function(void)
{
 
  char check_path[PATH_LEN] = {'\0'};
  char check_path_dec[PATH_LEN] = {'\0'};
  char kernel_module_path[PATH_LEN] = {'\0'};
  char check_bp[PATH_LEN] = {'\0'};
  int aes_decrypt_flag = 0;

  dev_t dev;
  int error = 0;

  error = alloc_chrdev_region(&dev, 0, 2, "cdrive");
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
  cdev_add(my_cdev, dev, 1);
  
  my_device = class_create(THIS_MODULE, "cdrive");
  device_create(my_device, NULL, dev, NULL, "vtpm-dev");

  sprintf(check_path, "%s%s", Base_value_path, "check");
  sprintf(check_path_dec, "%s.tmp", check_path);

 
  aes_decrypt_flag = tpm_aes_decrypt(check_path, check_path_dec, "0x81010003");
  if (aes_decrypt_flag != 0)
  {
    sprintf(check_bp, "%s%s", Base_value_path, ".check1");
    memset(check_path_dec, 0, PATH_LEN);
    strcpy(check_path_dec, check_bp);
    printk("TPM decrypt error!");
  }
  parse_code_base(check_path_dec, "agent", agent_base_hash, 64);

  if (aes_decrypt_flag == 0)
    con_rm(check_path_dec);

  return 0;
}

static void module_exit_function(void)
{
 
  device_destroy(my_device, dev_number);
  class_unregister(my_device);
  class_destroy(my_device);
  cdev_del(my_cdev);
  unregister_chrdev_region(dev_number, 2);
}


static int char_open(struct inode *inode, struct file *file)
{
  printk("vtpm device is open!");
  return 0;
}

static int char_release(struct inode *inode, struct file *file)
{
  printk("divice is closed!");
  return 0;
}

static ssize_t char_read(struct file *file, char *str, size_t size,
                         loff_t *offset)
{
  char *data = "I haven't recieve any data!\n";
  size_t datalen = strlen(data);
  char *data2 = "I have user data!\n";
  size_t datalen2 = strlen(data2);

  if (!read_num)
  {
    if (size > datalen)
    {
      size = datalen;
    }
    if (_copy_to_user(str, data, size))
    {
      return -EFAULT;
    }
  }
  else
  {
    if (size > datalen2)
    {
      size = datalen2;
    }
    if (_copy_to_user(str, data2, size))
    {
      return -EFAULT;
    }
  }
  printk("device is being read!");
  return size;
}


int parse_config_file(char *buf, char *delim_string, char *file_path)
{
  char *split_place = NULL;
  int delim_string_len = strlen(delim_string);
  int string_len = 0;
  int flag = 0;
  char ttmp[FILE_PATH_MAX] = {'\0'};

  strcpy(ttmp, buf);
  split_place = strstr(ttmp, delim_string);

  if (NULL != split_place)
  {

    while (ttmp[delim_string_len + string_len] != '\0')
    {
      file_path[string_len] = split_place[delim_string_len + string_len];
      string_len++;
    }
    file_path[string_len] = '\0';
    flag = 1;
    string_len = 0;
  }
  return flag;
}

int write_in_file(char *base_file_path, int index, char *file_path, unsigned char *hash, unsigned char *hash_extend, char *container_id, loff_t pos)
{
  char write_buf[500];
  struct file *fp_w = NULL;
  char time[HASH_SIZE] = {'\0'};

  struct timex txc;
  struct rtc_time tm;
  mm_segment_t fs;

  do_gettimeofday(&(txc.time)); 

  txc.time.tv_sec -= sys_tz.tz_minuteswest * 60;
  rtc_time_to_tm(txc.time.tv_sec, &tm);
  sprintf(time, "UTC time :%d-%d-%d %d:%d:%d \n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

  fp_w = filp_open(base_file_path, O_CREAT | O_RDWR, 0);
  if (IS_ERR(fp_w))
  {
    printk("bs error code %ld\n", PTR_ERR(fp_w));
    printk("create file error\n");
    return -1;
  }

  fs = get_fs();
  set_fs(KERNEL_DS);
  if (index == 1)
  {
    kernel_write(fp_w, time, strlen(time), &pos);
    kernel_write(fp_w, "Index,File path,Now Hash Value,Extend Hash Value,Container Id\n", strlen("Index,File path,Now Hash Value,Extend Hash Value,Container Id\n"), &pos);
  }
  sprintf(write_buf, "%d,%s,%s,%s,%s\n", index, file_path, hash, hash_extend, container_id);
  kernel_write(fp_w, write_buf, strlen(write_buf), &pos);
  filp_close(fp_w, NULL);
  set_fs(fs);

  return pos;
}

int write_to_buffer(int flag, char *buf, int index, char *file_path, unsigned char *hash, unsigned char *hash_extend, char *container_id)
{
  char write_buf[500] = {'\0'};
  char time[HASH_SIZE] = {'\0'};

  struct timex txc;
  struct rtc_time tm;

  do_gettimeofday(&(txc.time)); 

  txc.time.tv_sec -= sys_tz.tz_minuteswest * 60;
  rtc_time_to_tm(txc.time.tv_sec, &tm);
  sprintf(time, "UTC time :%d-%d-%d %d:%d:%d \n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

  if (index == 1 && flag != 0)
  {
    strcpy(buf, time);
    strcat(buf, "Index,File path,Now Hash Value,Extend Hash Value,Container Id\n");
  }
  if (index == 1 && flag == 0)
  {
    strcpy(buf, time);
    strcat(buf, "Index,File path,Now Hash Value,Extend Hash Value,Container Id\n");
  }

  sprintf(write_buf, "%d,%s,%s,%s,%s\n", index, file_path, hash, hash_extend, container_id);
  strcat(buf, write_buf);

  return 0;
}

char *file_read(char *filename)
{
  struct file *fp_r;
  int len = 0;
  char *config_content = NULL;
  mm_segment_t fs;
  loff_t pos = 0;
  int fsize;

  fp_r = filp_open(filename, O_CREAT | O_RDWR, 0);
  if (IS_ERR(fp_r))
  {
    printk("1 container error code %ld\n", PTR_ERR(fp_r));
    printk("create file error\n");
    return NULL;
  }

  fsize = fp_r->f_inode->i_size;
  config_content = (char *)kmalloc(fsize + 2, GFP_KERNEL);
  memset(config_content, 0, fsize + 2);

  fs = get_fs();
  set_fs(KERNEL_DS);

  len = kernel_read(fp_r, config_content, fsize, &pos);

  config_content[fsize] = '\0';

  filp_close(fp_r, NULL);
  set_fs(fs);
  return config_content;
}

char *get_file_path(char *log_file_path, char *connect_string)
{
  struct file *fp_r;
  mm_segment_t fs;
  loff_t pos = 0;
  int fsize = 0;
  int mem_size = 0;
  int len = 0;
  char *config_content = NULL;

  fp_r = filp_open(log_file_path, O_CREAT | O_RDWR, 0);
  if (IS_ERR(fp_r))
  {
    printk("1 container error code %ld\n", PTR_ERR(fp_r));
    printk("create file error\n");
    return NULL;
  }

  fsize = fp_r->f_inode->i_size;

  mem_size = fsize + strlen(connect_string) + 10;
  config_content = (char *)kmalloc(mem_size, GFP_KERNEL);
  memset(config_content, 0, mem_size);

  fs = get_fs();
  set_fs(KERNEL_DS);

  len = kernel_read(fp_r, config_content, fsize, &pos);

  config_content[fsize] = '\0';
  strcat(config_content, connect_string);

  filp_close(fp_r, NULL);
  set_fs(fs);
  return config_content;
}

int deal_code_seg_path(char *critical_file_name, char *suffix)
{
  char process_name[256] = {'\0'};
  int i = 0;
  for (i = 0; i < strlen(critical_file_name); i++)
  {
    if (critical_file_name[i] == '/')
    {
      critical_file_name[i] = '_';
    }
  }

  strcat(process_name, critical_file_name);
  strcat(process_name, suffix);
  memset(critical_file_name, '\0', strlen(critical_file_name));
  strcpy(critical_file_name, process_name);
  return 0;
}

int deal_dic_name(char *critical_file_name)
{
  char dic_name[256] = {'\0'};
  int i = 0;
  for (i = 0; i < strlen(critical_file_name); i++)
  {
    if (critical_file_name[i] == '_')
    {
      dic_name[i] = '/';
    }
    else
    {
      dic_name[i] = critical_file_name[i];
    }
  }
  strcpy(critical_file_name, dic_name);
  return 0;
}

int cal_extend_hash(char *buf, int buf_size, char *keyfile, char *config_file_path, char *write_path, char *hash_extend_buf, char *container_id, int dec_file_size, int flag_log)
{

  int flag = 0, flag1 = 0, flag2 = 0;
  int buf_len = 0;
  char *token = NULL;
  int index = 0;
  char hash_buf[HASH_SIZE] = {'\0'};
  char critical_file_name[FILE_PATH_MAX] = {'\0'};
  int malloc_size = 32;
  char *config_content = NULL;

  unsigned char *digest_now = (unsigned char *)kmalloc(malloc_size, GFP_KERNEL);

  unsigned char *digest_ext = (unsigned char *)kmalloc(malloc_size, GFP_KERNEL);
  memset(digest_now, 0, malloc_size);
  memset(digest_ext, 0, malloc_size);

  config_content = file_read(config_file_path);

  token = config_content;
  while (token != NULL)
  {
    token = strsep(&config_content, "\n");

    if (token != NULL && *token != '#' && *token != '\n')
    {
      flag = parse_config_file(token, "file_path=", critical_file_name);

     
      flag1 = parse_config_file(token, "directory_path=", critical_file_name);

      flag2 = parse_config_file(token, "process_path=", critical_file_name);

      if (flag2 == 1)
      {
        deal_code_seg_path(critical_file_name, "_text_segment_data"); 
      }

      if (flag1 == 1)
      {
        deal_code_seg_path(critical_file_name, ""); 
      }

      if (flag == 1 || flag1 == 1 || flag2 == 1)
      {
        index++;

     
        if (flag == 1 || flag2 == 1)
        {
          if (-1 == crypto_sha256(digest_now, critical_file_name, hash_buf))
          {
            printk("hash error!");
            return -1;
          }
        }
        else
        {
          
          if (-1 == hash_directory(critical_file_name, hash_buf, critical_file_name, digest_now, 0))
          {
            printk("hash directory error!");
            return -1;
          }
          deal_dic_name(critical_file_name);
        }

        vpcr_extend(digest_ext, digest_now, hash_extend_buf, 32);

        if (flag_log == 0)
        {
         
          write_to_buffer(dec_file_size, buf, index, critical_file_name, hash_buf, hash_extend_buf, container_id);
        }

        memset(hash_buf, 0, HASH_SIZE);
        memset(critical_file_name, 0, FILE_PATH_MAX);
        memset(digest_now, 0, malloc_size);
      }
    }
  }

  if (flag_log == 0)
  {
    buf_len = strlen(buf);
    aes_encrypt_buffer(write_path, keyfile, buf, buf_size, buf_len, dec_file_size);
  }

  kfree(config_content);
  kfree(digest_ext);
  kfree(digest_now);
  config_content = NULL;
  token = NULL;

  return index;
}

int get_log(char *config_content, int frow, int fcolumn, char *content, int fsize)
{
  int row = 0, i, j = 0;
  int col = 0;

  for (i = 0; i < fsize; i++)
  {
    if (config_content[i] == '\n')
      row++;
    if (row == frow)
    {

      if (config_content[i] == ',')
      {
        col++;
      }
      if (col == fcolumn && config_content[i] != ',')
      {
        content[j] = config_content[i];
        j++;
      }
    }
    if (row > frow)
      break;
  }

  return row;
}

int get_log_content(char *file, int row, int column, char *content)
{
  char *file_content = NULL;
  char *token = NULL;
  char *col_token = NULL;
  int i = 0, j = 0;
  char *row_content = (char *)kmalloc(512, GFP_KERNEL);
  file_content = file_read(file);
  token = file_content;

  while (token != NULL)
  {
    i++;
    token = strsep(&file_content, "\n");
    if (i == row)
    {
      my_strcpy(row_content, token);
      col_token = row_content;

      while (col_token != NULL)
      {
        j++;
        col_token = strsep(&row_content, ",");
        if (j == column)
        {
          my_strcpy(content, col_token);
          break;
        }
      }
      break;
    }
  }
  kfree(row_content);
  row_content = NULL;
  kfree(file_content);
  file_content = NULL;
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

int get_last_hang(char *config_content, int fsize)
{
  int i = 0, row = 0;
  for (i = 0; i < fsize; i++)
  {
    if (config_content[i] == '\n')
      row++;
  }
  config_content = NULL;
  return row;
}

int config_file_count(char *file_path)
{
  char *file_content = NULL;
  char *token = NULL;
  int file_count = 0;

  file_content = file_read(file_path);
  token = file_content;
  while (token != NULL)
  {
    token = strsep(&file_content, "\n");

    if (token != NULL && *token != '#' && *token != '\n')
    {
      file_count = file_count + 1;
    }
  }
  return file_count;
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

int record_abnormal(char *container_id)
{
  char time[HASH_SIZE] = {'\0'};
  char error_mess[300] = {'\0'};
  int result = 0;
  char cmdPath[] = "/bin/bash";
  char *cmdArgv[] = {cmdPath, "-c", "echo 111 >> 1.txt", NULL};
  char *cmdEnvp[] = {"HOME=/", "PATH=/sbin:/bin:/usr/bin", NULL};

  struct timex txc;
  struct rtc_time tm;

  do_gettimeofday(&(txc.time)); 

  txc.time.tv_sec -= sys_tz.tz_minuteswest * 60;
  rtc_time_to_tm(txc.time.tv_sec, &tm);
  sprintf(time, "UTC time :%04d-%02d-%02d %02d:%02d:%02d ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

  sprintf(error_mess, "echo 'Container measure error,ID is %s ,It may be attacked! By TCEC agent. %s' >> %s%s", container_id, time, Project_path, "TCEC_log");

  cmdArgv[2] = error_mess;
  result = call_usermodehelper(cmdPath, cmdArgv, cmdEnvp, UMH_WAIT_PROC);
  return 0;
}

static ssize_t char_write(struct file *file, const char *str, size_t size, loff_t *offset)
{
  size_t maxdata = 400, copied;
  int index = 0;
  char receive_data[maxdata];
  char receive_data_cp[maxdata];
  char op[2] = {'\0'};

  char base_value_path[FILE_PATH_MAX] = {'\0'};
  char config_file_path[FILE_PATH_MAX] = {'\0'};
  char container_id[container_id_size] = {'\0'};
  char extend_hash[HASH_SIZE] = {'\0'};
  char meas_value_path[HASH_SIZE] = {'\0'};
  char meas_extend_hash[HASH_SIZE] = {'\0'};
  char base_extend_hash[HASH_SIZE] = {'\0'};
  char measure_log_hash[HASH_SIZE] = {'\0'};
  char agent_path[HASH_SIZE] = {'\0'};
  char TCEC_path_in_container[HASH_SIZE] = {'\0'};
  char relat_tmp_key[FILE_PATH_MAX] = "/vtpm/tmp_key";
  int row = 0;
  int meas_flag = 0, config_count = 0;
  char *buf_base = NULL, *container_key = NULL, *tmp_key = NULL;
  int buf_base_size = 0, ffsize = 0, last_row = 0, write_meas_log_pos = 0;
  const char key_handle[] = "0x81010002";

  if (size < maxdata)
  {
    maxdata = size;
  }

  copied = _copy_from_user(receive_data, str, maxdata);
  if (copied == 0)
  {
    receive_data[maxdata] = '\0';

 
    strcpy(receive_data_cp, receive_data);
    parse_receive_string(receive_data_cp, 1, op);

    strcpy(receive_data_cp, receive_data);
    parse_receive_string(receive_data_cp, 2, container_id);

    
    strcpy(receive_data_cp, receive_data);
    memset(TCEC_path_in_container, 0, sizeof(TCEC_path_in_container));
    parse_receive_string(receive_data_cp, 3, TCEC_path_in_container);
    sprintf(config_file_path, "%s/%s", TCEC_path_in_container, "config");
    sprintf(agent_path, "%s/%s", TCEC_path_in_container, "agent");

    config_count = config_file_count(config_file_path);
    buf_base_size = config_count * LINE_LEN;
    buf_base = kmalloc(buf_base_size, GFP_KERNEL);
    memset(buf_base, 0, buf_base_size);

    container_key = get_file_path("/vtpm/log", "/container_key");
    tmp_key = get_file_path("/vtpm/log", "/tmp_key");


    unsigned char *agent_digest = (unsigned char *)kmalloc(32, GFP_KERNEL);
    char agent_buf[HASH_SIZE] = {'\0'};
    char agent_hash_res[HASH_SIZE] = {'\0'};
    memset(agent_digest, 0, 32);
    crypto_sha256(agent_digest, agent_path, agent_hash_res);

    if (hash_compare(agent_base_hash, agent_hash_res) == -1)
    {
      kfree(container_key);
      container_key = NULL;
      kfree(tmp_key);
      tmp_key = NULL;
      kfree(buf_base);
      buf_base = NULL;

      return 2;
    }

    switch (op[0])
    {
    case base_value_mode:
    {
     
      strcpy(base_value_path, vTPM_PATH);
      strcat(base_value_path, "base.csv");
      tpm_decrypt(container_key, tmp_key, key_handle);
      cal_extend_hash(buf_base, buf_base_size, relat_tmp_key, config_file_path, base_value_path, extend_hash, container_id, 0, 0);
      con_rm(tmp_key);
      break;
    }
    case measure_mode:
    {
      strcpy(meas_value_path, vTPM_PATH);
      strcat(meas_value_path, "measure.csv");

      strcpy(base_value_path, vTPM_PATH);
      strcat(base_value_path, "base.csv");
      tpm_decrypt(container_key, tmp_key, key_handle);
      ffsize = aes_decrypt_buffer(base_value_path, relat_tmp_key, buf_base);
      last_row = get_last_hang(buf_base, ffsize);
      get_log(buf_base, last_row - 1, 3, base_extend_hash, ffsize); // row

      memset(buf_base, 0, buf_base_size);
      row = cal_extend_hash(buf_base, buf_base_size, relat_tmp_key, config_file_path, meas_value_path, meas_extend_hash, container_id, 0, 1);

      meas_flag = hash_compare(base_extend_hash, meas_extend_hash);

      if (meas_flag != 0)
      {
        record_abnormal(container_id);
      }
      
      write_meas_log_pos = log_file_exist(meas_value_path);

      memset(measure_log_hash, 0, sizeof(measure_log_hash));

      if (write_meas_log_pos != 0)
      {
   
        int meas_buf_size = write_meas_log_pos + buf_base_size;
        char *meas_buf = (char *)kmalloc(meas_buf_size, GFP_KERNEL);
        memset(meas_buf, 0, meas_buf_size);

        ffsize = aes_decrypt_buffer(meas_value_path, relat_tmp_key, meas_buf);
        last_row = get_last_hang(meas_buf, ffsize);
        get_log(meas_buf, last_row - 1, 3, measure_log_hash, ffsize);

        if (hash_compare(meas_extend_hash, measure_log_hash) == -1)
        {
          cal_extend_hash(meas_buf, meas_buf_size, relat_tmp_key, config_file_path, meas_value_path, meas_extend_hash, container_id, ffsize, 0); //记录
          printk("%s", "record measure log!");
        }
        else
        {
          printk("%s", "the same as measure log,no need to update.");
        }
        con_rm(tmp_key);
        kfree(meas_buf);
        meas_buf = NULL;
      }
      else
      {
      
        memset(buf_base, 0, buf_base_size);
        cal_extend_hash(buf_base, buf_base_size, relat_tmp_key, config_file_path, meas_value_path, meas_extend_hash, container_id, 0, 0);
        con_rm(tmp_key);
      }

      break;
    }
    }
    kfree(container_key);
    container_key = NULL;
    kfree(tmp_key);
    tmp_key = NULL;
    kfree(buf_base);
    buf_base = NULL;
  }
  else
  {
    printk("communiate error!");
  }

  return meas_flag;
}

module_init(module_init_function);
module_exit(module_exit_function);
