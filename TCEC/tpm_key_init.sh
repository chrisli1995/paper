#! bash
tpm2_createprimary -c primary.ctx
tpm2_create -C primary.ctx -Grsa2048 -u key.pub -r key.priv
tpm2_load -C primary.ctx -u key.pub -r key.priv -c key.ctx
tpm2_evictcontrol -c key.ctx 0x81010002
if [ $? -eq 0 ];then
    echo "TPM master key has been saved"
else
    echo "Key saved error!"
fi

rm -rf primary.ctx key.pub key.ctx key.priv


tpm2_createprimary -c primary.ctx
tpm2_create -C primary.ctx -Gaes128 -u key.pub -r key.priv
tpm2_load -C primary.ctx -u key.pub -r key.priv -c key.ctx
tpm2_evictcontrol -c key.ctx 0x81010003
if [ $? -eq 0 ];then
    echo "TPM AES key has been saved"
else
    echo "Key saved error!"
fi

rm -rf primary.ctx key.pub key.ctx key.priv

mkdir base

pro_path=$(cd "$(dirname "$0")";pwd)
pro_path="${pro_path}/"

base_value_path="${pro_path}base/check"
base_value_path_tmp="${pro_path}base/.check"

make


module_name="crypto.ko operation_monitor.ko image_measure.ko communications.ko"  

for single_module_name in $module_name
do  
digest="$(sha256sum $single_module_name | cut -c 1-64 ) "
echo "$single_module_name:$digest" >> $base_value_path_tmp
done  

echo "agent:a14966bd772cfa7102526aa33cbc49df62798dce91692de4b55fee54b2bee817" >> $base_value_path_tmp

tpm2_encryptdecrypt -c 0x81010003 -o  $base_value_path $base_value_path_tmp
make clean

