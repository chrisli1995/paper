#! bash
timedatectl set-local-rtc 1

make

pro_path=$(cd "$(dirname "$0")";pwd)
pro_path="${pro_path}/"


base_value_path="${pro_path}base/"

# mkdir base
key_path="${base_value_path}base_key"
tpm2_getrandom 32 -o ${key_path}
tpm2_rsaencrypt -c 0x81010002 -o ${key_path} ${key_path}

gcc tcec.c -o tcec

chmod +x "${pro_path}TCEC"
ln -s "${pro_path}TCEC" /usr/bin/TCEC

echo "TCEC is starting."
echo "vTPM manager is runing."

insmod "${pro_path}crypto.ko" Project_path=$pro_path Base_value_path=$base_value_path
if [ $? -eq 0 ];then
    echo "crypto module is loading."
else
    echo "crypto loads error."
fi

insmod "${pro_path}image_measure.ko" Project_path=$pro_path Base_value_path=$base_value_path
if [ $? -eq 0 ];then
    echo "image_measure module is loading."
else
    echo "image_measure module loads error."
fi

insmod "${pro_path}communications.ko" Project_path=$pro_path Base_value_path=$base_value_path 
if [ $? -eq 0 ];then
    echo "communications module is loading."
else
    echo "communications module loads error."
fi

insmod "${pro_path}operation_monitor.ko" Project_path=$pro_path Base_value_path=$base_value_path
if [ $? -eq 0 ];then
    echo "operation_monitor module is loading."
else
    echo "operation_monitor module loads error."
fi

insmod "${pro_path}dec.ko" proj_path=$pro_path
