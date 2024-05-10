# TCEC Manual

1. It is recommended to copy the code file to the $/home$ directory and name it TCEC.

   

2. Initialization Key: Enter the TCEC directory in the source code file and use the command.

   ```
   bash tpm_key_init
   ```

   sh persists the master key in the TPM, generates two sets of keys, and stores them in the TPM. The symmetric key is read through the $0x81010003$ handle, and the asymmetric key is read through the 0x81010002 handle. As the program's first use, this step will automatically generate the benchmark values of each module of the TCEC program and store them in ciphertext.

   

3. Install each module of TCEC: Enter the TCEC directory in the source code file and use the command to install each module of the TCEC program automatically.

   ```
   bash install.sh
   ```

   

4. The current TCEC program version number will be prompted, indicating that the TCEC program is successfully installed.

   ```
   TCEC -v
   ```

5. For the trusted agent, we uploaded the container image with the agent installed on dockerhub.

   ```
   https://hub.docker.com/r/tccontainer/tcc_test
   ```

   