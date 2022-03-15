# About This Python Script

To use this script, run
```bash
tinker9.docker [CudaVersion] [Stage] | bash
```

Use command `tinker9.docker -h` to see the valid values of `CudaVersion` and `Stage`.

# Steps to Install Docker on Ubuntu

### 1. Remove the old docker version as necessary
```bash
sudo apt remove docker docker-engine docker.io containerd runc
```

### 2. Update apt
```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
```
Fix the `curl` package as necessary.

### 3. Add docker's official GPG key
```bash
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

### 4. Choose the stable version to install
```bash
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
```

### 5. Install the nvidia container toolkit
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 6. Manage docker as a non-root user and reboot the system
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
sudo reboot
```