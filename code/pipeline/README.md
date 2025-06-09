


ISAAC SIM Docker

Saved (24.05.24):
An image is saved on the external server under /home/thilo/workspace/docker_images/isaac-sim_pipeline.tar

New setup:
1. Follow instructions on https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html
2. Start interactive docker session with:
docker run --name isaac_sim --entrypoint bash -it --gpus '"'device=${CUDA_VISIBLE_DEVICES}'"' -e "ACCEPT_EULA=Y" --rm --network=host     -e "PRIVACY_CONSENT=Y"     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw     -v ~/workspace/:/isaac-sim/workspace:rw nvcr.io/nvidia/isaac-sim:2023.1.1

3. Install opencv-python with the given python.sh from Isaac Sim
    ~/python.sh -m pip opencv-python
4. Test if simulation_docker.py is runnig
    ~/python.sh simulation_docker.py
    - If Error ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory occures try the following:
        apt update
        apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
    and try again.
5. Save Running session as new image with name isaac-sim_pipeline
    - docker commit nvcr.io/nvidia/isaac-sim:2023.1.1 isaac-sim_pipeline

