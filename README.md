# co-ops-server
Server-side Docker image for federating Imitation-learned-driving policy networks

Docker image: (https://hub.docker.com/r/shuheif/co-ops-server)

# Requirements
Python=3.10

Example:
```
conda create -n Co-Ops python=3.10 -y
conda activate Co-Ops
pip install -r requirements.txt
```

# Run as a Server
1. Download the initial model weights here:
https://drive.google.com/file/d/1MOJl0HcsvWVgzmGFDNKfa2Dfdr9bBTGD/view?usp=drive_link

1. Run the command below to start the server:
```
python src/server.py --ckpt_path ./init_weights.ckpt
```
