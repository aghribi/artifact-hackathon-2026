# Operations

## What runs

Nothing since Thu 15 Oct, 17:14: both jobs below were cancelled. The image and the weights stay on disk, so `sbatch` starts it again without a download.

When running, one Slurm job in the reservation `pdl18409_oct2026`: vLLM in an Apptainer image, serving `RedHatAI/GLM-5.3-Flash-NVFP4` on two GB200 as `glm-5.3-flash`, port 8000 on the compute node. Compute nodes are reachable only from inside the cluster, so access goes through ssh and an API key.

Everything is in one folder, `/lustre/work/pdl18409/shared/vllm`, read and write for the group `pdl18409`. The scripts set `umask 007`, so files they create keep that.

| | Path on DALIA | Size |
|---|---|---|
| Scripts | `/lustre/work/pdl18409/shared/vllm/vllm_{serve,pull,download}.sh` | |
| Image | `/lustre/work/pdl18409/shared/vllm/vllm-glm53-flash.sif` | 7.7 GiB |
| Weights | `/lustre/work/pdl18409/shared/vllm/cache_huggingface` | 185 GiB |
| Image build cache | `/lustre/work/pdl18409/shared/vllm/cache_apptainer` | |
| Logs | `/lustre/work/pdl18409/shared/vllm/slurm_logs` | |
| Node name, API key | `/lustre/work/pdl18409/shared/vllm/{node,api_key}` | |

## Schedule

A job is capped at 2 days, so the server is a chain of jobs, each starting when the previous one ends.

| Job | From | To |
|---|---|---|
| 151655 | Tue 13 Oct, 17:36 | Thu 15 Oct, 17:14, cancelled |
| 151656 | never started | Thu 15 Oct, 17:14, cancelled |

Each start takes 11 minutes before the model answers. The node can change at each start; the job rewrites `node`. The key is kept.

## Commands

From `/lustre/work/pdl18409/shared/vllm`, by any member of the group:

```bash
squeue -n vllm_serve                             # state, node, time left, who started it
tail -f slurm_logs/vllm_serve_<job>.err          # startup and requests
scancel <job>                                    # stop (only the member who started it)
sbatch --time=2-00:00:00 vllm_serve.sh           # start
sbatch --dependency=afterany:<job> --time=22:00:00 vllm_serve.sh   # chain the next one
rm api_key                                       # new key at the next start
```

First install, once:

```bash
mkdir -p slurm_logs
sbatch vllm_pull.sh        # arm64 vLLM image to a .sif, 7 min, 1 GPU
sbatch vllm_download.sh    # 198 GB of weights, 3 min, no GPU
```

## To know

- Nothing restarts a crashed job. If one dies, the next in the chain starts early and the chain ends sooner.
- While running, the server holds two of the 12 hackathon GPUs.
- The official FP8 weights (`zai-org/GLM-5.3-Flash`) do not start on two GB200: they take 158 of the 184 GiB of each GPU.
- The reservation ends Fri 16 Oct, 16:00.
