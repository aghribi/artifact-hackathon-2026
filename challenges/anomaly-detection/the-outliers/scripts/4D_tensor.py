import glob
import torch

dossiers = [
    r"/lustre/work/pdl18409/shared/data/anomaly-detection/teams/the-outliers/DD/step3_DD_tensors/SPOKES_DD_tensors",
    r"/lustre/work/pdl18409/shared/data/anomaly-detection/teams/the-outliers/DD/step3_DD_tensors/HBL_DD_tensors",
    r"/lustre/work/pdl18409/shared/data/anomaly-detection/teams/the-outliers/DD/step3_DD_tensors/MBL_DD_tensors",
]

file_paths = []
for dossier in dossiers:
  fichiers_dossier = sorted(glob.glob(f"{dossier}/*.pt"))
  file_paths.extend(fichiers_dossier)

all_cycle_ids_sets = []

for path in file_paths:
  content = torch.load(path)
  cycle_ids = content["cycle_id"]
  if isinstance(cycle_ids, torch.Tensor):
    cycle_ids = cycle_ids.tolist()
  all_cycle_ids_sets.append(set(cycle_ids))
  
  del content

common_cycle_ids = sorted(list(set.intersection(*all_cycle_ids_sets)))
num_cycles = len(common_cycle_ids)
num_cavities = len(file_paths)

print(f"Number of cavities : {num_cavities}")
print(f"Number of common cycle_ids : {num_cycles}")

if num_cycles == 0:
  raise ValueError("No common cycle_ids found !")

sample_content = torch.load(file_paths[0])
sample_data = sample_content["data"]
if sample_data.dtype == torch.float64:
  sample_data = sample_data.float()
channels, length = sample_data.shape[1], sample_data.shape[2]
del sample_content, sample_data

final_data = torch.empty(
    (num_cycles, num_cavities, channels, length), dtype=torch.float32
)

for j, path in enumerate(file_paths):
  print(f"Processing cavity {j + 1}/{num_cavities}...")
  content = torch.load(path)
  data = content["data"]
  if data.dtype == torch.float64:
    data = data.float()
  cycle_ids = content["cycle_id"]
  if isinstance(cycle_ids, torch.Tensor):
    cycle_ids = cycle_ids.tolist()

  sys_dict = {cid: data[i] for i, cid in enumerate(cycle_ids)}
  
  for i, cid in enumerate(common_cycle_ids):
    final_data[i, j] = sys_dict[cid]
    
  del content, data, sys_dict

print("Shape of the final tensor :", final_data.shape)

output_path = r"/lustre/work/pdl18409/shared/data/anomaly-detection/teams/the-outliers/DD/4D_DD_tensor.pt"
torch.save(final_data, output_path)
print(f"Final tensor saved to {output_path}")