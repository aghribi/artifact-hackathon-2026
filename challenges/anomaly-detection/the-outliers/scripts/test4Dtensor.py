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

print(f"Nombre total de fichiers trouvés : {len(file_paths)}\n")

for i, path in enumerate(file_paths):
  content = torch.load(path)
  cycle_ids = content["cycle_id"]
  
  if isinstance(cycle_ids, torch.Tensor):
    cycle_ids = cycle_ids.tolist()

  print(f"--- Fichier {i + 1} : {path} ---")
  print(f"  Nombre de cycles (N) : {len(cycle_ids)}")
  print(f"  Type du premier cycle_id : {type(cycle_ids[0])} (Valeur: {cycle_ids[0]})")
  print(f"  Min cycle_id : {min(cycle_ids)} | Max cycle_id : {max(cycle_ids)}")
  print("")