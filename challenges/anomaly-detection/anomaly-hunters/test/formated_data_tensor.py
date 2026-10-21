from pathlib import Path
import numpy as np
import pandas as pd
import torch

signals = ["0", "5", "6"]
components = ["0", "1"]
length_waveform = 35221

# Define the exact 6 (signal, component) order required for the tensor channels
expected_pairs = [(s, c) for s in signals for c in components]
num_channels = len(expected_pairs)

input_dir = Path("/lustre/work/pdl18409/shared/The_outliers/PM/step2_PM_processedcavities/MBL_PM_grouped_by_cavity_name")
output_dir = Path("/lustre/work/pdl18409/shared/The_outliers/PM/step3_PM_tensors/MBL_PM_tensors")
output_dir.mkdir(parents=True, exist_ok=True)

for file in input_dir.glob("*.parquet"):
    print(f"Processing {file.name}...")
    df_global = pd.read_parquet(file)
    
    if df_global.empty:
        print(f"Skipping {file.name}: DataFrame is empty.")
        continue

    # 1. Extract the Dwn (signal) and Cmp (component) digits directly via regex
    extracted = df_global["pv"].str.extract(r"Dwn(\d+).*?Cmp(\d+)")
    df_global["signal"] = extracted[0]
    df_global["component"] = extracted[1]

    # 2. Keep only rows matching our target signals and components
    df_filtered = df_global[
        df_global["signal"].isin(signals) & 
        df_global["component"].isin(components)
    ]

    all_cycles_data = []
    valid_cycle_ids = []
    valid_timestamps = []

    # 3. Group by cycle_id
    for cycle_id, df_cycle in df_filtered.groupby("cycle_id", sort=False):
        pair_to_arr = {
            (row.signal, row.component): row.data_array 
            for row in df_cycle.itertuples(index=False)
        }

        cycle_list = []
        missing_data = False

        for sig, comp in expected_pairs:
            arr = pair_to_arr.get((sig, comp))
            
            if arr is None:
                print(f"Cycle {cycle_id} : PV manquant pour Dwn{sig} / Cmp{comp}")
                missing_data = True
                break
                
            if len(arr) != length_waveform:
                print(
                    f"Cycle {cycle_id} : Dwn{sig}/Cmp{comp} -> "
                    f"Longueur incorrecte : {len(arr)} (attendu : {length_waveform})"
                )
                missing_data = True
                break

            cycle_list.append(arr)

        if not missing_data:
            # Stack the 6 waveforms into a (6, 35221) float32 numpy array
            all_cycles_data.append(np.stack(cycle_list, axis=0).astype(np.float32))
            valid_cycle_ids.append(cycle_id)            

    # 4. Save dictionary safely
    if all_cycles_data:
        stacked_array = np.stack(all_cycles_data, axis=0)
        data_tensor = torch.from_numpy(stacked_array)
        
        # Création du dictionnaire final
        saved_dict = {
            "data": data_tensor,             # Tensor de shape (N, 6, 35221)
            "cycle_id": valid_cycle_ids     # Liste de taille N
        }
        
        output_tensor_path = output_dir / f"tensor_{file.stem}.pt"
        torch.save(saved_dict, output_tensor_path)
        print(
            f"Saved: {output_tensor_path} | "
            f"Events: {len(valid_cycle_ids)} | "
            f"Tensor shape: {data_tensor.shape}\n"
        )
    else:
        print(f"No valid cycles found in {file.name}. Nothing saved.\n")