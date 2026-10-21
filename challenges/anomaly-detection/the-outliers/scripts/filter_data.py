import tarfile
import io
from pathlib import Path
import pandas as pd

base_dir = Path("../data/raw")
parquet_dfs = []

section_selected = "MBL"
cryomodule_selected = "090"
cavity_selected = "101"
i=0

for tar_path in base_dir.rglob("*.tar"):
    print(f"Examen de l'archive : {tar_path.name}")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(section_selected + '.parquet'):
                    # Normalisation des slashes pour Windows
                    internal_path = member.name.replace('\\', '/')
                    
                    if 'DD' in internal_path:
                        i=i+1
                        print(f" -> Trouvé et chargé : {internal_path}")
                        
                        f_obj = tar.extractfile(member)
                        if f_obj:
                            df = pd.read_parquet(io.BytesIO(f_obj.read()))
                            
                            # Extract and filter using clean regex extraction
                            pv_extracted = df["pv"].str.extract(r"^[^-]*-([A-Za-z0-9]{3}).*?([A-Za-z0-9]{3}):")
                            
                            df = df[
                                (df["pv"].str.split("-", expand=True)[1].str[:3] == cryomodule_selected) &
                                (df["pv"].str.split(":", expand=True)[1].str[-3:] == cavity_selected) &  # Changed [0] to [1]
                                (df["beam_state"] == "BEAM_PRESENT_YES")
                            ]
                            
                            print(f"    -> DataFrame shape après filtrage : {df.shape}")
                            if not df.empty:
                                parquet_dfs.append(df)
                            
    except Exception as e:
        print(f"Error with {tar_path.name} : {e}")

print(f"\nNumber of tar files loaded: {i}")
if parquet_dfs:
    df_global = pd.concat(parquet_dfs, ignore_index=True)
    print(f"\nSuccess ! {len(parquet_dfs)} files loaded. DataFrame global : {df_global.shape}")
else:
    print("\nNo files match the filter in the internal paths.")