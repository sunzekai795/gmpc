# GMPC Dataset
This is the release repository for the GMPC dataset, with different synthetic versions of GMPC being continuously updated...


## Methods for Dataset Download and Merging
1. Download all chunked files:
```bash
parts=(aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az ba bb bc bd be bf bg bh bi bj bk)
for p in "${parts[@]}"; do
  wget "https://github.com/sunzekai795/gmpc/releases/download/v1/gmpc_v1.tar.gz.part_${p}"
done
```

2. Merge Chunks (Linux/macOS):
```bash
cat gmpc_v1.tar.gz.part_* > gmpc_v1.tar.gz
```

3. Extract and use:
```bash
tar -zxvf gmpc_v1.tar.gz
```

## Usage Restrictions
The GMPC dataset is restricted to **academic research only**, and is prohibited for commercial or clinical use.