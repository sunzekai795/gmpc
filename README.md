# GMPC Dataset
This is the release repository for the GMPC dataset, with different synthetic versions of GMPC being continuously updated...


## Methods for Dataset Download and Merging
1. Download all chunked files:
```bash
wget https://github.com/sunzekai795/gmpc/releases/download/v1/gmpc_v1.tar.gz.part_*
```

2. Merge Chunks (Linux/macOS):
```bash
cat gmpc_v1.tar.gz.part_* > gmpc_v1.tar.gz
```

3. Extract and use:
```bash
tar -zxvf gmpc_v1.tar.gz
```