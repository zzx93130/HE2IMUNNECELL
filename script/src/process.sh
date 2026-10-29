# apptainer shell --writable --nv -B /home/zhaozhangxiang/TLScell_Finder/:/home --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl

i=TCGA-49-4514
apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
    --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/run_czi_cellpose_pipeline.py \
    --tif_dir /home/TLScell_Finder/data/test_data/ \
    --out_dir /home/TLScell_Finder/data/test_data/seg \
    --patch_size 1024 \
    --stride 512 \
    --min_coverage 0.25 \
    --tissue_mode light \
    --save_format png \
    --num_workers 4
# apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
#     --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/run_cellpose_on_patches.py \
#     --patch_csv /home/TLScell_Finder/data/test_data/seg/tables/${i}_patch_index.csv \
#     --out_dir /home/TLScell_Finder/data/test_data/cellpose \
#     --model_name /home/cellpose/cpsam_v2 \
#     --gpu \
#     --batch_size 8
apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
    --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/run_cellcrops.py \
    --patch_csv /home/TLScell_Finder/data/test_data/seg/tables/${i}_patch_index.csv \
    --out_dir /home/TLScell_Finder/data/test_data/cellpose \
    --model_name /home/TLScell_Finder/data/cellpose/cpsam_v2 \
    --gpu \
    --num_gpus 1 \
    --gpu_ids 0 \
    --crop_size 64 \
    --batch_size 16

apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
    --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/model_eval.py \
    --meta_path /home/TLScell_Finder/data/test_data/cellpose/mask_tables/${i}_patch_index_cellpose_mask_instances.csv \
    --save_path /home/TLScell_Finder/data/cellpose//best_model.pth \
    --batch_size 16384 \
    --num_gpus 1
apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
    --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/draw_cell.py \
    --csv_path /home/TLScell_Finder/data/test_data/cellpose/mask_tables/${i}_patch_index_cellpose_mask_instances.csv \
    --tif_dir /home/TLScell_Finder/data/test_data \
    --output_dir /home/TLScell_Finder/data/test_data/draw_cell/ \
    --celltype_col celltype \
    --sample_col source_tif \
    --line_width 1 \
    --types CD4+T,B,CD8+T,Macrophage,T \
    # --mask_tif_path /home/TLScell_Finder/data/test_data/mask/TCGA-18-3406-01Z-00-DX1_tls_mask.tif 
