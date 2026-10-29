{
    read  
    while IFS=',' read -r File_ID File_Name Data_Category Data_Type Project_ID Case_ID Sample_ID Tissue_Type Tumor_Descriptor Specimen_Type Preservation_Method Case_ID2 mask_name
    do
        # echo ${Project_ID}
        if grep -Fxq "$Case_ID2" /home/zhaozhangxiang/TLScell_Finder/data/finish_res.txt
        then
            echo "${Case_ID2} is already done!!!"
            continue
        elif [ "$Project_ID" == "TCGA-LUAD" ]
        then
            cd /home/zhaozhangxiang/TLScell_Finder/data/tmp
            aliyunpan d --save /TCGA-HE/TCGA-LUAD/${File_ID}/${File_Name}
            mkdir -p "/home/zhaozhangxiang/TLScell_Finder/data/test_data/${Case_ID2}"
            mv ./TCGA-HE/TCGA-LUAD/${File_ID}/${File_Name} /home/zhaozhangxiang/TLScell_Finder/data/test_data/${Case_ID2}/${Case_ID2}.svs
            apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
                --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/run_czi_cellpose_pipeline.py \
                --tif_dir /home/TLScell_Finder/data/test_data/${Case_ID2}/ \
                --out_dir /home/TLScell_Finder/data/test_data/${Case_ID2}/seg \
                --patch_size 1024 \
                --stride 512 \
                --min_coverage 0.25 \
                --tissue_mode light \
                --save_format png \
                --num_workers 4
            apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
                --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/run_cellcrops.py \
                --patch_csv /home/TLScell_Finder/data/test_data/${Case_ID2}/seg/tables/${Case_ID2}_patch_index.csv \
                --out_dir /home/TLScell_Finder/data/test_data/${Case_ID2}/cellpose \
                --model_name /home/TLScell_Finder/data/cellpose/cpsam_v2 \
                --gpu \
                --num_gpus 1 \
                --gpu_ids 0 \
                --crop_size 64 \
                --batch_size 16
            apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
                --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/model_eval.py \
                --meta_path /home/TLScell_Finder/data/test_data/${Case_ID2}/cellpose/mask_tables/${Case_ID2}_patch_index_cellpose_mask_instances.csv \
                --save_path /home/TLScell_Finder/data/cellpose//best_model.pth \
                --batch_size 16384 \
                --num_gpus 1
            mask_path=/home/zhaozhangxiang/TLScell_Finder/data/masks/${mask_name}_tls_mask.tif
            if [ -f ${mask_path} ]
            then
                apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
                    --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/draw_cell.py \
                    --csv_path /home/TLScell_Finder/data/test_data/${Case_ID2}/cellpose/mask_tables/${Case_ID2}_patch_index_cellpose_mask_instances.csv \
                    --tif_dir /home/TLScell_Finder/data/test_data/${Case_ID2} \
                    --output_dir /home/TLScell_Finder/data/test_data/${Case_ID2}/draw_cell/ \
                    --celltype_col celltype \
                    --sample_col source_tif \
                    --line_width 1 \
                    --types CD4+T,B,CD8+T,Macrophage,T \
                    --mask_tif_path /home/TLScell_Finder/data/masks/${mask_name}_tls_mask.tif
            else
                apptainer exec --nv --underlay -B /home/zhaozhangxiang/:/home \
                    --pwd /home /home/zhaozhangxiang/software/apptainer/nvidia_cuda12.2_wsl python /home/TLScell_Finder/src/draw_cell.py \
                    --csv_path /home/TLScell_Finder/data/test_data/${Case_ID2}/cellpose/mask_tables/${Case_ID2}_patch_index_cellpose_mask_instances.csv \
                    --tif_dir /home/TLScell_Finder/data/test_data/${Case_ID2} \
                    --output_dir /home/TLScell_Finder/data/test_data/${Case_ID2}/draw_cell/ \
                    --celltype_col celltype \
                    --sample_col source_tif \
                    --line_width 1 \
                    --types CD4+T,B,CD8+T,Macrophage,T
            fi
            cd /home/zhaozhangxiang/TLScell_Finder/data/test_data/${Case_ID2}
            aliyunpan upload ./draw_cell/${Case_ID2}_boxes_overlay.png /TCGA-HE/TCGA_res/draw_cell/
            tar -czvf ${Case_ID2}_res.tar.gz *
            aliyunpan upload ${Case_ID2}_res.tar.gz /TCGA-HE/TCGA_res/
            echo ${Case_ID2} >> /home/zhaozhangxiang/TLScell_Finder/data/finish_res.txt
            rm -rf *
        fi
    done  
} < /home/zhaozhangxiang/TLScell_Finder/data/gdc_sample_sheet.2025-09-28.csv