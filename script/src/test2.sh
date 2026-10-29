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
            echo ${Case_ID2} >> /home/zhaozhangxiang/TLScell_Finder/data/nofinish_res.txt
        fi
    done
} < /home/zhaozhangxiang/TLScell_Finder/data/gdc_sample_sheet.2025-09-28.csv