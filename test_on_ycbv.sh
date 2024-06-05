

for i in $(seq 1 21);
do
    echo $i
    OBJ_ID=$i
    python test_on_ycbv.py --obj_id $OBJ_ID --single_opt --uni3d_layer=19 --pca_type=nope --result_fold_name=bop_ycbv_both
done