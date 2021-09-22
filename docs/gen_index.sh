PROJECT_DIR="/screamlab/home/tanch/variable-length-piano-expansion"
python gen_index.py \
    -src_dir "$PROJECT_DIR/expand_result/selected" \
    -dist_dir "$PROJECT_DIR/docs/assets/songs/expansion" \
    -index_file "$PROJECT_DIR/docs/index.html" \
    -result_file "$PROJECT_DIR/docs/result.txt"
