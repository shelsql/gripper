import os
import glob

for feat_path in glob.glob("./*_views/*/*"):
    if feat_path.endswith("ref_bags_23.npy") or feat_path.endswith("word_list_23.npy"):
        continue
    if feat_path.endswith("intrinsics.json"):
        continue
    prefix = '/'.join(feat_path.split('/')[:-1])
    num = int(feat_path.split('/')[-1].split('_')[0])
    suffix = '_'.join(feat_path.split('/')[-1].split('_')[1:])
    print(prefix, num, suffix) 
    new_path = "%s/%.6d_%s" % (prefix, num, suffix)
    print(new_path)
    os.rename(feat_path, new_path)