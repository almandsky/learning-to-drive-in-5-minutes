import os
import shutil
import glob

in_files = "./logs/tubs/tub*/*.jpg"
files = glob.glob(in_files, recursive=True)
print("found", len(files), "files")
dest_path = "./logs/images"
i = 0
for f in files:
    d = os.path.join(dest_path, "img_%d.jpg" % i)
    i += 1
    shutil.copyfile(f, d)
print("done")
