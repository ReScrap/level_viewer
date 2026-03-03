import re
import os
import sys
from collections import Counter
import glob
cnt=dict()
shaders=[]
for file in glob.glob(sys.argv[1]):
    lines=[]
    if not file.endswith(".psh"):
        continue
    for line in open(file):
        line=line.strip().lstrip("+").strip()
        line=line.split("//")[0].strip()
        line=line.split(";")[0].strip()
        if line.startswith(";") or line.startswith("//") or not line:
            continue
        if line=="ps.1.1":
            continue
        lines.append(line)
    shaders.append((file,lines))
for name,code in sorted(shaders,key=lambda s:len(s[1])):
    print(name)
    print("\n".join(code))
    print("="*10)
"""
from glob import glob
insts=set()
suffxs=set()
for f in glob("*.psh"):
    print(">>",f)
    for l in open(f):
        print(">"," ".join(l.strip().split()))
        l=l.strip().split("//")[0].strip().lstrip("+").split()
        if not l:
            continue
        inst,*suffx=l[0].split("_")
        if inst=="ps":
            continue
        insts.add(inst)
        suffxs|=set(suffx)
"""

# for k,v in sorted(cnt.items(),key=lambda v: v[1]):
#     print(k,v)