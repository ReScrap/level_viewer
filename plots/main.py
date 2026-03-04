import json
import numpy as np
import os
import matplotlib.pyplot as plt
with open("dump.json","r") as fh:
    for path,data in json.load(fh).items():
        for obj,anim in sorted(data.items()):
            if not anim:
                print(f"Skipping {obj}")
                continue
            obj=obj.strip()
            print(path,obj)
            for track,data in sorted(anim.items()):
                if (not data) or (len(data)==1):
                    continue
                os.makedirs(f"out/{path}/{obj}",exist_ok=True)
                data=np.array(data)
                plt.clf()
                plt.title(f"{obj} {track}")
                plt.plot(data)
                plt.tight_layout()
                plt.savefig(f"out/{path}/{obj}/{track}.png")