import os
import sys
import time
import json
import argparse

from tempfile import TemporaryDirectory


def get_machine_info(output="./machine_info.json", online=False):
    r"""
    Use pyarmor register a License.

    Args:

    output : A json file with machine information generated using revive sdk. E.g. : "/home/machine_info.json"
   
    """
    assert output.endswith(".json"), f"Machine info should be saved as a json file. -> {output}"
    with TemporaryDirectory() as dirname:
        py_path = os.path.join(dirname, "hdinfo.py")
        with open(os.path.join(dirname, "hdinfo.py"), "w") as f:
            f.writelines(["from pyarmor.pyarmor import main\n", "main(['hdinfo'])\n"])
        res_path = os.path.join(dirname, "res.out")
        
        os.system("nohup "+sys.executable+" -u "+py_path+">"+res_path+"&")
        time.sleep(1)
        with open(res_path, "r") as f:
            lines = f.readlines()

    hd_info = {
        "harddisk" : [],
        "mac" : [],
        "ip" : [],
    }
    for line in lines:
        if "default harddisk" in line:
            hd_info["harddisk"].append(line[line.index('"')+1:-2])

        if "Default Mac address" in line:
            hd_info["mac"].append(line[line.index('"')+1:-2])

        if "Ip address" in line:
            hd_info["ip"].append(line[line.index('"')+1:-2])
            
    machine_info = {"hd_txt": lines,  "hd_info": hd_info}

    if online:
        with open(output,"w") as f:
            json.dump(machine_info,f)
        with open(output, "r") as f:
            machine_info = f.readlines()[0]
        return machine_info

    with open(output,"w") as f:
        print(f"Svae machine info -> {output}")
        json.dump(machine_info,f)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The file path where the machine information is saved.')
    parser.add_argument('-o', '--output', default="./machine_info.json", help="The file save path of machine information.")

    args = parser.parse_args()
    get_machine_info(args.output)