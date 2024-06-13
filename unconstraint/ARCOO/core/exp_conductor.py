import json
import time
import os


class ExpConductor(object):
    def __init__(self, task, algo, config_lable='default.json', label=None):
        self.task = task
        self.algo = algo
        self.label = label
        
        with open(os.path.join("config", self.algo, self.task, config_lable)) as f:
            self.config = json.load(f)
        time_now  = int(round(time.time()*1000))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time_now/1000))
        self.config["log_dir"] = os.path.join(self.config["log_dir"], timestamp)
        
        self.mod_log(self.label)
            
    def mod_log(self, label=None):
        if label:
            self.config["log_dir"] += " (" + label + ")"

    def run(self):
        if self.algo == "arcoo":
            from core.arcoo.__init__ import arcoo
            arcoo(self.config)
    def save_config(self, config_lable='default.json'):
        p = '/'
        for w in self.config['log_dir'].split('/')[:-1]:
            p = os.path.join(p, w)
        self.config['log_dir'] = p
        
        with open(os.path.join("config", self.algo, self.task, config_lable), 'w') as f:
            conf = json.dumps(self.config)
            f.write(conf)