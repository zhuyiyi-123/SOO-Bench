from soo_bench.Taskdata import OfflineTask,set_use_cache
import os


UNCONSTRAINT = ['Arcoo', 'BO', 'CBAS', 'CMAES','CCDDEA','TTDDEA','Trimentoring']

class AlgorithmHelper:
    
    def __init__(self):
        pass

    def init_helper(self, algorithm_name, args, save_path): # called by each algorithm, to initialize the helper
        self.algorithm_name = algorithm_name
        self.args = args
        self.save_path=save_path

        set_use_cache()

        pass

    def _get_file_name(self):
        args = self.args
        save_name = self.algorithm_name + '-' + args.task + '-' + str(args.benchmark) + '-' +str(args.seed) + '-' + str(args.num) + '-'+ str(args.low) + '-' + str(args.high) +'.savedata.pkl'
        return save_name
    

    
    def _get_save_path(self):
        current_dir = os.path.dirname(os.path.abspath(self.script_path))
        for i in range(self.level):
            current_dir = os.path.dirname(current_dir)
        
        root_path = current_dir
        return os.path.join(root_path, 'results', 'unconstraint' if self.unconstraint else 'constraint')

    def _get_full_path(self): 
        save_path = self.save_path
        file_name = self._get_file_name()
        return os.path.join(save_path, file_name)
    
    def need_to_skip(self): # callled before generating samples, determine if we need to skip this run
        file_name = self._get_file_name()
        save_path = self._get_save_path()
        if os.path.exists(os.path.join(save_path, file_name)):
            print('Already exists:', file_name)
            return True
        return False

    def get_sample(self):# called when generating samples
        args = self.args
        task = OfflineTask(args.task, args.benchmark, args.seed)
        num = args.num*task.var_num
        if args.sample_method == 'sample_bound':
            task.sample_bound(num, args.low, args.high)
        elif args.sample_method == 'sample_limit':
            task.sample_limit(num, args.low, args.high)
        return task

    def gather_results(self, scores): # called after algorithm finishes
        os.makedirs(self._get_save_path(), exist_ok=True)
        path = self._get_full_path()
        print('saving to: ', path)
        with open (path, 'wb') as f:
            import pickle
            pickle.dump(scores, f)
        
