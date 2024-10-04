import ray
from ray.tune import Trainable
from ray.tune.resources import Resources
from ray.util.sgd.torch import TorchTrainer, BaseTorchTrainable
from ray.util.sgd.utils import BATCH_SIZE
from ray.util.sgd.torch.worker_group import LocalWorkerGroup, RemoteWorkerGroup
from ray.util.sgd.torch.distributed_torch_runner import \
    LocalDistributedRunner, DistributedTorchRunner
from ray.util.sgd.torch.torch_runner import TorchRunner

class ReviveTorchTrainer(TorchTrainer):
    def __init__(self, **kargv):
        if 'num_gpus_per_worker' in kargv.keys():
            self.num_gpus_per_worker = kargv.pop('num_gpus_per_worker')
        else:
            self.num_gpus_per_worker = 1
        super(ReviveTorchTrainer, self).__init__(**kargv)

    def _start_workers(self, num_workers):
        worker_config = self.config.copy()
        batch_size_per_worker = self._configure_and_split_batch(num_workers)
        if batch_size_per_worker:
            worker_config[BATCH_SIZE] = batch_size_per_worker
        params = dict(
            training_operator_cls=self.training_operator_cls,
            config=worker_config,
            serialize_data_creation=self.serialize_data_creation,
            use_fp16=self.use_fp16,
            use_gpu=self.use_gpu,
            use_tqdm=self.use_tqdm,
            scheduler_step_freq=self.scheduler_step_freq)

        dist_params = dict(
            backend=self.backend,
            add_dist_sampler=self.add_dist_sampler,
            wrap_ddp=self.wrap_ddp)

        worker_args = {
            "max_workers": self.max_replicas,
            "params": params,
            "dist_params": dist_params,
            "initialization_hook": self.initialization_hook,
            "num_cpus_per_worker": self.num_cpus_per_worker,
            "num_gpus_per_worker": self.num_gpus_per_worker, # additional
            "use_gpu": self.use_gpu,
            "timeout_s": self.timeout_s
        }

        # ---------------------------------------------------------------------------------- #

        if self.use_local: # we do not use this
            self.worker_group = LocalWorkerGroup(**worker_args) 
        else:
            self.worker_group = ReviveRemoteWorkerGroup(**worker_args)

        # ---------------------------------------------------------------------------------- #

        # TODO(amogkam): If not enough resources are available to create
        #  num_workers workers, this command will hang. Instead,
        #  start_workers should take into account available resources when
        #  determining how many workers to create.
        self.worker_group.start_workers(num_workers)
        
        # NOTE: This skip the resource check.
        # Temporary solution, should be done by another way.
        return True

    @classmethod
    def as_trainable(cls, *args, **kwargs):

        class TorchTrainable(BaseTorchTrainable):
            @classmethod
            def default_resource_request(cls, config):
                num_workers = config.get("num_workers",
                                         kwargs.get("num_workers", 1))
                num_cpus_per_worker = config.get(
                    "num_cpus_per_worker", kwargs.get("num_cpus_per_worker",
                                                      1))
                num_gpus_per_worker = config.get(
                    "num_gpus_per_worker", kwargs.get("num_gpus_per_worker",
                                                      1))
                use_gpu = config.get("use_gpu", kwargs.get("use_gpu"))
                use_local = config.get("use_local",
                                       kwargs.get("use_local", False))

                if use_local:
                    remote_worker_count = num_workers - 1
                    local_cpus = 1
                    local_gpus = int(use_gpu) * 0.9 * num_gpus_per_worker
                else:
                    remote_worker_count = num_workers
                    local_cpus = 0
                    local_gpus = 0

                return Resources(
                    cpu=int(local_cpus * num_cpus_per_worker),
                    gpu=int(local_gpus) * 0.9 * num_gpus_per_worker,
                    extra_cpu=int(remote_worker_count * num_cpus_per_worker),
                    extra_gpu=int(use_gpu) * 0.9 * num_gpus_per_worker * remote_worker_count)

            def _create_trainer(self, tune_config):
                """Overrides the provided config with Tune config."""
                provided_config = kwargs.get("config", {}).copy()
                provided_config.update(tune_config)
                kwargs["config"] = provided_config
                trainer = ReviveTorchTrainer(*args, **kwargs)
                return trainer

        return TorchTrainable

class ReviveRemoteWorkerGroup(RemoteWorkerGroup):

    def __init__(self, **kwargs):
        if 'num_gpus_per_worker' in kwargs.keys():
            self.num_gpus_per_worker = kwargs.pop('num_gpus_per_worker')
        else:
            self.num_gpus_per_worker = 1
        super(ReviveRemoteWorkerGroup, self).__init__(**kwargs)

    def _init_dist_workers(self, num_workers):
        """Create `num_workers` remote workers."""
        # Generate actor class
        RemoteRunner = ray.remote(
            num_cpus=self._num_cpus_per_worker,
            num_gpus=int(self._use_gpu) * self.num_gpus_per_worker * 0.9)(DistributedTorchRunner)

        # Start workers
        self.remote_workers = [
            RemoteRunner.remote(**{
                **self._params,
                **self._dist_params
            }) for _ in range(num_workers)
        ]

    def start_workers(self, num_workers):
        if num_workers == 1:
            RemoteRunner = ray.remote(
                num_cpus=self._num_cpus_per_worker,
                num_gpus=int(self._use_gpu) * self.num_gpus_per_worker * 0.9)(TorchRunner)
            self.remote_workers = [RemoteRunner.remote(**self._params)]
            ray.get(self.remote_workers[0].setup_operator.remote())
        else:
            self._init_dist_workers(num_workers)

            if self._initialization_hook:
                self.apply_all_workers(self._initialization_hook)

            # Make sure to get the IP address of the rank 0 worker node.
            address = ray.get(self.remote_workers[0].setup_address.remote())

            ray.get(
                self._setup_process_group(
                    address=address, world_size=num_workers))

            ray.get(self._setup_local_rank())
            ray.get(self._setup_operator())
