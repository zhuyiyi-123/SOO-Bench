import os
import sys
import time
import uuid
import yaml
import hashlib
import base64
import requests
import tempfile
import importlib
from loguru import logger

from revive.conf.config import base_url


def customer_createTrain(machineCode: str,
                         trainModelSimulatorTotalCount: str,
                         trainModelPolicyTotalCount: str,
                         trainDataRowsCount: str,
                         yamlNodeCount: str,
                         yamlFileClientUrl: str,
                         configFileClientUrl: str,
                         logFileClientUrl: str,
                         userPrivateKey: str,
                        ):
    """ 
    Verify the user's training privileges.
    
    API Reference: https://polixir.yuque.com/puhlon/rwxlag/gu7pg8#uFKnl 
    """
    url = base_url + "api/customer/createTrain"
    
    machineCode = base64.b64encode(machineCode.encode("utf-8")).decode()
    payload={'machineCode': machineCode,
             'trainModelSimulatorTotalCount': trainModelSimulatorTotalCount,
             'trainModelPolicyTotalCount': trainModelPolicyTotalCount,
             'trainDataRowsCount': trainDataRowsCount,
             'yamlNodeCount': yamlNodeCount,
             'yamlFileClientUrl': yamlFileClientUrl,
             'configFileClientUrl': configFileClientUrl,
             'logFileClientUrl': logFileClientUrl,
             'userPrivateKey': userPrivateKey}
             
    files=[
    ('yamlFile',(os.path.basename(yamlFileClientUrl),open(yamlFileClientUrl,'rb'),'application/octet-stream')),
    ('configFile',(os.path.basename(configFileClientUrl),open(configFileClientUrl,'rb'),'application/octet-stream'))
    ]

    headers = {
    'requestId': uuid.uuid4().hex,
    # 'Authorization': 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI5NTE5MzA4OTY5MzA0MzkxNjgiLCJleHAiOjE2NDkyMzU0Mjh9.HRb-btXtwI1_XVe9kLLwVW7y5zhpfOlc-Jp1GFWP2_Y'
    }
    response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=10)
    logger.info("customer_createTrain,{eval(response.text)['code']}")

    return eval(response.text)


def customer_uploadTrainFile(trainId: str,
                             accessToken: str,
                             yamlFile: str = None,
                             configFile: str = None,
                             logFile: str = None,
                             
                            ):
    """ 
    Upload the history train log.
    
    API Reference: https://polixir.yuque.com/puhlon/rwxlag/gu7pg8#r5IPw
    """
    url = base_url + "api/customer/uploadTrainFile"

    upload_files = {name: file for name,file in zip(['yamlFile', 'configFile', 'logFile'],[yamlFile, configFile, logFile]) if file is not None}
    payload_files = {name: file for name,file in zip(['yamlFileClientUrl', 'configFileClientUrl', 'logFileClientUrl'],[yamlFile, configFile, logFile]) if file is not None}

    payload={'trainId': trainId,}
    payload.update(payload_files)

    tmp = tempfile.NamedTemporaryFile(delete=True)
    files = {file:(file_name,tmp,'application/octet-stream') for file,file_name in zip(['yamlFile', 'configFile', 'logFile'],['none.yaml', 'none.json', 'none.log'])}
    for name, file in upload_files.items():
        files[name] = (os.path.basename(file),open(file,'rb'),'application/octet-stream')

    files = [(k,v) for k,v in files.items()]
    
    headers = {'requestId': uuid.uuid4().hex,
               'Authorization': accessToken,
              }

    response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=10)
    tmp.close()
    logger.info(f"customer_uploadTrainFile,{eval(response.text)['code']}")

    return eval(response.text)

def customer_uploadTrainLog(trainId: str,
                            logFile: str,
                            trainType: str,
                            trainResult: str,
                            trainScore: str,
                            accessToken: str,
                            ):
    """ 
    Upload the log after a trail is trained.
    
    API Reference: https://polixir.yuque.com/puhlon/rwxlag/gu7pg8#KvKWx
    """
    url = base_url + "api/customer/uploadTrainLog"

    payload={'trainId': trainId,
             'trainType': trainType,
             'trainResult' : trainResult,
             'trainScore' : trainScore}
    
    files=[('logFile',(os.path.basename(logFile),open(logFile,'rb'),'application/octet-stream')),]
    headers = {'requestId': uuid.uuid4().hex,
               'Authorization': accessToken,
              }
    response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=10)
    logger.info(f"customer_uploadTrainLog,{eval(response.text)['code']}")
    
    return eval(response.text)

def check_license(cls):
    try:
        REVIVE_LICENSE = os.getenv("REVIVE_LICENSE")
        sys.PYARMOR_LICENSE = REVIVE_LICENSE
        if not sys.PYARMOR_LICENSE:
            sys.PYARMOR_LICENSE = "/"
        
        try:
            importlib.import_module(f'revive.algo.venv.revive')
            logger.info(f"import revive.algo.venv.revive")
        except:
            if not isinstance(REVIVE_LICENSE, str):
                logger.info(f"Don't find 'REVIVE_LICENSE' in environment variables.")
                raise NotImplementedError
            else:
                try:
                    importlib.import_module(f'revive.dist.algo.venv.revive')
                    logger.info(f"import revive.dist.algo.venv.revive")
                except:
                    logger.info(f"Can't find local REVIVE_LICENSE file.")
                    raise NotImplementedError
    except:
        logger.info(f"Get online REVIVE_LICENSE file.")
        from revive.utils.license_utils import get_machine_info
        machineCode = get_machine_info(os.path.join("/tmp/machine_info.json"), online=True)
        

        if cls.venv_mode is None and cls.policy_mode is None:
            logger.warning(f"Don't train venv and policy. Please check the venv_mode and policy_mode.")
            sys.exit()

        trainModelSimulatorTotalCount = 0
        
        if cls.venv_mode == "once":
            trainModelSimulatorTotalCount += 1
        elif cls.venv_mode == "tune":
            trainModelSimulatorTotalCount += cls.config["train_venv_trials"]
        else:
            pass
        
        trainModelPolicyTotalCount = 0
        if cls.policy_mode == "once":
            trainModelPolicyTotalCount += 1
        elif cls.policy_mode == "tune":
            trainModelPolicyTotalCount += cls.config["train_policy_trials"]
        else:
            pass

        trainDataRowsCount = len(cls.dataset)
        yamlNodeCount = len(cls.dataset.graph)
        yamlFileClientUrl = os.path.abspath(cls.config_file)
        configFileClientUrl = os.path.abspath(cls.revive_config_file_path)
        logFileClientUrl = os.path.abspath(cls.log_path)

        config_folder = os.path.join(os.path.expanduser('~'),".revive")
        with open(os.path.join(config_folder,'config.yaml'), 'r', encoding='utf-8') as f:
            revive_config = yaml.load(f, Loader=yaml.FullLoader)
        
        if "accesskey" not in revive_config.keys():
            logger.error(f"Please check the ``~/.revive/config.yaml`` file for the configuration.")
            sys.exit() 
        userPrivateKey = revive_config["accesskey"]
        
        args = {'machineCode': machineCode,
                'trainModelSimulatorTotalCount': trainModelSimulatorTotalCount,
                'trainModelPolicyTotalCount': trainModelPolicyTotalCount,
                'trainDataRowsCount': trainDataRowsCount,
                'yamlNodeCount': yamlNodeCount,
                'yamlFileClientUrl': yamlFileClientUrl,
                'configFileClientUrl': configFileClientUrl,
                'logFileClientUrl': logFileClientUrl,
                'userPrivateKey': userPrivateKey}

        args = {k:str(v) for k,v in args.items()}

        for i in range(1):
            result = customer_createTrain(**args)
            if result["code"] == "General.Success":
                trainId = result["data"]["trainId"]
                license = result["data"]["license"]
                accessToken = result["data"]["accessToken"]

                cls.config["trainId"] = trainId
                cls.config["accessToken"] = accessToken

                license_path = os.path.abspath(os.path.join(cls.workspace, "license.lic"))
                
                with open(license_path, "w") as f:
                    f.writelines([license+"\n",])
                
                sys.PYARMOR_LICENSE = license_path
                time.sleep(1)
                break
        logger.info(f"{result['code']}")
        if result["code"] == "General.Success":
            try:
                if result["data"]["beforeTrainId"]:
                    # logger.info(f"{result['msg']}")
                    args = {'trainId': result["data"]["beforeTrainId"],
                            'accessToken': result["data"]["accessToken"],}
                    if result["data"]["needYamlFileClientUrl"]:
                        if os.path.exists(result["data"]["needYamlFileClientUrl"]):
                            args["yamlFile"] = result["data"]["needYamlFileClientUrl"]
                    if result["data"]["needConfigFileClientUrl"]:
                        if os.path.exists(result["data"]["needConfigFileClientUrl"]):
                            args["configFile"] = result["data"]["needConfigFileClientUrl"]
                    if result["data"]["needLogFileClientUrl"]:
                        if os.path.exists(result["data"]["needLogFileClientUrl"]):
                            args["logFile"] = result["data"]["needLogFileClientUrl"]
                    if len(args) > 2:
                        customer_uploadTrainFile(**args)
            except Exception as e:
                logger.info(f"{e}")
        else:
            logger.error(f"{result['code']}. Please check the ``~/.revive/config.yaml`` file for the configuration.")
            sys.exit() 

        # importlib.import_module(f'revive.dist.algo.venv.ppo')
        logger.info(f"Import encryption venv algorithm module -> ppo!")