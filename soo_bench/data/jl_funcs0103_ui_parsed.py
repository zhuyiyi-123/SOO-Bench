from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict

from revive.computation.modules import MLP, DistributionWrapper, FeedForwardPolicy
from revive.computation.graph import NetworkDecisionNode

GLOBAL_BING_ZS = np.array([
    1000., 1250., 1500., 1750., 2000., 2250., 2500., 2650., 2750., 3000.,
    3500., 4000., 4500., 5000., 5500.
])
GLOBAL_BING_NJ = np.array([
    70.8, 94.1, 117., 130.5, 131., 142., 142.8, 142., 141.3, 142.4, 141.9,
    141.7, 141.2, 141., 141.
])
FIXED_CHUAN_NEEDW_INDEX = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 60,
                                    70, 80])

FIXED_CHUAN_NEEDW_INDEX_SEARCH = np.array([1, 15, 16, 17,
                                           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 60,
                                           70, 80])


def w_2_t(w, v, factor=3600 / (np.pi * 0.337 * (2 * 2 ** 2))):
    """
    Demand power -> demand torque

    :param w: Demand power, power
    :param v: Vehicle speed
    :param factor: Multiplier
    :return: Demand torque, torque
    """
    t = w / v * factor
    np.nan_to_num(t, 0.)  # When speed is empty, the default at this time is 0
    return t


def v_2_engine_v(v, factor=24.40060948):
    """
    In parallel mode, the speed -> engine speed
    """
    engine_v = v * factor
    return engine_v


def mode_policy(inputs):
    """
    According to the previous mode, the current brake, the current soc, the current speed, and the current demand power to calculate the current mode
    """
    last_mode = inputs[:1]
    a_driver = inputs[1:2]
    _soc = inputs[2:3]
    _speed = inputs[3:4]
    conditions = inputs[4:-1]
    mode_ver = inputs[-1:]
    # Stop
    if _speed < 0.08 and a_driver == 0:
        return 3
    # Braking
    if a_driver <= 0:
        return 4

    last_mode = last_mode[0]
    # New rule v1
    if mode_ver == 1:
        if _speed < 45:
            return 1
        return 5
    # New rule v2
    elif mode_ver == 2:
        if last_mode in [3, 4] and a_driver > 1e-3:
            return 1
        if last_mode == 1 and _speed >= 45 and a_driver >= 6:
            return 5
        if last_mode == 5 and _speed < 45:
            return 1
        return last_mode
    # Original rules
    else:
        # 3,4->1
        if last_mode in [3, 4] and a_driver > 1e-3:
            return 1
        # 1->2
        if last_mode == 1:
            if _soc < conditions[0] or (_soc < conditions[1]
                                        and a_driver > conditions[2]):
                last_mode = 2
        # 5->2
        if last_mode == 5:
            if _soc > conditions[5] or _speed < conditions[6]:
                last_mode = 2
            # if _soc > conditions[3] and _speed < 99.203:
            #     last_mode = 1
            #     return last_mode

        # 2->1,5
        if last_mode == 2:
            if _soc > conditions[3]:
                last_mode = 1
                return last_mode
            if _speed > conditions[4]:
                last_mode = 5
    return last_mode


def engine_policy(inputs):
    """
    According to the previous mode, the current brake, the current soc, the current speed, and the current demand power to calculate the current mode
    """
    mode = inputs[:1]
    speed = inputs[1:2]
    need_w = inputs[2:3]
    # need_w = np.clip(a_driver, 0, np.inf)
    engine_index = inputs[3:]
    zhuansu_index = engine_index[:53]
    niuju_index = engine_index[53:106]
    bing_index = engine_index[106:]
    if mode == 2:
        chuan_zhuansu_p = interp1d(FIXED_CHUAN_NEEDW_INDEX, zhuansu_index, fill_value='extrapolate')
        chuan_niuju_p = interp1d(FIXED_CHUAN_NEEDW_INDEX, niuju_index, fill_value='extrapolate')
        engine_zhuansu = chuan_zhuansu_p(need_w)
        engine_niuju = chuan_niuju_p(need_w)
    elif mode == 5:
        bing_needed_t = w_2_t(need_w, speed)
        engine_zhuansu = v_2_engine_v(speed)
        bing_niuju_p = interp1d(GLOBAL_BING_ZS, GLOBAL_BING_NJ, fill_value='extrapolate')
        bing_niuju = bing_niuju_p(engine_zhuansu)
        engine_niuju = np.clip(bing_needed_t, bing_index[-1] * bing_niuju, bing_index[0] * bing_niuju)
    else:
        engine_zhuansu, engine_niuju = 0., 0.
    return np.array([engine_zhuansu, engine_niuju]).flatten()


def next_mode_node(data: Dict[str, np.ndarray]):
    """
    Expert function node, output the current混动 mode, next_mode corresponds to the node name in the revive graph, the same as next transition.
    Input: The previous mode, the current SOC, the current speed, the current brake, the current demand power
    Output: The current mode
    """
    mode = data.get('mode')
    obs = data.get('obs')
    soc = data.get('soc')
    a_driver = data.get('driver')
    mode_index = data.get("mode_index")
    mode_ver = data.get("mode_ver")
    original_shape = mode.shape

    mode = mode.reshape(-1, 1)
    mode_index = mode_index.reshape(-1, 7)
    mode_ver = mode_ver.reshape(-1, 1)
    mode_index = mode_index if mode.shape[:-1] == mode_index.shape[:-1] else \
        mode_index.repeat(mode.shape[0] / mode_index.shape[0], axis=0)
    soc = soc[..., 0].reshape(-1, 1)
    speed = obs[..., 0].reshape(-1, 1)
    a_driver = a_driver.reshape(-1, 1)
    inputs = np.concatenate([mode, a_driver, soc, speed, mode_index, mode_ver], axis=-1)
    outputs = np.apply_along_axis(mode_policy, 1, inputs)
    return outputs.reshape(original_shape).astype(np.float32)


def engine_node(data: Dict[str, np.ndarray]):
    """
    Expert function node, output the current engine speed and torque.
    1. Series function: input
    2. Parallel function: output
    3. Except for the series and parallel functions, EV, brake, parking and other situations: the torque is 0 by default.
    :param data: Input: current mode, demand power, speed.
    :return:   Output: engine speed, torque.
    """
    # print("============")
    mode = data.get('next_mode')
    obs = data.get('obs')
    a_driver = data.get('driver')
    engine_index = data.get("engine_index")
    original_shape = [*mode.shape[:-1], 2]
    mode = mode.reshape(-1, 1)
    speed = obs[..., 0].reshape(-1, 1)
    a_driver = a_driver.reshape(-1, 1)
    engine_index = engine_index.reshape(-1, 108)
    # should be unannotated when training in Revive Web
    engine_index = np.concatenate([engine_index[:, 2:], engine_index[:, :2]], axis=1)  # should be annotated when plot
    a_driver.clip(0)
    inputs = np.concatenate([mode, speed, a_driver, engine_index], axis=-1)
    outputs = np.apply_along_axis(engine_policy, 1, inputs)
    return outputs.reshape(original_shape).astype(np.float32)


def clip_engine_node(data: Dict[str, np.ndarray]):
    mode = data.get('next_mode')
    engine = data.get('engine')
    mode_one_hot = np.eye(5)[(mode - 1).astype(np.int_)]
    chuan_mode = mode_one_hot[..., 1]
    bing_mode = mode_one_hot[..., 4]
    flag = chuan_mode + bing_mode
    engine *= flag
    return engine


def next_soc_node(data: Dict[str, np.ndarray]):
    soc = data.get("soc")
    delta_soc = data.get("delta_soc")
    next_soc = soc + delta_soc
    return next_soc


def clip_m1_node(data: Dict[str, np.ndarray]):
    mode = data.get("next_mode")
    m1 = data.get("m1")
    mode_one_hot = np.eye(5)[(mode - 1).astype(np.int_)]
    chuan_mode = mode_one_hot[..., 1]
    bing_mode = mode_one_hot[..., 4]
    m1[..., None, 0] *= chuan_mode + bing_mode
    m1[..., None, 1] *= chuan_mode
    return m1


def clip_m2_node(data: Dict[str, np.ndarray]):
    mode = data.get("next_mode")
    m2 = data.get("m2")
    mode_one_hot = np.eye(5)[(mode - 1).astype(np.int_)]
    ev_mode = mode_one_hot[..., 0]
    chuan_mode = mode_one_hot[..., 1]
    brake_mode = mode_one_hot[..., 3]
    bing_mode = mode_one_hot[..., 4]
    m2_flag = ev_mode + chuan_mode + brake_mode + bing_mode
    m2 *= m2_flag
    return m2


if __name__ == "__main__":
    BASE_PARAMS = {
        "mode_index": [42, 50, 15, 50, 60, 60, 55],
        "engine_index": {
            "chuan_xqgl": [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 60, 70, 80
            ],
            "chuan_speed": [
                1500., 1500., 1500., 1500., 1500., 1500., 1500., 1500., 1500.,
                1500., 1500., 1500., 1500., 1500., 1504.45454542, 1562.32727277,
                1620.20000011, 1678.07272746, 1735.9454548, 1773.70491816,
                1805.01311485, 1836.32131155, 1867.62950824, 1898.93770493,
                1930.24590163, 1961.55409832, 1992.86229501, 2030.97478985,
                2071.09663862, 2111.21848739, 2151.34033616, 2191.46218493,
                2231.5840337, 2286.38028187, 2353.62676077, 2420.87323967,
                2488.11971857, 2555.36619727, 2622.61267592, 2689.85915467,
                2757.10563349, 2824.35211239, 2891.59859129, 2958.84507018, 3000.,
                3000., 3000., 3000., 3000., 3000., 3000., 3000., 3000.
            ],
            "chuan_torque": [
                95., 95., 95., 95., 95., 95., 95., 95., 95., 95., 95., 95., 95.,
                95., 95.17818182, 97.49309091, 99.808, 102.1229091, 104.43781819,
                107.37049182, 110.50131149, 113.63213115, 116.76295082,
                119.89377049, 123.02459016, 126.15540983, 129.2862295,
                131.48678991, 133.41263865, 135.33848739, 137.26433614,
                139.19018488, 141.11603362, 142., 142., 142., 142., 142., 142.,
                142., 142., 142., 142., 142., 142., 142., 142., 142., 142., 142.,
                142., 142., 142.
            ],
            "bing_limit": [1, 0.7],
            "bing_speed": [
                1000., 1250., 1500., 1750., 2000., 2250., 2500., 2650., 2750.,
                3000., 3500., 4000., 4500., 5000., 5500.
            ],
            "bing_torque": [
                70.8, 94.1, 117., 130.5, 131., 142., 142.8, 142., 141.3, 142.4,
                141.9, 141.7, 141.2, 141., 141.
            ],
        }
    }
    inputs = {
        "next_mode": [2],
        "obs": [60, 3],
        "driver": [3],
        "engine_index": BASE_PARAMS["engine_index"]["chuan_speed"] + BASE_PARAMS["engine_index"]["chuan_torque"]
                        + BASE_PARAMS["engine_index"]["bing_limit"]
    }
    bing_up = [1.001900438618351, 1.1502747422963755, 1.2409807404124298, 1.15610290558843, 1.181752982455563,
               1.2686453076969255, 1.0327763149759408, 1.1379900910710767, 1.258295047586033, 1.3408162806282078,
               1.0343957141500058, 1.1833749681402737, 1.2423123279170345, 1.3785328103755046]
    bing_down = [0.8687531371254823, 0.7564017450878473, 0.8478712816977994, 0.8125440093001004,
                 0.8163066169669907, 0.8916213599325444, 0.7545151760304187, 0.8433590494342222, 0.7188489721474843,
                 0.8976326047794624, 0.8118948062948732, 0.8455359104632356, 0.7881981648683251, 0.7015133656967698]
    # inputs["engine_index"][:14] = 1.0 + 0.5 * np.random.random(14)
    # inputs["engine_index"][53:67] = 0.7 + 0.2 * np.random.random(14)
    inputs["engine_index"][:14] = bing_up
    inputs["engine_index"][53:67] = bing_down
    inputs = {k: np.array(v).reshape(1, -1).repeat(3, axis=0) for k, v in inputs.items()}
    print(inputs["obs"].shape)
    outs = engine_node(inputs)
    print(outs)
