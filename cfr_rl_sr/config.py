class NetworkConfig(object):
    scale = 100

    max_step = 1000 * scale

    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1

    save_step = 10 * scale
    max_to_keep = 1000

    Conv2D_out = 128
    Dense_out = 128

    # optimizer = 'RMSprop'
    optimizer = 'Adam'

    logit_clipping = 10  # 10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = 'TE_v2'

    project_name = 'CFR-RL'

    method = 'actor_critic'
    # method = 'pure_policy'

    model_type = 'Conv'

    data_dir = '../data_cfr/'
    # dataset = 'geant_tm'
    dataset = 'abilene_tm'
    topology_file = dataset
    traffic_file = 'train'
    test_traffic_file = 'test'
    timeout = 1
    tm_history = 1

    max_moves = 10  # percentage

    # For pure policy
    baseline = 'avg'  # avg, best

    # For model saving
    model_saved_dir = '../logs/cfr_mssr/tf_ckpts/'

    # evaluate method:
    eval_methods = ['cfr_rl', 'cfr_topk', 'topk', 'optimal', 'sp']
    eval_method = eval_methods[0]


def get_config(FLAGS):
    config = Config

    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)

    return config
