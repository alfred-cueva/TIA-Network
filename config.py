class AgentConfig:
    
    #Latent Vector z's length
    latent_size = 16 #128
    #Length of memory buffer tau
    tau_max_length = 200
    pred_weight = .5
    td_weight = .5

    TD_step = 3

    learning_start_dqn = 100
    # learning_freq_dqn = 30
    learning_freq_dqn = 1
    learning_start_encd = 20
    learning_freq_encd = 10
    learning_freq_pred = 10
    learning_start_pred = 100


    # Learning
    gamma = 0.99 #0.99
    memory_size = 1000 #2002
    batch_size_rl_dqn = 32
    batch_size_rl_pred = 16
    batch_size_seq = 16

    epsilon_minimum = 0.02 #0.0005
    epsilon_decay_rate = 0.9998 #0.99993
    
    weight_decay_dqn = 0.999
    weight_decay_encd = 0.999
    weight_decay_pred = 0.999
    
    alpha_minimum = 0.0001
    alpha_min_run = 0.001
    alpha_decay_rate = 0.9995
    
    learning_rate_dqn = 0.0005
    learning_rate_pred = 0.0005
    learning_rate_encd = 0.0005
    
    target_update_freq = 200

    max_step = 10000000      # 40M steps max
    
    weight_save_min = 30000
    weight_save_freq = 5000

    # RNN
    n_layers =  1 #Dont Change!! 
    enc_dropout = 0
    dec_dropout = 0
    teacher_forcing_ratio = 0.5


    #Progress 
    progress_freq = 100


class EnvConfig:
    # env_list = ['CartPole-v0']
    # env_list = ['CartPoleStay-v0', 'CartPoleStay-v0', 'CartPoleStay-v0', 'CartPoleStay-v0', 'CartPoleStay-v0', 'CartPoleStay-v0']
    env_list = ['CartPoleStay-v0' for i in range(10)]
    # env_list = ['Assault-ram-v4','BeamRider-ram-v4','AirRaid-ram-v4','Enduro-ram-v4','Skiing-ram-v4']
