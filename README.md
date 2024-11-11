# PyTorch Implementation of DQN with MDP 2 VEC
	
	to run the code with out experiment of Cartpole, you must modify your cartpole gym environment following 
	https://github.com/alfred-cueva/gym/tree/main
	
### Hyperparameters Used
    
    #Latent Vector z's length
    latent_size = 16 
    task_vec_size = 13

    learning_start_rl = 400
    learning_freq_rl = 1
    target_freq = 200
    gamma = 0.99
    memory_size = 4000000
    batch_size_rl = 32

    epsilon_minimum = 0.01
    epsilon_decay_rate = 0.9999
    learning_rate_dqn = 0.001
    learning_rate_embd = 0.001

    learning_rate_adaptor = 0.001
    epoch_adaptor = 50
    batch_size_adaptor = 32

    max_step = 4000000    # 40M steps max


