def BIN_config_SIDE():
    config = {}
    config['batch_size'] = 200
    config['train_epoch'] = 100
    config['max_RNA1_seq'] =  98
    config['max_RNA2_seq'] = 98
    config['emb_size'] = 64
    config['dropout_rate'] = 0.01
    
    #DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3
    
    # Encoder
    config['intermediate_size'] = 128
    config['num_attention_heads'] = 8
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 27648
    return config