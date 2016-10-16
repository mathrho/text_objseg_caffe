class Config():
    def __init__(self):
        # Model Params
        self.T = 20
        self.N = 50
        self.input_H = 224 
        self.input_W = 224 
        self.featmap_H = (self.input_H // 32)
        self.featmap_W = (self.input_W // 32)
        self.vocab_size = 8803
        self.embed_dim = 1000
        self.lstm_dim = 1000
        self.mlp_hidden_dims = 500

        # Training Params
        self.gpu_id = 2
        self.max_iter = 25000

        self.weights = './VGG_ILSVRC_16_layers.caffemodel'
        self.fix_vgg = True  # set False to finetune VGG net
        self.vgg_dropout = False
        self.mlp_dropout = False

        # Data Params
        self.data_provider = 'referit_data_provider'
        self.data_provider_layer = 'ReferitDataProviderLayer'

        self.data_folder = './referit/data/train_batch_det/'
        self.data_prefix = 'referit_train_det'


