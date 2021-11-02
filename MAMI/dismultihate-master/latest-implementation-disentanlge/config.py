import argparse 

def parse_opt():
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--DATASET',type=str,default='off')#mmhs / mem / off
    parser.add_argument('--MODEL',type=str,default='latent')
    
    #path configuration
    parser.add_argument('--DATA',type=str,
                        default='/home/ruicao/NLP/datasets/hate-speech/hate-mem')
    parser.add_argument('--MM',type=str,
                        default='/home/ruicao/NLP/datasets/hate-speech/MMHS')
    parser.add_argument('--OFF',type=str,
                        default='/home/ruicao/NLP/datasets/hate-speech/OFF')
    parser.add_argument('--RESULT',type=str,default='./result')
    parser.add_argument('--DICT',type=str,default='./dictionary')
    #fixed parameters
    parser.add_argument('--GAMMA',type=int,default=4)
    parser.add_argument('--NUM_REGIONS',type=int,default=196)
    
    #BAN related parameters
    parser.add_argument('--FEAT_FC_DIM',type=int,default=2048)
    parser.add_argument('--FEAT_POOL_DIM',type=int,default=1024)
    parser.add_argument('--V_HIDDEN',type=int,default=1024)
    
    #hyper parameters configuration
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.5)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.1) 
    parser.add_argument('--ATT_DROPOUT',type=float,default=0.3) 
    parser.add_argument('--MIN_OCC',type=int,default=2)
    parser.add_argument('--BATCH_SIZE',type=int,default=64)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    parser.add_argument('--NUM_EXT',type=int,default=27)
    parser.add_argument('--MID_DIM',type=int,default=512)
    parser.add_argument('--ATT_MID_DIM',type=int,default=128)
    parser.add_argument('--PROJ_DIM',type=int,default=512)
    parser.add_argument('--NUM_HIDDEN',type=int,default=512)
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--NUM_HEAD',type=int,default=8)
    parser.add_argument('--NUM_LATENT',type=int,default=6)
    parser.add_argument('--TRANS_LAYER',type=int,default=1)
    parser.add_argument('--BIDIRECT',type=bool,default=False)
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.3)
    parser.add_argument('--TEMPERATURE',type=float,default=0.3)
    parser.add_argument('--LATENT_RATE',type=float,default=0.04)
    
    parser.add_argument('--LENGTH',type=int,default=25)
    parser.add_argument('--NUM_RACE',type=int,default=8)
    parser.add_argument('--NUM_ENTITY',type=int,default=20)
    
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    parser.add_argument('--DEBUG',type=bool,default=False)
    parser.add_argument('--OFF_EVAL',type=bool,default=False)
    parser.add_argument('--ABLATION',type=bool,default=False)
    parser.add_argument('--SAVE_NUM',type=int,default=0)
    parser.add_argument('--EPOCHS',type=int,default=12)
    
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    parser.add_argument('--CUDA_DEVICE', type=int, default=7)
    
    parser.add_argument('--WARM_UP',type=int,default=2000)
    args=parser.parse_args()
    return args
