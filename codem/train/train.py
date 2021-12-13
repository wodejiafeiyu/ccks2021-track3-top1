import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import sys
sys.path.append(os.getcwd())
import pandas as pd
from codem.configm.config import args
import  codem.utils  as ut







def main():

    ut.set_seed(args.seed)
    if args.pretrain:
        train_path1 = 'tcdata/round1_train.txt'
        train_path2 = 'tcdata/round2_train.txt'
        test_path='tcdata/Xeon3NLP_round1_test_20210524.txt'
        #test_path2='tcdata/test2_release.txt' #加上线上的
        train_data1 = ut.get_data(train_path1, 'train')
        train_data2=ut.get_data(train_path2,'train')
        test_data=ut.get_data(test_path,'test')
        test_data['label']=-1
        datat=pd.concat([train_data1,train_data2,test_data])
        pp_dict = {'不匹配': 0, '部分匹配': 1, '完全匹配': 2,-1:-1}

        datat=datat.drop_duplicates(subset=['query','candidate'])
        datat.index=range(len(datat))
        datat['label']=datat['label'].apply(lambda x:pp_dict[x])


        ut.simmodels(model_path+args.model_path,
                     save_path+args.save_path,
                     model_type=args.model_type,).train(datat)


    else:

        pp_dict = {'不匹配': 0, '部分匹配': 1, '完全匹配': 2}

        for i in range(3):
            train_data = pd.read_csv('user_data/kfold/data_KFold_24/data{}/train.csv'.format(i))
            dev_data = pd.read_csv('user_data/kfold/data_KFold_24/data{}/dev.csv'.format(i))
            train_data['label']=train_data['label'].apply(lambda x:pp_dict[x])
            dev_data['label']=dev_data['label'].apply(lambda x: pp_dict[x])

            ut.simmodels(model_path+args.model_path,
                         save_path+args.save_path+f'/{i}',
                         model_type=args.model_type,).train(train_data,dev_data)

if __name__=='__main__':
    if args.pretrain:

        model_path='user_data/model_param/pretrain_model_param/'
        save_path='user_data/model_param/pretrained_model_param/'
    else:
        model_path='user_data/model_param/pretrained_model_param/'
        save_path='user_data/model_param/saved_model/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    main()




