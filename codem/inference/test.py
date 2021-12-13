import sys
import os
sys.path.append(os.getcwd())

import numpy as np

import  codem.utils  as ut
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]='0'



def output_data(test_data):
    pp_dict={0:'不匹配',1:'部分匹配',2:'完全匹配'}
    test_data['label']=test_data['label'].apply(lambda x: pp_dict[x])

    tmp=[]
    for idx,sub_data in test_data.groupby(by='text_id',sort=False):

        sub_dic={}

        sub_dic['text_id'] = list(sub_data['text_id'])[0]
        sub_dic['query'] = list(sub_data['query'])[0]

        candidate_list=[]
        for q in sub_data.itertuples():
            candidate_dic = {}
            candidate_dic['text']=q.candidate
            candidate_dic['label'] = q.label
            candidate_list.append(candidate_dic)
        sub_dic['candidate']=candidate_list
        tmp.append(sub_dic)
    import  json
    with open ('result.txt','w' ) as f:
        for i in tmp:
            f.write(json.dumps(i,ensure_ascii=False,)+'\n')




def get_model(test_name,test_data,weighted,sd):



    opr = ut.OptimizedRounder()

    pre_label=[]
    pp_dict={'不匹配':0,'部分匹配':1,'完全匹配':2}

    for i in range(3):
        if 'bert' in test_name:
            modeltype='bert'
        else:
            modeltype='nezha'

        pre=ut.simmodels(test_name + f"/{i}",
                     None,
                     model_type=modeltype, ).predict(test_data)



        dev_data = pd.read_csv(f'user_data/kfold/data_KFold_{sd}/data{i}/dev.csv')
        dev_data['label']=dev_data['label'].apply(lambda x: pp_dict[x])

        pre_dev=ut.simmodels(test_name + f"/{i}",
                     None,
                     model_type=modeltype, ).predict(dev_data)


        opr.fit(X=pre_dev, y=list(dev_data['label']))

        pre_label.append(weighted*pre)




    coef = opr.coef_['x']
    sub = opr.predict(np.array(pre_label), coef)

    return sub

def main():




    #test_path='/tcdata/test2_release.txt' #线上文件
    test_path='tcdata/Xeon3NLP_round1_test_20210524.txt'
    test_data = ut.get_data(test_path, 'test')


    tmp=[]
    for i,weighted,ss in test_model_list:

        tmp.extend(get_model(i,test_data,weighted,ss))


    sub_all=np.argmax(np.mean(tmp,axis=0),axis=-1)




    test_data['label']=sub_all

    output_data(test_data)


if __name__=='__main__':


    test_model_name1='user_data/model_param/saved_model/nezha_base/'
    test_model_name2='user_data/model_param/saved_model/nezha_wwm/'
    test_model_name3='user_data/model_param/saved_model/mac_bert/'


    test_model_list=[(test_model_name1,0.45,42),
                     (test_model_name2,0.35,24),
                     (test_model_name3,0.2,33),
                     ]





    main()


