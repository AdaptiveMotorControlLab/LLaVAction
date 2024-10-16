import json
import glob
import os
import numpy as np
class PredictionAnalysis:
    """
    We save data that can be used for ad-hoc analysis

    We want to save the following:

    # saving global index to make distributed code work better
    {global_index: {
        llava_pred: pred_name,
        gt_name: pred_name,
        avion_preds: avion_predictions,
        # to locate the video clip
        dataset_name: '',
        start_second: '',    
        end_second: '',
        vid_path: ''
    }
    """
    def __init__(self, save_folder = '.', rank = 0, prefix = 'prediction_analysis_buf'):
        self.save_folder = save_folder
        self.rank = rank
        self.prefix = prefix
        self.save_path = os.path.join(save_folder, f'{self.prefix}_rank{rank}.json')       
        self.data = {}   
    def log(self, 
            global_index,
            llava_pred,
            gt_name,
            avion_preds,
            start_second,
            end_second,
            vid_path,
            dataset_name = 'EK100',
            ):
        self.data[global_index] = {
            'llava_pred': llava_pred,
            'gt_name': gt_name,
            'avion_preds': avion_preds,
            'dataset_name' : dataset_name,
            'start_second' : start_second,
            'end_second': end_second,
            'vid_path': vid_path
        }

        # print ('check what is here')
        # print (self.data[global_index])

    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent = 4)


    def load(self):
        save_folder = self.save_folder
        if self.rank == 0:
            files = glob.glob(os.path.join(save_folder,self.prefix + '*'))
            for file in files:
                print ('loading pred checkpoint from: ', file)
                with open(file, 'r') as f:
                    _data = json.load(f)
                    self.data.update(_data)
            print ('length', len(self.data))
            assert len(self.data) == 9668
            #print (sorted(list(self.data.keys()), key = lambda x: int(x)))

    def analysis(self):

        N = len(self.data)
        llava_wrong_verb_collections = []
        llava_wrong_noun_collections = []
        llava_wrong_verb_noun_collections = []

        avion_wrong_verb_collections = []
        avion_wrong_noun_collections = []
        avion_wrong_verb_noun_collections = []

        wrong_llava_collections = [0] * N
        wrong_avion_collections = [0] * N

        indices = sorted(list(self.data.keys()), key = lambda x: int(x))

        for idx, index in enumerate(indices):
            items = self.data[index]
            llava_pred = items['llava_pred']
            gt_name = items['gt_name']
            # only replacing the first : 
            avion_pred = items['avion_preds']['predictions'][0].replace(':', ' ', 1)
            
            llava_verb, llava_noun = llava_pred.split(' ')
            avion_verb, avion_noun = avion_pred.split(' ')
            gt_verb, gt_noun = gt_name.split(' ')

            if llava_pred != gt_name:               
                wrong_llava_collections[idx] = 0
            else:
                wrong_llava_collections[idx] = 1
            if avion_pred!= gt_name:
                wrong_avion_collections[idx] = 0
            else:
                wrong_avion_collections[idx] = 1

            
            if llava_verb == gt_verb and llava_noun!=gt_noun:
                llava_wrong_noun_collections.append((llava_pred, gt_name))
            if llava_noun == gt_noun and llava_verb!=gt_verb:
                llava_wrong_verb_collections.append((llava_pred, gt_name))
            if llava_noun!= gt_noun and llava_verb!=gt_verb:
                llava_wrong_verb_noun_collections.append((llava_pred, gt_name))

            if avion_verb == gt_verb and avion_noun!=gt_noun:
                avion_wrong_noun_collections.append((avion_pred, gt_name))
            if avion_noun == gt_noun and avion_verb!=gt_verb:
                avion_wrong_verb_collections.append((avion_pred, gt_name))
            if avion_noun!= gt_noun and avion_verb!=gt_verb:
                avion_wrong_verb_noun_collections.append((avion_pred, gt_name))

        wrong_llava_collections = np.array(wrong_llava_collections)
        wrong_avion_collections = np.array(wrong_avion_collections)
        llava_wrong_noun_collections = np.array(llava_wrong_noun_collections)
        llava_wrong_verb_collections = np.array(llava_wrong_verb_collections)
        llava_wrong_verb_noun_collections = np.array(llava_wrong_verb_noun_collections)
        avion_wrong_noun_collections = np.array(avion_wrong_noun_collections)
        avion_wrong_verb_collections = np.array(avion_wrong_verb_collections)
        avion_wrong_verb_noun_collections = np.array(avion_wrong_verb_noun_collections)
                
        # first, the correlation between avion and llava
        correlation = np.corrcoef(wrong_llava_collections, wrong_avion_collections)[0, 1]

        print("Correlation:", correlation)

        print ('llava top1 action accuracy {:.3f}'.format(np.sum(wrong_llava_collections == 1) / len(wrong_llava_collections)))
        print ('avion top1 action accuracy {:.3f}'.format(np.sum(wrong_avion_collections == 1) / len(wrong_avion_collections)))

        print ('llava percentage of wrong noun {:.2f}'.format(len(llava_wrong_noun_collections) / np.sum(wrong_llava_collections == 0)))
        print ('llava percentage of wrong verb {:.2f}'.format(len(llava_wrong_verb_collections) / np.sum(wrong_llava_collections == 0)))
        print ('llava percentage of both verb noun wrong {:.2f}'.format(len(llava_wrong_verb_noun_collections) / np.sum(wrong_llava_collections == 0)))


        print ('avion percentage of wrong noun {:.2f}'.format(len(avion_wrong_noun_collections) / np.sum(wrong_avion_collections == 0)))
        print ('avion percentage of wrong verb {:.2f}'.format(len(avion_wrong_verb_collections) / np.sum(wrong_avion_collections == 0)))
        print ('avion percentage of both verb noun wrong {:.2f}'.format(len(avion_wrong_verb_noun_collections) / np.sum(wrong_avion_collections == 0)))




if __name__ == '__main__':


    prediction_analysis = PredictionAnalysis(save_folder = '/storage-rcp-pure/upmwmathis_scratch/shaokai/LLaVA-NeXT/llavavideo_avion_mc_top10_5epoch_preds',
                                             prefix = 'prediction_analysis_buf')
    prediction_analysis.load()
    prediction_analysis.analysis()
