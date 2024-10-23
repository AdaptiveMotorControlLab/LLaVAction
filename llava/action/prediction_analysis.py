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
                with open(file, 'r') as f:
                    _data = json.load(f)
                    self.data.update(_data)

            self.data = {int(k): v for k, v in self.data.items()}
            print ('length', len(self.data))
            assert len(self.data) == 9668
            #print (sorted(list(self.data.keys()), key = lambda x: int(x)))
    def get_wrong_examples(self):
        if not hasattr(self, 'llava_pred_mask'):
            self.analysis()

        wrong_indices = self.llava_pred_mask == 0
        wrong_indices = np.where(wrong_indices)[0]

        wrong_examples = {k: self.data[k] for k in wrong_indices}

        return wrong_examples


    def analysis(self):

        self.load()
        
        N = len(self.data)
        llava_wrong_verb_collections = []
        llava_wrong_noun_collections = []
        llava_wrong_verb_noun_collections = []

        avion_wrong_verb_collections = []
        avion_wrong_noun_collections = []
        avion_wrong_verb_noun_collections = []

        llava_pred_mask = [0] * N
        avion_pred_mask = [0] * N

        indices = sorted(list(self.data.keys()), key = lambda x: int(x))

        for idx, index in enumerate(indices):
            items = self.data[index]
            llava_pred = items['llava_pred']
            gt_name = items['gt_name']
            # only replacing the first : 
            avion_pred = items['avion_preds']['predictions'][0].replace(':', ' ', 1)
            avion_preds = items['avion_preds']['predictions'][:5]
            avion_preds = [e.replace(':', ' ', 1) for e  in avion_preds]
            try:
                llava_verb, llava_noun = llava_pred.split(' ')
            except:
                lst =  llava_pred.split(' ')
                llava_verb, llava_noun = lst[0], lst[1]
            avion_verb, avion_noun = avion_pred.split(' ')
            gt_verb, gt_noun = gt_name.split(' ')

            if llava_pred == gt_name:     
                llava_pred_mask[idx] = 1
           
            if avion_pred== gt_name:
                avion_pred_mask[idx] = 1           
            
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

        llava_pred_mask = np.array(llava_pred_mask)
        avion_pred_mask = np.array(avion_pred_mask)

        self.llava_pred_mask = llava_pred_mask

        llava_wrong_noun_collections = np.array(llava_wrong_noun_collections)
        llava_wrong_verb_collections = np.array(llava_wrong_verb_collections)
        llava_wrong_verb_noun_collections = np.array(llava_wrong_verb_noun_collections)
        avion_wrong_noun_collections = np.array(avion_wrong_noun_collections)
        avion_wrong_verb_collections = np.array(avion_wrong_verb_collections)
        avion_wrong_verb_noun_collections = np.array(avion_wrong_verb_noun_collections)
                
        # first, the correlation between avion and llava
        correlation = np.corrcoef(llava_pred_mask, avion_pred_mask)[0, 1]

        print("Correlation:", correlation)

        print ('llava top1 action accuracy {:.3f}'.format(np.sum(llava_pred_mask == 1) / len(llava_pred_mask)))
        print ('avion top1 action accuracy {:.3f}'.format(np.sum(avion_pred_mask == 1) / len(avion_pred_mask)))

        print ('llava percentage of wrong noun {:.2f}'.format(len(llava_wrong_noun_collections) / np.sum(llava_pred_mask == 0)))
        print ('llava percentage of wrong verb {:.2f}'.format(len(llava_wrong_verb_collections) / np.sum(llava_pred_mask == 0)))
        print ('llava percentage of both verb noun wrong {:.2f}'.format(len(llava_wrong_verb_noun_collections) / np.sum(llava_pred_mask == 0)))


        print ('avion percentage of wrong noun {:.2f}'.format(len(avion_wrong_noun_collections) / np.sum(avion_pred_mask == 0)))
        print ('avion percentage of wrong verb {:.2f}'.format(len(avion_wrong_verb_collections) / np.sum(avion_pred_mask == 0)))
        print ('avion percentage of both verb noun wrong {:.2f}'.format(len(avion_wrong_verb_noun_collections) / np.sum(avion_pred_mask == 0)))




if __name__ == '__main__':

    # at rcp server
    save_folder = '/storage-rcp-pure/upmwmathis_scratch/shaokai/LLaVA-NeXT/llavavideo_avion_mc_top10_5epoch_preds_without_frame_time'
    # at amg0 
    #save_folder = '/data/epic_kitchen/llavavideo_avion_mc_top10_5epoch_preds'


    prediction_analysis = PredictionAnalysis(save_folder = save_folder,
                                             prefix = 'prediction_analysis_buf')
    
    prediction_analysis.analysis()
