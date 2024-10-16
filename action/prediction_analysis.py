import json
import glob
import os
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
    def __init__(self, save_folder = '.', rank = 0):
        self.save_folder = save_folder
        self.rank = rank
        self.prefix = 'prediction_analysis_buf'
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

            print (sorted(list(self.data.keys()), key = lambda x: int(x)))

    def wrong_verb(self):

        N = len(self.data)
        llava_wrong_verb_collections = []
        llava_wrong_noun_collections = []
        llava_wrong_verb_noun_collections = []

        avion_wrong_verb_collections = []
        avion_wrong_noun_collections = []
        avion_wrong_verb_noun_collections = []

        wrong_llava_collections = []
        wrong_avion_collections = []

        indices = sorted(list(self.data.keys()), key = lambda x: int(x))

        for index in indices:
            items = self.data[index]
            llava_pred = items['llava_pred']
            gt_name = items['gt_name']
            # only replacing the first : 
            avion_pred = items['avion_preds']['predictions'][0].replace(':', ' ', 1)
            
            if llava_pred != gt_name:
                wrong_llava_collections.append((llava_pred, gt_name))
            if avion_pred!= gt_name:
                # pred, gt
                wrong_avion_collections.append((avion_pred, gt_name))
            

if __name__ == '__main__':


    prediction_analysis = PredictionAnalysis(save_folder = '/storage-rcp-pure/upmwmathis_scratch/shaokai/LLaVA-NeXT')
    prediction_analysis.load()
