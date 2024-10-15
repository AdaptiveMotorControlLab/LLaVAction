import json
import glob

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
    def __init__(self, save_path):
        self.save_path = save_path
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


class Analysis:
    """

    This same code should be applied to the training too.

    collect all the wrong top-1 prediction from avion
    collect all the wrong top-1 prediction from llava

    Determine percentage of wrong llava prediction that has wrong verb only
    Determine percentage of wrong llava prediction that has wrong noun only
    Determine percentage of wrong llava prediciton that has both verb and noun wrong
    Determine percentage of wrong llava prediction that was wrong because the answer not in the top k
    """
    pass

    def __init__(self, prefix):

        files = glob.glob(prefix + '*')

        self.data = {}

        for file in files:
            print ('loading pred checkpoint from: ', file)
            with open(file, 'r') as f:
                _data = json.load(f)
                self.data.update(_data)

        # add some assertion for number of keys in the data

    def wrong_verb(self):

        N = len(self.data)

        wrong_verb_collections = []
        wrong_noun_collections = []
        wrong_verb_noun_collections = []

        wrong_llava_collections = []
        wrong_avion_collections = []

        indices = sorted(self.data.keys())

        for index in indices:
            items = self.data[index]
        


if __name__ == '__main__':
    pass