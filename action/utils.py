import csv 
import numpy as np
import random
import os

def generate_label_map(anno_root):
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}
    # from id to name
    verb_maps = {}
    noun_maps = {}
    for f in [      
        os.path.join(anno_root,'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:
        csv_reader = csv.reader(open(f))
        _ = next(csv_reader)  # skip the header
        for row in csv_reader:
            
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            narration = row[8]
            if row[10] not in verb_maps.keys():
                verb_maps[row[10]] = row[9]
            if row[12] not in noun_maps.keys():
                noun_maps[row[12]] = row[11]

            if vn not in vn_list:
                vn_list.append(vn)
            if vn not in mapping_vn2narration:
                mapping_vn2narration[vn] = [narration]
            else:
                mapping_vn2narration[vn].append(narration)
            # mapping_vn2narration[vn] = [narration]
    vn_list = sorted(vn_list)
    print('# of action= {}'.format(len(vn_list)))
    mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}

    labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
    return labels, mapping_vn2act, verb_maps, noun_maps


def match_answer(pred, gt):
    
    pred = set(pred)
    gt = set(gt)
    
    return pred.intersection(gt) == gt


def parse_avion_predictions(predictions):
    new_predictions = []
    for pred in predictions:
        # the prediction looks like verb:noun1:noun2..
        # we want to it look like verb noun1:noun2 
        first_sep = pred.index(':')
        prediction = pred[:first_sep] + ' ' + pred[first_sep+1:]
        new_predictions.append(prediction)
    return new_predictions


class MultiChoiceGenerator:
    """
    Generating multi choice
    """
    def __init__(self, ann_root):
        self.ann_root = ann_root
        _, self.mapping_vn2act, self.verb_maps, self.noun_maps = generate_label_map(ann_root)
    

    def generate_multi_choice(self, gt_vn, k):
        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list

        """        

        # let v_id and n_id be string type
        gt_v_id, gt_n_id = gt_vn.split(':')    
        assert isinstance(gt_v_id, str) and isinstance(gt_n_id, str)
        gt_v_name, gt_n_name = self.verb_maps[gt_v_id], self.noun_maps[gt_n_id]

        # letters as A, B, C, D, .. Note we maximally support 26 letters
        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]
        vn_list = list(self.mapping_vn2act.keys())
        action_list = [f"{self.verb_maps[e.split(':')[0]]} {self.noun_maps[e.split(':')[1]]}" for e in vn_list]
        wrong_answers = np.random.choice(action_list, size = k-1, replace = False)
        gt_answer = f'{gt_v_name} {gt_n_name}'

        answers = [gt_answer] + list(wrong_answers)
        random.shuffle(answers)

        options = []
        for answer, letter in zip(answers, letters):
            options.append(f'{letter}. {answer}')

        gt_letter = letters[answers.index(gt_answer)]
        data = {
                'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer'},
                'option': {0: options},
                # the correct letter in mc
                # for inspecting
                'gt_answer_letter': {0: gt_letter},
                'gt_answer_name': {0: gt_answer},
                'valid_letters': letters
            }
        
        return data
    
class AvionMultiChoiceGenerator(MultiChoiceGenerator):
    """
    Generate multichoice using avion predictions
    """
    def __init__(self, ann_root):
        super().__init__(ann_root)
    
    def generate_multi_choice(self, gt_vn, avion_predictions, k):
        """
        Generate k multiple choices from gt_vn pairs

        randomly pick 1 letter for gt_vn
        randomly pick k-1 letters from vn_list that is not gt_vn (this is important as avion_predictions can contain correct prediction)        

        """    
        gt_v_id, gt_n_id = gt_vn.split(':')
        gt_v_name, gt_n_name = self.verb_maps[gt_v_id], self.noun_maps[gt_n_id]
        gt_answer = f'{gt_v_name} {gt_n_name}'

        letters = [chr(65+i) for i in range(26)][:k]
        options = list(range(26))[:k]

        # we should have plenty of predictions to select, so let's not always pick the hardest
        assert len(avion_predictions) > 2*k
        avion_predictions = avion_predictions[:k*2]
        avion_predictions = parse_avion_predictions(avion_predictions)
        if gt_answer in avion_predictions:
            avion_predictions.remove(gt_answer)
        # just so that it's not strictly desending with confidence
        random.shuffle(avion_predictions)
        avion_predictions = avion_predictions[:k-1]

        answers = [gt_answer] + avion_predictions
        random.shuffle(answers)

        options = []
        for answer, letter in zip(answers, letters):
            options.append(f'{letter}. {answer}')

        gt_letter = letters[answers.index(gt_answer)]
        
        data = {
                'question': {0: 'the video is an egocentric view of a person. What is the person doing? Pick the the letter that has the correct answer'},
                'option': {0: options},
                # the correct letter in mc
                # for inspecting
                'gt_answer_letter': {0: gt_letter},
                'gt_answer_name': {0: gt_answer},
                'valid_letters': letters
            }        
        return data


if __name__ == '__main__':

    anno_root = "/storage-rcp-pure/upmwmathis_scratch/shaokai/epic-kitchens-100-annotations/"
    #generator = MultiChoiceGenerator(anno_root)
    generator = AvionMultiChoiceGenerator(anno_root)
    import json

    with open('/storage-rcp-pure/upmwmathis_scratch/shaokai/avion_predictions_train.json') as f:
        predictions = json.load(f)

    print (len(predictions))
    print (predictions['0'])
    print (len(predictions['0']['predictions']))
    
    print (generator.generate_multi_choice('3:3',  predictions['0']['predictions'],  5))

    pass