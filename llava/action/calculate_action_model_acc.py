import json

tim_action_path = '/data/shaokai/TIM_PREDS/tim_pred_ids_val.json'
avion_action_path = '/data/shaokai/AVION_PREDS/avion_pred_ids_val.json'

def calc_acc(action_path):

    with open(action_path, 'r') as f:
        data = json.load(f)
            
    # calculate top-1, top-5, top-10, top-20 accuracy

    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    top40 = 0

    for i in range(len(data)):
        preds = data[str(i)]['predictions']
        target = data[str(i)]['target']
        
        if target in preds[:1]:
            top1 += 1
        if target in preds[:5]:
            top5 += 1
        if target in preds[:10]:
            top10 += 1
            
        if len(preds) >= 20:
            if target in preds[:20]:
                top20 += 1
                
        if len(preds) >= 40:
            if target in preds[:40]:
                top40 += 1
            
    print('Top-1 accuracy:', top1/len(data))    
    print('Top-5 accuracy:', top5/len(data))
    print('Top-10 accuracy:', top10/len(data))
    if len(preds) >= 20:
        print('Top-20 accuracy:', top20/len(data))
    if len(preds) >= 40:
        print('Top-40 accuracy:', top40/len(data))

print ('tim accs:')        
calc_acc(tim_action_path)

print ('avion accs:')
calc_acc(avion_action_path)