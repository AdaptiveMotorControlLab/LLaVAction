import numpy as np

import ast
from PIL import Image, ImageDraw, ImageFont

color_rgb = [(255,255,0), (255, 128,0), (128,255,0), (0,128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127), (255,0,0), (255,204,153), (255,102,102), (153,255,153), (153,153,255), (0,0,153)]
color_rgba = [(255,255,0,70), (255, 128,0,70), (128,255,0,70), (0,128,255,70), (0,0,255,70), (127,0,255,70), (255,0,255,70), (255,0,127,70), (255,0,0,70), (255,204,153,70), (255,102,102,70), (153,255,153,70), (153,153,255,70), (0,0,153,70)]


hand_rgb = [(0, 90, 181), (220, 50, 32)] 
hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]

obj_rgb = (255, 194, 10)
obj_rgba = (255, 194, 10, 70)

side_map = {'l':'Left', 'r':'Right'}
side_map2 = {0:'Left', 1:'Right'}
side_map3 = {0:'L', 1:'R'}
state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}

vis_settings = {'font_size':20, 'line_width':2, 'point_radius':4, 'hand_color':hand_rgb, 'hand_alpha':[None, None], 'obj_color':obj_rgb, 'obj_alpha':None, 'text_alpha':(255, 255, 255, 255)}

def calculate_center(bb):
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

def filter_object(obj_dets, hand_dets):
    filtered_object = []
    object_cc_list = []
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)
    img_obj_id = []
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0:
            img_obj_id.append(-1)
            continue
        hand_cc = np.array(calculate_center(hand_dets[i,:4]))
        point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist)
        img_obj_id.append(dist_min)
    return img_obj_id

def draw_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height):
    font = ImageFont.truetype('llavaction/action/times_b.ttf', size=vis_settings['font_size'])
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(obj_bbox, outline=vis_settings['obj_color'], width=vis_settings['line_width'], fill=vis_settings['obj_alpha']) 
    image.paste(mask, (0,0), mask)  

    draw.rectangle([obj_bbox[0], max(0, obj_bbox[1]-vis_settings['font_size']), obj_bbox[0]+vis_settings['font_size']+2, 
                    max(0, obj_bbox[1]-vis_settings['font_size'])+vis_settings['font_size']], 
                    fill=vis_settings['text_alpha'], outline=vis_settings['obj_color'], width=vis_settings['line_width'])
    draw.text((obj_bbox[0]+5, max(0, obj_bbox[1]-vis_settings['font_size'])-2), f'O', font=font, fill=(0,0,0)) #

    return image

def draw_hand_mask(image, draw, hand_idx, hand_bbox, hand_score, side, state, width, height):
    font = ImageFont.truetype('llavaction/action/times_b.ttf', size=vis_settings['font_size'])
    if side == 0:
        side_idx = 0
    elif side == 1:
        side_idx = 1
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(hand_bbox, outline=vis_settings['hand_color'][side_idx], width=vis_settings['line_width'], fill=vis_settings['hand_alpha'][side_idx])
    image.paste(mask, (0,0), mask)
    # text
    
    draw = ImageDraw.Draw(image)
    draw.rectangle([hand_bbox[0], max(0, hand_bbox[1]-vis_settings['font_size']), hand_bbox[0]+vis_settings['font_size']*2+2, 
                    max(0, hand_bbox[1]-vis_settings['font_size'])+vis_settings['font_size']], 
                    fill=vis_settings['text_alpha'], outline=vis_settings['hand_color'][side_idx], width=vis_settings['line_width'])
    draw.text((hand_bbox[0]+6, max(0, hand_bbox[1]-vis_settings['font_size'])-2), f'{side_map3[int(float(side))]}-{state_map2[int(float(state))]}', font=font, fill=(0,0,0)) # 

    return image
    
def draw_line_point(draw, side_idx, hand_center, object_center):
    
    draw.line([hand_center, object_center], fill=vis_settings['hand_color'][side_idx], width=vis_settings['line_width'])
    x, y = hand_center[0], hand_center[1]
    r=vis_settings['point_radius']
    draw.ellipse((x-r, y-r, x+r, y+r), fill=vis_settings['hand_color'][side_idx])
    x, y = object_center[0], object_center[1]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=vis_settings['obj_color'])

def vis_detections_PIL(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for hand_idx, i in enumerate(range(np.minimum(10, dets.shape[0]))):
        bbox = list(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, 4]
        lr = dets[i, -1]
        state = dets[i, 5]
        if score > thresh:
            image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height)
            
    return image

def vis_detections_filtered_objects_PIL(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01):

    # convert to PIL
    im = im[:,:,::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size 

    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            if score > thresh_obj and i in img_obj_id:
                # viz obj by PIL
                image = draw_obj_mask(image, draw, obj_idx, bbox, score, width, height)

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -1]
            state = hand_dets[i, 5]
            if score > thresh_hand:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height)

                if state > 0: # in contact hand

                    obj_cc, hand_cc =  calculate_center(obj_dets[img_obj_id[i],:4]), calculate_center(bbox)
                    # viz line by PIL
                    if lr == 0:
                        side_idx = 0
                    elif lr == 1:
                        side_idx = 1
                    draw_line_point(draw, side_idx, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))

    elif hand_dets is not None:
        image = vis_detections_PIL(im, 'hand', hand_dets, thresh_hand)
        
    return image

def render_frame(im, hand_dets, obj_dets, thresh_hand=0.5, thresh_obj=0.5):
    import cv2
    im_show = im.copy()
    im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
    hand_dets = np.array(ast.literal_eval(hand_dets)) if hand_dets != '[]' else None
    obj_dets = np.array(ast.literal_eval(obj_dets)) if obj_dets != '[]' else None
    im_show = vis_detections_filtered_objects_PIL(im_show, obj_dets, hand_dets, thresh_hand, thresh_obj)
    # im_show.save('test.png')
    im_show = np.array(im_show)
    return im_show