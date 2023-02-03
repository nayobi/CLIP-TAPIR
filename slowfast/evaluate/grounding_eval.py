import traceback
import numpy as np
import json
from tqdm import tqdm

def boxiou(bb1,bb2):
    x1 = max(bb1[0],bb2[0])
    y1 = max(bb1[1],bb2[1])
    x2 = min(bb1[2],bb2[2])
    y2 = min(bb1[3],bb2[3])
    if x2<x1 or y2<y1:
        return 0.0
    elif y2==y1 and bb1[1]==bb1[3]==bb2[1]==bb2[3]:
        return 1
    elif x2==x1 and bb1[0]==bb1[2]==bb2[0]==bb2[2]:
        return 1
    inter = (x2-x1)*(y2-y1)
    area1 = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    if (area1+area2-inter)==0:
        breakpoint()
    box_iou = inter/(area1+area2-inter)

    assert box_iou>=0 and box_iou<=1
    return box_iou

def eval_grounding(results,task):
    # breakpoint()
    TP = 0
    FP = 0
    T = 0
    for name in tqdm(results,desc='Evaluating grounding'):
        for ground in results[name]['grounds']:
            if task == 'single_grounding':
                assert len(ground['bboxes'])==len(ground[f'prob_{task}']), f"{ground['bboxes']} & {ground[f'prob_{task}']}"

                ind = np.argmax(ground[f'prob_{task}'])
                gt_box = list(map(float,ground['gt']))
                pred_box = list(map(float,ground['bboxes'][ind]))
                if pred_box == [0.0, 0.0, 1.0, 1.0]:
                    iou = 0
                else:
                    iou = boxiou(gt_box,pred_box)

                if iou>=0.5:
                    TP += 1
                else:
                    FP += 1

                T+=1

            elif task == 'combs_grounding' or task == 'perms_grounding' or task == 'action_grounding':
                assert len(ground['bboxes'])==len(ground[f'prob_{task}']), f"{ground['bboxes']} & {ground[f'prob_{task}']}"

                scores = 1/(1 + np.exp(-np.array(ground[f'prob_{task}'])))
                gt_box1 = list(map(float,ground['gt'][0]))
                gt_box2 = list(map(float,ground['gt'][1]))

                batch_boxes = np.array(ground['bboxes'])
                pred_boxes = batch_boxes[scores>=0.5]

                ious1 = [boxiou(gt_box1,p_box) for p_box in pred_boxes]

                if gt_box2 != [0,0,0,0]:
                    ious2 = [boxiou(gt_box2,p_box) for p_box in pred_boxes]
                    assert len(ious1)==len(ious2)
                    T += max(2,len(ious2))

                    if len(ious1):
                        arg1 = np.argmax(ious1)
                        arg2 = np.argmax(ious2)

                        if ious1[arg1]>=0.5 and ious2[arg2]>=0.5 and arg1!=arg2:
                            TP+=2
                            FP+=max(0,len(ious1)-2)

                        elif ious1[arg1]>=0.5 and ious2[arg2]>=0.5:
                            TP += 1
                            FP += max(1,len(ious1)-1)
                        
                        elif ious1[arg1]>=0.5:
                            TP += 1
                            FP += max(1,len(ious1)-1)
                        
                        elif ious2[arg2]>=0.5:
                            TP += 1
                            FP += max(1,len(ious1)-1)
                        
                        else:
                            FP += max(2,len(ious2))
                    else:
                        FP += max(2,len(ious2))
                else:
                    T += max(1,len(ious1))
                    if len(ious1) and max(ious1)>=0.5:
                        TP += 1
                        FP += len(ious1)-1
                    else:
                        FP += max(1,len(ious1))
            
            elif task == 'phrase_grounding':

                scores = np.argmax(np.array(ground[f'prob_{task}']),axis=1)
                gt_box = np.array(ground['gt'])
                pred_box = np.array(ground['bboxes'])[scores]
                assert len(gt_box)==len(pred_box), f'{gt_box} & {pred_box}'

                # breakpoint()
                for gt_b,p_box in zip(gt_box.tolist(),pred_box.tolist()):
                    if p_box == [0.0, 0.0, 1.0, 1.0]:
                        iou = 0
                    else:
                        iou = boxiou(gt_b,p_box)

                    if iou>=0.5:
                        TP += 1
                    else:
                        FP += 1

                    T+=1
            elif 'phrase' in task:
                # breakpoint()
                # if name == 'CASE001/02328.jpg':
                #     continue

                scores = np.array([x for x in ground[f'prob_{task}'] if x!=[0.0]])
                gt_boxes1 = np.array(ground['gt'][0])
                gt_boxes2 = np.array(ground['gt'][1])
                batch_boxes = np.array(ground['bboxes'])

                for p_id in range(len(scores)):


                    try:
                        gt_box1 = gt_boxes1[p_id].tolist() #list(map(float,ground['gt'][0]))
                        gt_box2 = gt_boxes2[p_id].tolist() #list(map(float,ground['gt'][1]))
                        i_scores = scores[p_id]
                    except:
                        traceback.print_exc()
                        breakpoint()

                    pred_boxes = batch_boxes[i_scores>=0.5].tolist()

                    ious1 = [boxiou(gt_box1,p_box) for p_box in pred_boxes]

                    if gt_box2[:-1] != [0,0,0]:
                        ious2 = [boxiou(gt_box2,p_box) for p_box in pred_boxes]
                        assert len(ious1)==len(ious2)
                        T += max(2,len(ious2))

                        if len(ious1):
                            arg1 = np.argmax(ious1)
                            arg2 = np.argmax(ious2)

                            if ious1[arg1]>=0.5 and ious2[arg2]>=0.5 and arg1!=arg2:
                                TP+=2
                                FP+=max(0,len(ious1)-2)

                            elif ious1[arg1]>=0.5 and ious2[arg2]>=0.5:
                                TP += 1
                                FP += max(1,len(ious1)-1)
                            
                            elif ious1[arg1]>=0.5:
                                TP += 1
                                FP += max(1,len(ious1)-1)
                            
                            elif ious2[arg2]>=0.5:
                                TP += 1
                                FP += max(1,len(ious1)-1)
                            
                            else:
                                FP += max(2,len(ious2))
                        else:
                            FP += max(2,len(ious2))
                    else:
                        T += max(1,len(ious1))
                        if len(ious1) and max(ious1)>=0.5:
                            TP += 1
                            FP += len(ious1)-1
                        else:
                            FP += max(1,len(ious1))
    
    assert T==(TP+FP), "FP: {FP}, TP: {TP}, T: {T}"
    return TP/T