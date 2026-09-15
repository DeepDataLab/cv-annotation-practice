from ultralytics import YOLO 

def calculate_iou(box_a, box_b): 
    xa1, ya1, xa2, ya2 = box_a 
    xb1, yb1, xb2, yb2 = box_b 

    inter_x1 = max(xa1, xb1) 
    inter_y1 = max(ya1, yb1) 
    inter_x2 = min(xa2, xb2) 
    inter_y2 = min(ya2, yb2) 

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1) 

    area_a = (xa2 - xa1) * (ya2 - ya1) 
    area_b = (xb2 - xb1) * (yb2 - yb1) 

    union_area = area_a + area_b - inter_area 
    return inter_area / union_area if union_area > 0 else 0 

model = YOLO("yolo26n.pt") 

ground_truth_car = [78.89, 64.24, 448.68, 336.14] 

results = model.predict(source="Screenshot 2026-09-08 at 17.46.27.png") 
predicted_boxes = results[0].boxes.xyxy.tolist() 

if predicted_boxes: 
    predicted_car = predicted_boxes[0] 
    iou=calculate_iou(ground_truth_car, predicted_car) 
    print(f"Car frame IoU: {iou:.2f}") 
else: 
    print("Car frame - model found nothing (unexpected)") 

results2 = model.predict(source="Screenshot 2026-09-08 at 17.46.37.png") 
predicted_boxes2 = results2[0].boxes.xyxy.tolist() 

if predicted_boxes2: 
    print("Sign frame - false positive: model detected an object where ground truth has none") 
else: 
    print("Sign frame - correct: no detection, matches ground truth") 
