
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import ImageFont

id_to_label = {
    0: "102",
    1: "103a",
    2: "103b",
    3: "103c",
    4: "104",
    5: "106a",
    6: "106b",
    7: "112",
    8: "117",
    9: "123a",
    10: "123b",
    11: "124a",
    12: "124c",
    13: "127.40",
    14: "127.50",
    15: "127.60",
    16: "127.70",
    17: "127.80",
    18: "128",
    19: "130",
    20: "131a",
    21: "131b",
    22: "131c",
    23: "136",
    24: "137",
    25: "201",
    26: "203",
    27: "205",
    28: "207",
    29: "208",
    30: "221",
    31: "224",
    32: "225",
    33: "227",
    34: "239a",
    35: "239b",
    36: "245",
    37: "301",
    38: "302",
    39: "303",
    40: "401",
    41: "402",
    42: "423",
    43: "425",
    44: "434",
    45: "443",
    46: "other1",
    47: "other12",
    48: "other13",
}


def get_faster_rcnn_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Thiết lập mô hình
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_faster_rcnn_model(num_classes=50)
model.load_state_dict(torch.load("path-to-your-model", map_location=device))
model.to(device)


# Hàm dự đoán và lưu hình ảnh với bounding boxes và nhãn
def predict_and_save(image_path, model, device, transform, output_path, id_to_label, threshold=0.5):
    original_image = Image.open(image_path)
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    high_confidence_predictions = pred_scores >= threshold
    final_boxes = pred_boxes[high_confidence_predictions]
    final_labels = pred_labels[high_confidence_predictions]
    final_scores = pred_scores[high_confidence_predictions]

    draw = ImageDraw.Draw(original_image)

    for box, label, score in zip(final_boxes, final_labels, final_scores):
        label_name = id_to_label.get(label, 'Unknown')
        text = f"{label_name}, Score: {score:.2f}"

        # Tính kích thước của văn bản
        text_size = draw.textsize(text)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        draw.rectangle([(box[0], box[1] - text_size[1]), (box[0] + text_size[0], box[1])], fill="white")
        draw.text((box[0], box[1] - text_size[1]), text, fill="black")

    original_image.save(output_path)


# Các thông số và đối tượng cần thiết
transform = transforms.Compose([transforms.ToTensor()])
image_path = "path-to-your-images"  # Thay thế bằng đường dẫn của hình ảnh muốn dự đoán
output_path = "save-predict-images"

# Gọi hàm dự đoán và lưu hình ảnh
predict_and_save(image_path, model, device, transform, output_path, id_to_label, threshold=0.5)

