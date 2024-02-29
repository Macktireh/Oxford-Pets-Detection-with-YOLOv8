from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
model = YOLO("models/best.pt")  # pretrained YOLOv8n model

img1 = "data/test/images/american_pit_bull_terrier_134_jpg.rf.a54d3d1580baca55216d2e195d12c515.jpg"
img2 = "data/test/images/Abyssinian_15_jpg.rf.0e12ac0df99238e4f77a9eb02877b769.jpg"

# Run batched inference on a list of images
# results = model(img)  # return a list of Results objects
results = model.predict((img1, img2))

print()
print(results[0].boxes.data)
print()


# # Process results list
for result in results:
    x1, y1, x2, y2, score, label = result.boxes.data[0]
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    result.save(filename='dog.jpg' if int(label) == 1 else 'cat.jpg')  # save to disk
