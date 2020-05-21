import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import darknet

output_file = ''

""" Definicion """
""" ********** """


def convert(bounding_box):
    x_min = int(round(bounding_box[0] - (bounding_box[2] / 2)))
    x_max = int(round(bounding_box[0] + (bounding_box[2] / 2)))
    y_min = int(round(bounding_box[1] - (bounding_box[3] / 2)))
    y_max = int(round(bounding_box[1] + (bounding_box[3] / 2)))
    return (x_min, y_min, x_max, y_max)


def rescale(frame_size, draknet_size, bounding_box):
    factor_x = frame_size[0] / draknet_size[0]
    factor_y = frame_size[1] / draknet_size[1]
    x_min = int(bounding_box[0] * factor_x)
    y_min = int(bounding_box[1] * factor_y)
    x_max = int(bounding_box[2] * factor_x)
    y_max = int(bounding_box[3] * factor_y)
    return (x_min, y_min, x_max, y_max)


# Se define la transformacion.
data_transforms = transforms.Compose(
    [transforms.Resize((160, 160)), transforms.ToTensor()])

# Se define el dispositivo.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('result/cough_result.model', map_location=device)
model = model.to(device)
model.eval()

# Carga del video.
capture = cv2.VideoCapture('')
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video frames={length} width={width} height={height}')

# Iniciando modelo YOLO.
darknet_meta = darknet.load_meta('data/coco.data'.encode('ascii'))
darknet_model = darknet.load_net_custom(
    'cfg/yolov4.cfg'.encode('ascii'), 'yolov4.weights'.encode('ascii'), 0, 1)

darknet_width = darknet.network_width(darknet_model)
darknet_height = darknet.network_height(darknet_model)
darknet_image = darknet.make_image(darknet_width, darknet_height, 3)


# Definiendo el formato de video de salida.
codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
framerate = 30
resolution = (width, height)
output_video = cv2.VideoWriter(output_file, codec, framerate, resolution)

if capture.isOpened():
    ret, frame = capture.read()

    while ret:
        ret, frame = capture.read()
        frame_resized = cv2.resize(
            frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # Evaluacion de resultados.
        results = darknet.detect_image(
            darknet_model, darknet_meta, darknet_image, thresh=0.25)
        for result in results:
            class_id, confidence, bounding_box = result
            class_id = class_id.decode('utf-8')
            if class_id == 'person':
                print(f'{class_id}: {confidence}')

                # Transformacion de coordenadas.
                bounding_box = convert(bounding_box)
                bounding_box = rescale((width, height), (darknet_width,
                                                darknet_height), bounding_box)
                """
                person = image[box[1]:box[3], box[0]:box[2]]
                print(person)

                # Clasificador de accion.
                image = data_transforms(image)
                image = image.unsqueeze(0)
                if torch.cuda.is_available():
                    image = image.cuda()
                output = model(image)
                prob, pred = torch.max(output, 1)
                """
    capture.release()
    output_video.release()
