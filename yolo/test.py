import os
import torch
import cv2
import argparse
from ultralytics import YOLO
import ultralytics

# Add BOTH necessary classes to the list of safe globals
torch.serialization.add_safe_globals([
    ultralytics.nn.tasks.PoseModel,
    torch.nn.modules.container.Sequential,
    ultralytics.nn.modules.conv.Conv,
    torch.nn.modules.conv.Conv2d,
    torch.nn.modules.batchnorm.BatchNorm2d,
    torch.nn.modules.activation.SiLU,
    ultralytics.nn.modules.block.C2f,
    ultralytics.nn.modules.block.Bottleneck,
    ultralytics.nn.modules.block.SPPF,
    torch.nn.modules.pooling.MaxPool2d,
    torch.nn.modules.upsampling.Upsample,
    ultralytics.nn.modules.conv.Concat,
    ultralytics.nn.modules.head.Pose,
    ultralytics.nn.modules.block.DFL,
    getattr,
    ultralytics.nn.modules.head.Detect,
    ultralytics.utils.IterableSimpleNamespace,
    ultralytics.utils.loss.v8PoseLoss,
    torch.nn.modules.loss.BCEWithLogitsLoss,
    ultralytics.utils.tal.TaskAlignedAssigner,
    ultralytics.utils.loss.BboxLoss,
    ultralytics.utils.loss.KeypointLoss,
    torch.nn.modules.container.ModuleList  # Add this new class
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', action='store_true')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--model_extension', type=str, default='engine', help='pt or engine (engine is much faster)')
    parser.add_argument('--model_size', type=str, default='m')
    parser.add_argument('--resolution', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.6)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    data_root = args.data_root # this should be one file if single_frame is True
    model_extension = args.model_extension
    resolution = args.resolution
    conf = args.conf
    device = args.device

    model_name = f'{args.model_size}_{resolution}'
    model_file = f'runs/pose/{model_name}/weights/best.{model_extension}'

    model = YOLO(model_file)
    if model_extension == 'pt' and device != 'cpu' and torch.cuda.is_available():
        model = model.to(device)


if args.sequence:
    # Normalize filenames by stripping leading zeros before converting to int
    frames = [
        int(file.split('.')[0].lstrip('0') or '0')  # handles '0000.png' safely
        for file in os.listdir(data_root)
        if file.endswith('.png')
    ]
    
    start_frame, end_frame = min(frames), max(frames)
    
    for frame_num in range(start_frame, end_frame + 1):
        # Match files by zero-stripped integer part
        matching_files = [
            file for file in os.listdir(data_root)
            if file.endswith('.png') and int(file.split('.')[0].lstrip('0') or '0') == frame_num
        ]
        
        if not matching_files:
            print(f"Frame {frame_num} not found.")
            continue
        
        frame_file = matching_files[0]
        frame_path = os.path.join(data_root, frame_file)

        results = model.track(frame_path, conf=conf, imgsz=resolution, device=device, persist=True, tracker='bytetrack.yaml')
        img = results[0].plot()
        cv2.imshow('result', img)
        cv2.waitKey(500)

    else:
        results = model(data_root, conf=conf, imgsz=resolution, device=device)
        for result in results:
            img = result.plot()
            cv2.imshow('result', img)
            cv2.waitKey(500)
