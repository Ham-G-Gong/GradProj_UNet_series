from argparse import ArgumentParser, Namespace
from typing import List, Tuple, BinaryIO
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
# from utils.fanet import FANet
from utils.UNet import UNet
import cv2
from imageio import imread
from cv2.mat_wrapper import Mat
import pkg_resources
import sys
import os
import io
import zipfile
import pkg_resources


SIZE: List[int] = [512, 512]


def getArgs() -> Namespace:
    # NOTE: These variables can be changed
    programName: str = "LPCVC 2023 Sample Solution"
    authors: List[str] = ["Nicholas M. Synovic", "Ping Hu"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = f"This {programName} does create a single segmentation map of arieal scenes of disaster environments captured by unmanned arieal vehicles (UAVs)"
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    # NOTE: Do not change these flags
    parser: ArgumentParser = ArgumentParser(prog, usage, description, epilog)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )

    return parser.parse_args()


def loadImageToTensor(imagePath: str) -> torch.Tensor:
    MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    STANDARD_DEVIATION: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    image: Array = imread(uri=imagePath)
    resizedImage: Mat = cv2.resize(image, tuple(SIZE), interpolation=cv2.INTER_AREA)
    imageTensor: Tensor = transforms.ToTensor()(resizedImage)
    imageTensor: Tensor = transforms.Normalize(mean=MEAN, std=STANDARD_DEVIATION)(
        imageTensor
    )
    imageTensor: Tensor = imageTensor.unsqueeze(0)

    return imageTensor

def color_mapping(image):
    color_map = np.array([
        [0, 0, 0],       # 배경 (검정색)
        [128, 0, 0],     # 클래스 1 (어두운 빨강)
        [0, 128, 0],     # 클래스 2 (어두운 초록)
        [128, 128, 0],   # 클래스 3 (어두운 옥색)
        [0, 0, 128],     # 클래스 4 (어두운 파랑)
        [128, 0, 128],   # 클래스 5 (어두운 보라)
        [0, 128, 128],   # 클래스 6 (어두운 시안)
        [128, 128, 128], # 클래스 7 (회색)
        [64, 0, 0],      # 클래스 8 (진한 빨강)
        [192, 0, 0],     # 클래스 9 (밝은 빨강)
        [64, 128, 0],    # 클래스 10 (노란)
        [192, 128, 0],   # 클래스 11 (주황)
        [64, 0, 128],    # 클래스 12 (보라)
        [192, 0, 128],   # 클래스 13 (분홍)
    ])

    # 색으로 변환
    return color_map[image]



def main() -> None:
    args: Namespace = getArgs()

    # NOTE: modelPath should be the path to your model in respect to your solution directory
    modelPath: str = "Unet.pkl"

    image_files: List[str] = os.listdir(args.input)


    with pkg_resources.resource_stream(__name__, modelPath) as model_file:
        model: UNet = UNet()
        device = torch.device("cuda")
        model.to(device)
        model.load_state_dict(
                state_dict=torch.load(f=model_file, map_location=torch.device("cuda")), strict=False
        )
        for image_file in image_files:
            input_image_path: str = os.path.join(args.input, image_file)
            output_image_path: str = os.path.join(args.output, image_file)
            imageTensor: torch.Tensor = loadImageToTensor(imagePath=input_image_path)
            imageTensor = imageTensor.to(device)
            outTensor: torch.Tensor = model(imageTensor)
            outTensor: torch.Tensor = F.interpolate(
                outTensor, SIZE, mode="bilinear", align_corners=True
            )

            outArray: np.ndarray = outTensor.cpu().data.max(1)[1].numpy()
            outArray: np.ndarray = outArray.astype(np.uint8)


            outImage: np.ndarray = np.squeeze(outArray, axis=0)
            # print(outImage.shape)

            ########################################
            ### 이미지 파일로 만들고 싶은 경우 주석 해제 ###
            ### 아래 Image.from ~~ 두 코드 주석할것. ###
            ########################################
            outImage = color_mapping(outImage)
            cv2.imwrite(output_image_path, outImage)

            # outImage = Image.fromarray(outImage, mode='L')
            # outImage.save(output_image_path)

        del model
        del imageTensor
        del outTensor
        torch.cuda.empty_cache()
        model_file.close()
