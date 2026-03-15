from pathlib import Path
from typing import Any, List, Tuple, Union, Dict
from PIL import Image, ImageFont, ImageDraw
from enum import Enum
from dataclasses import dataclass, field
from PaddleOCR.ppocr.utils.logging import get_logger
import json
import itertools
import random
import cv2
import shutil

TTF_PATH = Path("configs/genshin.ttf")
DATASET_PATH = Path("dataset/")
DATASET_CROP_FOLDER_NAME = "crop"
DATASET_IMAGES_FOLDER_NAME = "images"
DATASET_IMAGES_PATH = DATASET_PATH.joinpath(DATASET_IMAGES_FOLDER_NAME)
DATASET_CROP_PATH = DATASET_PATH.joinpath(DATASET_CROP_FOLDER_NAME)

BACKGROUND_PATH = Path("configs/background/")

TRAIN_COUNT = 4  # 训练集数量
VAL_COUNT = 1  # 验证集数量
SEED = 1  # 随机种子
CROP_GAP = 2  # 裁剪边缘距离
PADDING_MIN = 4  # 最小边缘距离
PADDING_MAX = 40  # 最大边缘距离
COLOR_OFFSET = 5  # 颜色偏移量

logger = get_logger()


def read_file(file: Path) -> str:
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def read_json(file: Path) -> Any:
    return json.loads(read_file(file))


def write_file(file: Path, content: str) -> None:
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)


def points_to_bbox(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    return tuple(
        [
            min(p[0] for p in points),
            min(p[1] for p in points),
            max(p[0] for p in points),
            max(p[1] for p in points),
        ]
    )


def image_crop(image: Image.Image, points: List[Tuple[int, int]]) -> Image.Image:
    return image.crop(points_to_bbox(points))


seed_counter = itertools.count(start=SEED, step=1)


def random_randint(a: int, b: int) -> int:
    random.seed(next(seed_counter))
    return random.randint(a, b)


def random_uniform(a: float, b: float) -> float:
    random.seed(next(seed_counter))
    return random.uniform(a, b)


id_counter = itertools.count(start=1, step=1)


def id() -> int:
    return next(id_counter)


def flatten(*args: List[Any]) -> List[Any]:
    result: List[Any] = []
    for item in args:
        if isinstance(item, list):
            result.extend(flatten(*item))
        else:
            result.append(item)
    return result


def random_padding(
    min: int = PADDING_MIN, max: int = PADDING_MAX
) -> Tuple[int, int, int, int]:
    return tuple(random_randint(min, max) for _ in range(4))


def random_color(
    base: Tuple[int, int, int], offset: int = COLOR_OFFSET
) -> Tuple[int, int, int]:
    return tuple(
        max(0, min(255, base[i] + random_randint(-offset, offset))) for i in range(3)
    )


class Element(Enum):
    NONE = "无"
    PYRO = "火"
    HYDRO = "水"
    ELECTRO = "雷"
    GEO = "岩"
    ANEMO = "风"
    CRYO = "冰"
    DENDRO = "草"


class ArtifactTier(Enum):
    GOLD = "金"
    PURPLE = "紫"
    BLUE = "蓝"
    GREEN = "绿"
    WHITE = "白"

    @classmethod
    def from_rarity_list(cls, rarity_list: List[int]) -> List["ArtifactTier"]:
        mapping = {
            5: cls.GOLD,
            4: cls.PURPLE,
            3: cls.BLUE,
            2: cls.GREEN,
            1: cls.WHITE,
        }
        return [mapping.get(rarity, cls.WHITE) for rarity in rarity_list]


class Part(Enum):
    FLOWER = "生之花"
    PLUME = "死之羽"
    SANDS = "时之沙"
    GOBLET = "空之杯"
    CIRCLET = "理之冠"


class ArtifactAttribute(Enum):
    HP = ["生命值", int, [717, 4780]]
    HP_PERCENT = ["生命值", float, [7.0, 46.6]]
    ATK = ["攻击力", int, [47, 311]]
    ATK_PERCENT = ["攻击力", float, [7.0, 46.6]]
    DEF_PERCENT = ["防御力", float, [8.7, 58.3]]
    ENERGY_RECHARGE = ["元素充能效率", float, [7.8, 51.8]]
    ELEMENTAL_MASTERY = ["元素精通", int, [28, 187]]
    CRIT_RATE = ["暴击率", float, [4.7, 31.1]]
    CRIT_DMG = ["暴击伤害", float, [9.4, 62.2]]
    HEALING_BONUS = ["治疗加成", float, [5.4, 35.9]]
    PHYSICAL_DMG_BONUS = ["物理伤害加成", float, [8.7, 58.3]]
    PYRO_BONUS = ["火元素伤害加成", float, [7.0, 46.6]]
    HYDRO_BONUS = ["水元素伤害加成", float, [7.0, 46.6]]
    ELECTRO_BONUS = ["雷元素伤害加成", float, [7.0, 46.6]]
    GEO_BONUS = ["岩元素伤害加成", float, [7.0, 46.6]]
    ANEMO_BONUS = ["风元素伤害加成", float, [7.0, 46.6]]
    CRYO_BONUS = ["冰元素伤害加成", float, [7.0, 46.6]]
    DENDRO_BONUS = ["草元素伤害加成", float, [7.0, 46.6]]

    @property
    def artifact_name(self) -> Union[int, float]:
        return self.value[0]

    @property
    def range(self) -> List[Union[int, float]]:
        return self.value[2]

    @property
    def type(self) -> type:
        return self.value[1]

    def random_value(self) -> str:
        if issubclass(self.type, int):
            return str(random_randint(self.range[0], self.range[1]))
        elif issubclass(self.type, float):
            return f"{round(random_uniform(self.range[0], self.range[1]), 1)}%"
        else:
            raise ValueError("Invalid attribute type")

    def random_name_value(self) -> str:
        value = ""
        if issubclass(self.type, int):
            value = str(random_randint(self.range[0], self.range[1]))
        elif issubclass(self.type, float):
            value = f"{round(random_uniform(self.range[0], self.range[1]), 1)}%"
        else:
            raise ValueError("Invalid attribute type")
        return f"{self.artifact_name}+{value}"

    def random_unactivated_name_value(self) -> str:
        return f"{self.random_name_value()} (待激活)"

    def random_unactivated_name(self) -> str:
        return f"{self.artifact_name} (待激活)"


class GenshinDB:
    def __init__(self):
        self.data_root_path = Path("genshin-db/src/data/ChineseSimplified/")

    def read_files(self, folder: str) -> List[Any]:
        return [
            read_json(file)
            for file in self.data_root_path.joinpath(folder).glob("*.json")
        ]


@dataclass
class ImageData:
    text: str
    image: Image.Image
    image_crop: Image.Image
    points: List[Tuple[int, int]]
    name: str

    def get_det(self) -> str:
        data = [{"transcription": self.text, "points": self.points}]
        return f"{Path(DATASET_IMAGES_FOLDER_NAME, self.name)}\t{json.dumps(data, ensure_ascii=False)}"

    def get_rec(self) -> str:
        return f"{Path(DATASET_CROP_FOLDER_NAME, self.name)}\t{self.text}"

    def save_image(self) -> None:
        self.image.save(DATASET_IMAGES_PATH.joinpath(self.name))
        self.image_crop.save(DATASET_CROP_PATH.joinpath(self.name))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.text}, {self.points})"

    def __post_init__(self) -> None:
        self.name = f"{str(self.name).zfill(7)}.png"


@dataclass
class BuildOption:
    id: int = field(default_factory=id)
    text: str = "Genshin Impact"
    font_size: int = 12
    # (r, g, b)
    color: Tuple[int, int, int] = (0, 0, 0)
    # (left, top, right, bottom)
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        self.color = random_color(self.color)
        self.padding = random_padding()
        logger.info(self)


@dataclass
class PlainBuildOption(BuildOption):
    background: Tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.background = random_color(self.background)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, text={self.text}, font_size={self.font_size}, color={self.color}, padding={self.padding}, background={self.background})"


SkyBackgroundType = Union[Element, ArtifactTier]


@dataclass
class SkyBuildOption(BuildOption):
    background: SkyBackgroundType = Element.NONE

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, text={self.text}, font_size={self.font_size}, color={self.color}, padding={self.padding}, background={self.background})"


@dataclass
class SkyBackground:
    name: str = ""
    width: int = 0
    height: int = 0
    frames: int = 0
    cap: cv2.VideoCapture = None

    def __post_init__(self) -> None:
        self.cap = cv2.VideoCapture(BACKGROUND_PATH.joinpath(f"{self.name}.mp4"))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, width={self.width}, height={self.height}, frames={self.frames})"


backgrounds: Dict[str, SkyBackground] = {}

GenerateImageOption = Union[PlainBuildOption, SkyBuildOption]


def generate_image(option: GenerateImageOption) -> ImageData:
    font = ImageFont.truetype(TTF_PATH, option.font_size)
    bbox = font.getbbox(option.text)
    width = bbox[2] - bbox[0] + option.padding[0] + option.padding[2]
    height = bbox[3] - bbox[1] + option.padding[1] + option.padding[3]

    if isinstance(option, SkyBuildOption):
        sky_background = backgrounds.get(option.background.name.lower())
        if not sky_background:
            raise ValueError(f"Invalid sky background: {option.background}")
        frame_index = random_randint(0, sky_background.frames - 1)
        sky_background.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        _, frame = sky_background.cap.read()
        sky_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将图片随机画到图片上
        start_x = random_randint(0, sky_background.width - width)
        start_y = random_randint(0, sky_background.height - height)

        image = Image.fromarray(sky_frame)
        image = image_crop(
            image, [(start_x, start_y), (start_x + width, start_y + height)]
        )
    else:
        image = Image.new("RGB", (width, height), option.background)

    draw = ImageDraw.Draw(image)

    draw.text(
        (option.padding[0] - bbox[0], option.padding[1] - bbox[1]),
        text=option.text,
        font=font,
        fill=option.color,
    )
    start_x = option.padding[0] - CROP_GAP
    start_y = option.padding[1] - CROP_GAP
    end_x = width - option.padding[2] + CROP_GAP
    end_y = height - option.padding[3] + CROP_GAP
    points = [(start_x, start_y), (end_x, start_y), (end_x, end_y), (start_x, end_y)]
    return ImageData(
        name=option.id,
        text=option.text,
        image=image,
        image_crop=image_crop(image, points),
        points=points,
    )


class ImageGenerator:
    def __init__(self, text: str):
        self.text = text

    def generate(self) -> List[GenerateImageOption]:
        pass

    def generate_train(self) -> List[GenerateImageOption]:
        return flatten([self.generate() for _ in range(TRAIN_COUNT)])

    def generate_val(self) -> List[GenerateImageOption]:
        return flatten([self.generate() for _ in range(VAL_COUNT)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.text})"


class CharacterImageGenerator(ImageGenerator):
    def __init__(self, text: str, element: Element):
        super().__init__(text)
        self.element = element

    def generate_sky(self) -> GenerateImageOption:
        return SkyBuildOption(
            text=self.text,
            font_size=22,
            color=(211, 188, 142),
            background=self.element,
        )

    def generate_equipped(self) -> GenerateImageOption:
        return PlainBuildOption(
            text=f"{self.text}已装备",
            font_size=42,
            color=(73, 83, 102),
            background=(255, 231, 187),
        )

    def generate(self) -> List[GenerateImageOption]:
        # 排除主角, 奇偶
        if self.element == Element.NONE:
            return []
        return [self.generate_sky(), self.generate_equipped()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.text}, {self.element})"


class ArtifactImageGenerator(ImageGenerator):
    def __init__(self, text: str, tiers: List[ArtifactTier], part_names: List[str]):
        super().__init__(text)
        self.tiers = tiers
        self.part_names = part_names
        self.tier_background: Dict[ArtifactTier, Tuple[int, int, int]] = {
            ArtifactTier.WHITE: (112, 118, 138),
            ArtifactTier.GREEN: (42, 142, 113),
            ArtifactTier.BLUE: (80, 126, 201),
            ArtifactTier.PURPLE: (159, 85, 222),
            ArtifactTier.GOLD: (186, 105, 52),
        }

    def generate_part(self) -> List[GenerateImageOption]:
        result: List[GenerateImageOption] = []

        for part_name in self.part_names:
            for tier in self.tiers:
                background = self.tier_background.get(tier)
                result.append(
                    PlainBuildOption(
                        text=part_name,
                        font_size=30,
                        color=(255, 255, 255),
                        background=background,
                    )
                )

        return result

    def generate_set(self) -> GenerateImageOption:
        return PlainBuildOption(
            text=self.text,
            font_size=22,
            color=(92, 178, 86),
            background=(236, 229, 216),
        )

    def generate_filter(self) -> List[GenerateImageOption]:
        return [
            # 可选
            PlainBuildOption(
                text=self.text,
                font_size=22,
                color=(236, 229, 216),
                background=(44, 56, 66),
            ),
            # 不可选
            PlainBuildOption(
                text=self.text,
                font_size=22,
                color=(127, 129, 133),
                background=(44, 56, 66),
            ),
        ]

    def generate(self) -> List[GenerateImageOption]:
        return flatten(
            self.generate_part(), self.generate_set(), self.generate_filter()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.text}, {self.tiers}, {self.part_names})"
        )


class ArtifactTierImageGenerator(ImageGenerator):
    def __init__(self, text: str, tier: ArtifactTier):
        super().__init__(text)
        self.tier = tier
        self.tier_background: Dict[ArtifactTier, Tuple[int, int, int]] = {
            ArtifactTier.WHITE: (79, 87, 98),
            ArtifactTier.GREEN: (74, 90, 94),
            ArtifactTier.BLUE: (81, 85, 119),
            ArtifactTier.PURPLE: (94, 88, 134),
            ArtifactTier.GOLD: (110, 86, 83),
        }
        self.background = self.tier_background.get(self.tier)
        self.sky_background = backgrounds.get(self.tier.name.lower())

    def generate_part(self) -> List[GenerateImageOption]:
        return [
            PlainBuildOption(
                text=part.value,
                font_size=30,
                color=(255, 255, 255),
                background=self.background,
            )
            for part in Part
        ]

    def generate_main_attribute(self) -> GenerateImageOption:
        result: List[GenerateImageOption] = []
        for attribute in ArtifactAttribute:
            # 词条
            result.append(
                PlainBuildOption(
                    text=attribute.artifact_name,
                    font_size=20,
                    color=(188, 178, 206),
                    background=self.background,
                )
            )
            # 词条值
            result.append(
                PlainBuildOption(
                    text=attribute.random_value(),
                    font_size=38,
                    color=(255, 255, 255),
                    background=self.background,
                )
            )

        return result

    def generate_sky_unactivated_attribute(self) -> GenerateImageOption:
        result: List[GenerateImageOption] = []
        for attribute in ArtifactAttribute:
            # 待激活词条
            result.append(
                SkyBuildOption(
                    text=attribute.random_unactivated_name(),
                    font_size=27,
                    color=(181, 164, 149),
                    background=self.sky_background,
                )
            )
            # 强化词条值
            result.append(
                SkyBuildOption(
                    text=attribute.random_value(),
                    font_size=27,
                    color=(181, 164, 149),
                    background=self.sky_background,
                )
            )

        return result

    def generate_sky_attribute(self) -> GenerateImageOption:
        result: List[GenerateImageOption] = []
        for attribute in ArtifactAttribute:
            # 强化词条值
            result.append(
                SkyBuildOption(
                    text=attribute.artifact_name,
                    font_size=27,
                    color=(255, 255, 255),
                    background=self.sky_background,
                )
            )
            # 强化词条值
            result.append(
                SkyBuildOption(
                    text=attribute.random_value(),
                    font_size=27,
                    color=(255, 255, 255),
                    background=self.sky_background,
                )
            )
        return result

    def generate(self) -> List[GenerateImageOption]:
        return flatten(
            self.generate_part(),
            self.generate_main_attribute(),
            self.generate_sky_attribute(),
            self.generate_sky_unactivated_attribute(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.text}, {self.tier})"


class ArtifactCommonImageGenerator(ImageGenerator):
    def __init__(self):
        super().__init__("")

    def generate_level(self) -> List[GenerateImageOption]:
        return [
            PlainBuildOption(
                text=f"+{i}",
                font_size=22,
                color=(255, 255, 255),
                background=(57, 68, 79),
            )
            for i in range(1, 21)
        ]

    def generate_attribute(self) -> List[GenerateImageOption]:
        return [
            PlainBuildOption(
                text=attribute.random_name_value(),
                font_size=38,
                color=(73, 83, 102),
                background=(236, 229, 216),
            )
            for attribute in ArtifactAttribute
        ]

    def generate_unactivated_attribute(self) -> List[GenerateImageOption]:
        return [
            PlainBuildOption(
                text=attribute.random_unactivated_name_value(),
                font_size=38,
                color=(154, 156, 159),
                background=(236, 229, 216),
            )
            for attribute in ArtifactAttribute
        ]

    def generate(self) -> List[GenerateImageOption]:
        return flatten(
            self.generate_level(),
            self.generate_attribute(),
            self.generate_unactivated_attribute(),
        )


class DataParser:
    def parser(self) -> List[ImageGenerator]:
        raise NotImplementedError()


class CharacterParser(DataParser):
    def __init__(self, db: GenshinDB):
        self.db = db

    def parser(self) -> List[ImageGenerator]:
        return [
            CharacterImageGenerator(
                character["name"], Element(character["elementText"])
            )
            for character in self.db.read_files("characters")
        ]


class ArtifactParser(DataParser):
    def __init__(self, db: GenshinDB):
        self.db = db

    def parser(self) -> List[ImageGenerator]:
        result = []
        for artifact in self.db.read_files("artifacts"):
            part_names = []
            for part in Part:
                part_json = artifact.get(part.name.lower())
                if part_json is not None:
                    part_names.append(part_json["name"])

            result.append(
                ArtifactImageGenerator(
                    artifact["name"],
                    ArtifactTier.from_rarity_list(artifact["rarityList"]),
                    part_names,
                )
            )
        return result


class ArtifactTierParser(DataParser):
    def parser(self) -> List[ImageGenerator]:
        return [ArtifactTierImageGenerator(tier.value, tier) for tier in ArtifactTier]


class ArtifactCommonParser(DataParser):
    def parser(self) -> List[ImageGenerator]:
        return [ArtifactCommonImageGenerator()]


def init_dataset() -> None:
    global backgrounds

    if DATASET_PATH.exists():
        logger.info("Dataset already exists, removing...")
        shutil.rmtree(DATASET_PATH)
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    DATASET_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    DATASET_CROP_PATH.mkdir(parents=True, exist_ok=True)

    logger.info("Loading backgrounds videos...")
    for file in BACKGROUND_PATH.glob("*.mp4"):
        backgrounds[file.stem] = SkyBackground(file.stem)


if __name__ == "__main__":
    init_dataset()

    db = GenshinDB()

    parsers: List[DataParser] = [
        CharacterParser(db),
        ArtifactParser(db),
        ArtifactTierParser(),
        ArtifactCommonParser(),
    ]

    det_train: List[str] = []
    det_val: List[str] = []

    rec_train: List[str] = []
    rec_val: List[str] = []

    logger.info("Building dataset...")

    for parser in parsers:
        for builder in parser.parser():
            for option in builder.generate_train():
                image_data = generate_image(option)
                rec_train.append(image_data.get_rec())
                det_train.append(image_data.get_det())
                image_data.save_image()

            for option in builder.generate_val():
                image_data = generate_image(option)
                rec_val.append(image_data.get_rec())
                det_val.append(image_data.get_det())
                image_data.save_image()

    det_train_path = DATASET_PATH.joinpath("det_train.txt")
    det_val_path = DATASET_PATH.joinpath("det_val.txt")
    write_file(det_train_path, "\n".join(det_train))
    write_file(det_val_path, "\n".join(det_val))
    logger.info(f"Det train ({len(det_train)}): {det_train_path}")
    logger.info(f"Det val ({len(det_val)}): {det_val_path}")

    rec_train_path = DATASET_PATH.joinpath("rec_train.txt")
    rec_val_path = DATASET_PATH.joinpath("rec_val.txt")
    write_file(rec_train_path, "\n".join(rec_train))
    write_file(rec_val_path, "\n".join(rec_val))
    logger.info(f"Rec train ({len(rec_train)}): {rec_train_path}")
    logger.info(f"Rec val ({len(rec_val)}): {rec_val_path}")
