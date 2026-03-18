from dataclasses import dataclass, field
from typing import (
    List,
    Union,
    Any,
    Optional,
    NamedTuple,
    Tuple,
    TypeVar,
    Generic,
    Type,
    Dict,
    Set,
)
from pathlib import Path
from enum import Enum, IntEnum
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from PaddleOCR.ppocr.utils.logging import get_logger
from dataclasses_json import dataclass_json, LetterCase
from PIL import Image, ImageFont, ImageDraw
from functools import reduce
from copy import deepcopy
import shutil
import cv2
import json
import random
import itertools
import math

logger = get_logger("dataset")
DATASET = Path("dataset")
BACKGROUND = Path("configs/background/")
GENSHIN_DB = Path("genshin-db")
GENSHIN_TTF = Path("configs/genshin.ttf")
CROP_IMG_NAME = "crop_img"
IMAGES_NAME = "images"
CROP_IMG = DATASET.joinpath(CROP_IMG_NAME)
IMAGES = DATASET.joinpath(IMAGES_NAME)


class ArgsNamespace(Namespace):
    seed: int
    rec_padding: int
    det_padding: int
    color_offset: int
    font_size_offset: int
    train_count: int
    val_count: int


class ArgsParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgsParser, self, *args, **kwargs).__init__(
            formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(
                prog, max_help_position=40, width=120
            )
        )
        self.add_argument("--seed", type=int, default=1, help="Random seed")
        self.add_argument(
            "--rec-padding",
            type=int,
            default=3,
            help="Padding for recognition",
        )
        self.add_argument(
            "--det-padding",
            type=int,
            default=16,
            help="Padding for detection",
        )
        self.add_argument(
            "--color-offset",
            type=int,
            default=5,
            help="Color offset for random color",
        )
        self.add_argument(
            "--font-size-offset",
            type=int,
            default=2,
            help="Font size offset for random font size",
        )
        self.add_argument(
            "--train-count",
            type=int,
            default=8,
            help="Number of training images",
        )
        self.add_argument(
            "--val-count",
            type=int,
            default=2,
            help="Number of validation images",
        )

    def parse_args(self, *args, **kwargs) -> ArgsNamespace:
        return super(ArgsParser, self).parse_args(*args, **kwargs)


args = ArgsParser().parse_args()
logger.info(f"Args: {args}")


class IdGenerator:
    def __init__(self, start: int = 1):
        self.id_counter = itertools.count(start=start, step=1)

    def __call__(self) -> int:
        return next(self.id_counter)


def read_file(file: Path) -> str:
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def append_file(file: Path, content: str) -> None:
    with open(file, "a", encoding="utf-8") as f:
        f.write(content)


seed_counter = IdGenerator(args.seed)


def random_randint(a: int, b: int) -> int:
    random.seed(seed_counter())
    return random.randint(a, b)


def random_uniform(a: float, b: float) -> float:
    random.seed(seed_counter())
    return random.uniform(a, b)


L = TypeVar("L")


def random_shuffle(arr: List[L]) -> List[L]:
    random.seed(seed_counter())
    random.shuffle(arr)
    return arr


def random_color(
    base: Tuple[int, int, int], offset: int = args.color_offset
) -> Tuple[int, int, int]:
    return tuple(
        max(0, min(255, base[i] + random_randint(-offset, offset))) for i in range(3)
    )


def random_font_size(base: int, offset: int = 10) -> int:
    return max(1, base + random_randint(-offset, offset))


def flatten(*args: List[Any]) -> List[Any]:
    result: List[Any] = []
    for item in args:
        if isinstance(item, list):
            result.extend(flatten(*item))
        else:
            result.append(item)
    return result


class Slot(Enum):
    FLOWER = "生之花"
    PLUME = "死之羽"
    SANDS = "时之沙"
    GOBLET = "空之杯"
    CIRCLET = "理之冠"


class ElementText(Enum):
    NONE = "无"
    PYRO = "火"
    HYDRO = "水"
    ELECTRO = "雷"
    GEO = "岩"
    ANEMO = "风"
    CRYO = "冰"
    DENDRO = "草"


class Rarity(IntEnum):
    GOLD = 5
    PURPLE = 4
    BLUE = 3
    GREEN = 2
    WHITE = 1


T = TypeVar("T", bound=Union[int, float])


class AffixInfo(NamedTuple, Generic[T]):
    name: str
    type: Type[T]
    range: Tuple[T, T]


class Affix(Enum):
    HP = AffixInfo("生命值", int, [717, 4780])
    HP_PERCENT = AffixInfo("生命值", float, [7.0, 46.6])
    ATK = AffixInfo("攻击力", int, [47, 311])
    ATK_PERCENT = AffixInfo("攻击力", float, [7.0, 46.6])
    DEF_PERCENT = AffixInfo("防御力", float, [8.7, 58.3])
    ENERGY_RECHARGE = AffixInfo("元素充能效率", float, [7.8, 51.8])
    ELEMENTAL_MASTERY = AffixInfo("元素精通", int, [28, 187])
    CRIT_RATE = AffixInfo("暴击率", float, [4.7, 31.1])
    CRIT_DMG = AffixInfo("暴击伤害", float, [9.4, 62.2])
    HEALING_BONUS = AffixInfo("治疗加成", float, [5.4, 35.9])
    PHYSICAL_DMG_BONUS = AffixInfo("物理伤害加成", float, [8.7, 58.3])
    PYRO_BONUS = AffixInfo("火元素伤害加成", float, [7.0, 46.6])
    HYDRO_BONUS = AffixInfo("水元素伤害加成", float, [7.0, 46.6])
    ELECTRO_BONUS = AffixInfo("雷元素伤害加成", float, [7.0, 46.6])
    GEO_BONUS = AffixInfo("岩元素伤害加成", float, [7.0, 46.6])
    ANEMO_BONUS = AffixInfo("风元素伤害加成", float, [7.0, 46.6])
    CRYO_BONUS = AffixInfo("冰元素伤害加成", float, [7.0, 46.6])
    DENDRO_BONUS = AffixInfo("草元素伤害加成", float, [7.0, 46.6])

    @property
    def name(self) -> str:
        return self.value.name

    @property
    def type(self) -> type:
        return self.value.type

    def random_value(self) -> str:
        if issubclass(self.type, int):
            return str(random_randint(self.value.range[0], self.value.range[1]))
        elif issubclass(self.type, float):
            return (
                f"{round(random_uniform(self.value.range[0], self.value.range[1]), 1)}%"
            )
        else:
            raise ValueError("Invalid attribute type")

    def random_name_value(self) -> str:
        value = ""
        if issubclass(self.type, int):
            value = str(random_randint(self.value.range[0], self.value.range[1]))
        elif issubclass(self.type, float):
            value = (
                f"{round(random_uniform(self.value.range[0], self.value.range[1]), 1)}%"
            )
        else:
            raise ValueError("Invalid attribute type")
        return f"{self.name}+{value}"

    def random_unactivated_name_value(self) -> str:
        return f"{self.random_name_value()} (待激活)"

    def random_unactivated_name(self) -> str:
        return f"{self.name} (待激活)"


Color = Tuple[int, int, int]
StarrySky = Union[ElementText, Rarity]
Background = Union[StarrySky, Color]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CharacterJson:
    name: str
    element_text: ElementText


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ArtifactSlotJson:
    name: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ArtifactJson:
    name: str
    rarity_list: List[Rarity]

    flower: Optional[ArtifactSlotJson] = None
    plume: Optional[ArtifactSlotJson] = None
    sands: Optional[ArtifactSlotJson] = None
    goblet: Optional[ArtifactSlotJson] = None
    circlet: Optional[ArtifactSlotJson] = None


class GenshinDatabase:
    def __init__(self):
        self.chinese_simplified = GENSHIN_DB.joinpath("src/data/ChineseSimplified/")

    def read_files(self, folder: str) -> List[str]:
        return [
            read_file(file)
            for file in self.chinese_simplified.joinpath(folder).glob("*.json")
        ]

    def read_characters(self) -> List[CharacterJson]:
        return [CharacterJson.from_json(file) for file in self.read_files("characters")]

    def read_artifacts(self) -> List[ArtifactJson]:
        return [ArtifactJson.from_json(file) for file in self.read_files("artifacts")]


database = GenshinDatabase()


class StarrySkyVideo:
    def __init__(self, name: str):
        self.name = name
        self.cap = cv2.VideoCapture(BACKGROUND.joinpath(f"{name}.mp4"))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class StarrySkyBackground:
    def __init__(self):
        self.starry_skies: Dict[StarrySky, StarrySkyVideo] = {}
        for file in BACKGROUND.glob("*.mp4"):
            name = file.stem
            video = StarrySkyVideo(name)
            if name.upper() in ElementText.__members__:
                self.starry_skies[ElementText[name.upper()]] = video
            elif name.upper() in Rarity.__members__:
                self.starry_skies[Rarity[name.upper()]] = video

    def get_random_frame(self, starry_sky: StarrySky) -> Image.Image:
        video = self.starry_skies[starry_sky]
        frame_index = random_randint(0, video.frames - 1)
        video.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.cap.read()
        if not ret:
            raise ValueError("Failed to read frame")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @property
    def min_width(self) -> int:
        return min(video.width for video in self.starry_skies.values())

    @property
    def min_height(self) -> int:
        return min(video.height for video in self.starry_skies.values())


starry_sky_background = StarrySkyBackground()


@dataclass
class Blueprint:
    text: str
    color: Color
    font_size: int
    background: Background

    def __post_init__(self):
        self.color = random_color(self.color)
        self.font_size = random_font_size(self.font_size, args.font_size_offset)
        logger.debug(self)


# left, top, width, height
Rect = Tuple[float, float, float, float]


@dataclass
class BlueprintBrick:
    rect: Rect
    blueprint: Blueprint

    @property
    def points(self) -> List[Tuple[int, int]]:
        left, upper, right, lower = self.box
        return [[left, upper], [right, upper], [right, lower], [left, lower]]

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (
            max(0, math.floor(self.rect[0] - args.rec_padding)),
            max(0, math.floor(self.rect[1] - args.rec_padding)),
            min(
                starry_sky_background.min_width,
                math.ceil(self.rect[0] + self.rect[2] + args.rec_padding),
            ),
            min(
                starry_sky_background.min_height,
                math.ceil(self.rect[1] + self.rect[3] + args.rec_padding),
            ),
        )

    def zoom(self, zoom: float) -> "BlueprintBrick":
        new_brick = deepcopy(self)
        new_brick.rect = tuple(item * zoom for item in self.rect)
        return new_brick


@dataclass
class Rec:
    brick: BlueprintBrick
    image: Image.Image
    zoom: float = 1.0
    prefix: str = ""
    id: int = field(default_factory=IdGenerator())

    @property
    def image_hash(self) -> str:
        return str(hash(self.image.tobytes()))

    @property
    def filename(self) -> str:
        return f"{self.prefix}-{str(self.id).zfill(5)}-{self.zoom:.2f}.png"

    @property
    def label_line(self) -> str:
        return f"{Path(CROP_IMG_NAME).joinpath(self.filename).as_posix()}\t{self.brick.blueprint.text}"

    def save(self) -> None:
        self.image.save(f"{CROP_IMG.joinpath(self.filename)}")


@dataclass
class Det:
    bricks: List[BlueprintBrick]
    image: Image.Image
    id: int = field(default_factory=IdGenerator())
    prefix: str = ""
    _zoom: float = field(init=False, default=1.0)
    _recs: List[Rec] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._recs = [
            Rec(brick, self.zoom_image.crop(brick.box), self.zoom)
            for brick in self.zoom_bricks
        ]

    def update_zoom(self, zoom: float) -> None:
        self._zoom = zoom
        for index, brick in enumerate(self.zoom_bricks):
            self._recs[index].zoom = zoom
            self._recs[index].image = self.zoom_image.crop(brick.box)
            self._recs[index].brick = brick

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def zoom_image(self) -> Image.Image:
        return self.image.resize(
            (
                int(self.image.width * self.zoom),
                int(self.image.height * self.zoom),
            )
        )

    @property
    def zoom_bricks(self) -> List[BlueprintBrick]:
        return [brick.zoom(self.zoom) for brick in self.bricks]

    @property
    def filename(self) -> str:
        return f"{self.prefix}-{str(self.id).zfill(5)}-{self.zoom:.2f}.png"

    @property
    def label_line(self) -> str:
        data = [
            {"transcription": brick.blueprint.text, "points": brick.points}
            for brick in self.zoom_bricks
        ]
        return f"{Path(IMAGES_NAME).joinpath(self.filename).as_posix()}\t{json.dumps(data, ensure_ascii=False)}"

    def save(self) -> None:
        self.zoom_image.save(f"{IMAGES.joinpath(self.filename)}")

    @property
    def recs(self) -> List[Rec]:
        return self._recs


class ImageBuilder:
    def __init__(self, blueprints: List[Blueprint]):
        self.blueprints = blueprints
        self.blueprint_groups: Dict[Background, List[Blueprint]] = reduce(
            lambda acc, blueprint: {
                **acc,
                blueprint.background: acc.get(blueprint.background, []) + [blueprint],
            },
            blueprints,
            {},
        )

    def group_blueprints(
        self, blueprints: List[Blueprint]
    ) -> List[List[BlueprintBrick]]:
        result: List[List[BlueprintBrick]] = []
        filled_height = 0
        item: List[BlueprintBrick] = []
        while len(blueprints) > 0:
            blueprint = blueprints.pop(0)
            font = ImageFont.truetype(GENSHIN_TTF, blueprint.font_size)
            bbox = font.getbbox(blueprint.text)
            assert bbox[2] <= starry_sky_background.min_width
            filled_height += bbox[3]
            item.append(
                BlueprintBrick(
                    (
                        0,
                        0,
                        bbox[2],
                        bbox[3],
                    ),
                    blueprint,
                )
            )
            interval = (starry_sky_background.min_height - filled_height) / (
                len(item) + 1
            )
            if (
                filled_height > starry_sky_background.min_height
                or len(blueprints) == 0
                or interval < args.det_padding
            ):
                y = interval
                for i, brick in enumerate(item):
                    brick.rect = (
                        random_uniform(
                            args.rec_padding,
                            starry_sky_background.min_width
                            - brick.rect[2]
                            - args.rec_padding,
                        ),
                        y,
                        brick.rect[2],
                        brick.rect[3],
                    )
                    y += interval + brick.rect[3]
                result.append(item)
                item = []
                filled_height = 0
        return result

    def build_det(self, bricks: List[BlueprintBrick], background: Background) -> Det:
        if isinstance(background, StarrySky):
            image = starry_sky_background.get_random_frame(background)
        else:
            image = Image.new(
                "RGB",
                (starry_sky_background.min_width, starry_sky_background.min_height),
                background,
            )
        draw = ImageDraw.Draw(image)
        for brick in bricks:
            font = ImageFont.truetype(GENSHIN_TTF, brick.blueprint.font_size)
            draw.text(
                brick.rect[:2],
                text=brick.blueprint.text,
                font=font,
                fill=brick.blueprint.color,
            )
        return Det(bricks, image)

    def build(self) -> List[Det]:
        result: List[Det] = []
        for background, blueprints in self.blueprint_groups.items():
            for bricks in self.group_blueprints(random_shuffle(deepcopy(blueprints))):
                det = self.build_det(bricks, background)
                logger.debug(f"Generated {det}")
                result.append(det)
        return result


class Generator:
    def __init__(self, text: str):
        self.text = text

    def generate(self) -> List[Blueprint]:
        raise NotImplementedError()


class CharacterImageGenerator(Generator):
    def __init__(self, character: CharacterJson):
        super().__init__(character.name)
        self.character = character

    def generate_sky(self) -> Blueprint:
        return Blueprint(
            text=self.text,
            font_size=22,
            color=(211, 188, 142),
            background=self.character.element_text,
        )

    def generate_equipped(self) -> Blueprint:
        return Blueprint(
            text=f"{self.text}已装备",
            font_size=42,
            color=(73, 83, 102),
            background=(255, 231, 187),
        )

    def generate(self) -> List[Blueprint]:
        # 排除主角, 奇偶
        if self.character.element_text == ElementText.NONE:
            return []
        return [self.generate_sky(), self.generate_equipped()]


class ArtifactImageGenerator(Generator):
    def __init__(
        self, text: str, rarities: List[Rarity], slots: List[ArtifactSlotJson]
    ):
        super().__init__(text)
        self.rarities = rarities
        self.slots = slots
        self.rarity_background: Dict[Rarity, Tuple[int, int, int]] = {
            Rarity.WHITE: (112, 118, 138),
            Rarity.GREEN: (42, 142, 113),
            Rarity.BLUE: (80, 126, 201),
            Rarity.PURPLE: (159, 85, 222),
            Rarity.GOLD: (186, 105, 52),
        }

    def generate_part(self) -> List[Blueprint]:
        result: List[Blueprint] = []
        for slot in self.slots:
            for rarity in self.rarities:
                background = self.rarity_background.get(rarity)
                result.append(
                    Blueprint(
                        text=slot.name,
                        font_size=30,
                        color=(255, 255, 255),
                        background=background,
                    )
                )
        return result

    def generate_set(self) -> Blueprint:
        return Blueprint(
            text=self.text,
            font_size=22,
            color=(92, 178, 86),
            background=(236, 229, 216),
        )

    def generate_filter(self) -> List[Blueprint]:
        return [
            # 可选
            Blueprint(
                text=self.text,
                font_size=22,
                color=(236, 229, 216),
                background=(44, 56, 66),
            ),
            # 不可选
            Blueprint(
                text=self.text,
                font_size=22,
                color=(127, 129, 133),
                background=(44, 56, 66),
            ),
        ]

    def generate(self) -> List[Blueprint]:
        return flatten(
            self.generate_part(), self.generate_set(), self.generate_filter()
        )


class ArtifactRarityImageGenerator(Generator):
    def __init__(self, text: str, rarity: Rarity):
        super().__init__(text)
        self.rarity = rarity
        self.rarity_background: Dict[Rarity, Tuple[int, int, int]] = {
            Rarity.WHITE: (79, 87, 98),
            Rarity.GREEN: (74, 90, 94),
            Rarity.BLUE: (81, 85, 119),
            Rarity.PURPLE: (94, 88, 134),
            Rarity.GOLD: (110, 86, 83),
        }

    def generate_slot(self) -> List[Blueprint]:
        return [
            Blueprint(
                text=slot.value,
                font_size=30,
                color=(255, 255, 255),
                background=self.rarity_background.get(self.rarity),
            )
            for slot in Slot
        ]

    def generate_main_attribute(self) -> Blueprint:
        result: List[Blueprint] = []
        background = self.rarity_background.get(self.rarity)
        for affix in Affix:
            # 词条
            result.append(
                Blueprint(
                    text=affix.name,
                    font_size=20,
                    color=(188, 178, 206),
                    background=background,
                )
            )
            # 词条值
            result.append(
                Blueprint(
                    text=affix.random_value(),
                    font_size=38,
                    color=(255, 255, 255),
                    background=background,
                )
            )
        return result

    def generate_sky_unactivated_attribute(self) -> Blueprint:
        result: List[Blueprint] = []
        for affix in Affix:
            # 待激活词条
            result.append(
                Blueprint(
                    text=affix.random_unactivated_name(),
                    font_size=27,
                    color=(181, 164, 149),
                    background=self.rarity,
                )
            )
            # 强化词条值
            result.append(
                Blueprint(
                    text=affix.random_value(),
                    font_size=27,
                    color=(181, 164, 149),
                    background=self.rarity,
                )
            )
        return result

    def generate_sky_attribute(self) -> Blueprint:
        result: List[Blueprint] = []
        for affix in Affix:
            # 强化词条值
            result.append(
                Blueprint(
                    text=affix.name,
                    font_size=27,
                    color=(255, 255, 255),
                    background=self.rarity,
                )
            )
            # 强化词条值
            result.append(
                Blueprint(
                    text=affix.random_value(),
                    font_size=27,
                    color=(255, 255, 255),
                    background=self.rarity,
                )
            )
        return result

    def generate(self) -> List[Blueprint]:
        return flatten(
            self.generate_slot(),
            self.generate_main_attribute(),
            self.generate_sky_attribute(),
            self.generate_sky_unactivated_attribute(),
        )


class ArtifactCommonImageGenerator(Generator):
    def __init__(self):
        super().__init__("")

    def generate_level(self) -> List[Blueprint]:
        return [
            Blueprint(
                text=f"+{i}",
                font_size=22,
                color=(255, 255, 255),
                background=(57, 68, 79),
            )
            for i in range(1, 21)
        ]

    def generate_attribute(self) -> List[Blueprint]:
        return [
            Blueprint(
                text=affix.random_name_value(),
                font_size=22,
                color=(73, 83, 102),
                background=(236, 229, 216),
            )
            for affix in Affix
        ]

    def generate_unactivated_attribute(self) -> List[Blueprint]:
        return [
            Blueprint(
                text=affix.random_unactivated_name_value(),
                font_size=22,
                color=(154, 156, 159),
                background=(236, 229, 216),
            )
            for affix in Affix
        ]

    def generate_sanctifying_elixir(self) -> Blueprint:
        return Blueprint(
            text="祝圣之霜定义",
            font_size=20,
            color=(102, 61, 153),
            background=(220, 192, 255),
        )

    def generate(self) -> List[Blueprint]:
        return flatten(
            self.generate_level(),
            self.generate_attribute(),
            self.generate_unactivated_attribute(),
            self.generate_sanctifying_elixir(),
        )


class CommonImageGenerator(Generator):
    def __init__(self):
        super().__init__("")

    def generate_not_equipped_red_dot(self) -> List[Blueprint]:
        return [
            Blueprint(
                text=str(count),
                font_size=20,
                color=(255, 255, 255),
                background=(230, 69, 95),
            )
            for count in range(1, 6)
        ]

    def generate_element_text(self) -> List[Blueprint]:
        return [
            Blueprint(
                text=f"{element_text.value}元素",
                font_size=22,
                color=(211, 188, 142),
                background=element_text,
            )
            for element_text in ElementText
            if element_text != ElementText.NONE
        ]

    def generate(self) -> List[Blueprint]:
        return flatten(
            self.generate_not_equipped_red_dot(),
            self.generate_element_text(),
        )


class Parser:
    def parser(self) -> List[Generator]:
        raise NotImplementedError()


class CharacterParser(Parser):
    def parser(self) -> List[Generator]:
        return [
            CharacterImageGenerator(character)
            for character in database.read_characters()
        ]


class ArtifactParser(Parser):
    def parser(self) -> List[Generator]:
        return [
            ArtifactImageGenerator(
                artifact.name,
                [Rarity(rarity) for rarity in artifact.rarity_list],
                [
                    getattr(artifact, slot.name.lower())
                    for slot in Slot
                    if getattr(artifact, slot.name.lower()) is not None
                ],
            )
            for artifact in database.read_artifacts()
        ]


class ArtifactRarityParser(Parser):
    def parser(self) -> List[Generator]:
        return [ArtifactRarityImageGenerator(rarity.value, rarity) for rarity in Rarity]


class ArtifactCommonParser(Parser):
    def parser(self) -> List[Generator]:
        return [ArtifactCommonImageGenerator()]


class CommonParser(Parser):
    def parser(self) -> List[Generator]:
        return [CommonImageGenerator()]


def init() -> None:
    if DATASET.exists():
        logger.info("Dataset already exists, removing...")
        shutil.rmtree(DATASET)
    DATASET.mkdir(parents=True, exist_ok=True)
    CROP_IMG.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init()

    parsers: List[Parser] = [
        CharacterParser(),
        ArtifactParser(),
        ArtifactRarityParser(),
        ArtifactCommonParser(),
        CommonParser(),
    ]

    def build_dataset(dataset_type: str, count: int, zooms: List[float]) -> None:
        det_file = DATASET.joinpath(f"det_{dataset_type}.txt")
        rec_file = DATASET.joinpath(f"rec_{dataset_type}.txt")
        blueprints: List[Blueprint] = []
        image_hashes: Set[str] = set()

        append_file(det_file, "")
        append_file(rec_file, "")

        for parser in parsers:
            for builder in parser.parser():
                [blueprints.extend(builder.generate()) for _ in range(count)]

        for det in ImageBuilder(blueprints).build():
            for zoom in zooms:
                det.update_zoom(zoom)
                det.prefix = dataset_type
                append_file(det_file, det.label_line + "\n")
                det.save()
                for rec in det.recs:
                    if rec.image_hash in image_hashes:
                        continue
                    rec.prefix = dataset_type
                    append_file(rec_file, rec.label_line + "\n")
                    image_hashes.add(rec.image_hash)
                    rec.save()

    build_dataset("train", args.train_count, [1.0, 0.83, 0.66])
    build_dataset("val", args.val_count, [1.0, 0.83, 0.66])

    logger.info("Dataset built successfully.")
