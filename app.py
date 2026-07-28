import io
import re
from typing import Dict, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="FindVision AI - 조립식 인상착의",
    page_icon="🧩",
    layout="wide",
)

# -----------------------------
# 유틸 / 파싱
# -----------------------------
COLOR_TABLE = {
    "검정색": ("black", (35, 35, 35)),
    "검은색": ("black", (35, 35, 35)),
    "검정": ("black", (35, 35, 35)),
    "검은": ("black", (35, 35, 35)),
    "블랙": ("black", (35, 35, 35)),
    "흰색": ("white", (245, 245, 245)),
    "하얀색": ("white", (245, 245, 245)),
    "흰": ("white", (245, 245, 245)),
    "화이트": ("white", (245, 245, 245)),
    "빨간색": ("red", (220, 53, 69)),
    "빨강": ("red", (220, 53, 69)),
    "빨간": ("red", (220, 53, 69)),
    "레드": ("red", (220, 53, 69)),
    "파란색": ("blue", (52, 101, 214)),
    "파랑": ("blue", (52, 101, 214)),
    "파란": ("blue", (52, 101, 214)),
    "블루": ("blue", (52, 101, 214)),
    "남색": ("navy", (33, 56, 97)),
    "네이비": ("navy", (33, 56, 97)),
    "회색": ("gray", (140, 140, 140)),
    "회색빛": ("gray", (140, 140, 140)),
    "그레이": ("gray", (140, 140, 140)),
    "노란색": ("yellow", (247, 208, 70)),
    "노랑": ("yellow", (247, 208, 70)),
    "노란": ("yellow", (247, 208, 70)),
    "옐로우": ("yellow", (247, 208, 70)),
    "초록색": ("green", (52, 168, 83)),
    "초록": ("green", (52, 168, 83)),
    "녹색": ("green", (52, 168, 83)),
    "그린": ("green", (52, 168, 83)),
    "베이지": ("beige", (214, 194, 153)),
    "갈색": ("brown", (130, 90, 60)),
    "브라운": ("brown", (130, 90, 60)),
    "주황색": ("orange", (242, 137, 48)),
    "오렌지": ("orange", (242, 137, 48)),
    "보라색": ("purple", (119, 77, 147)),
    "보라": ("purple", (119, 77, 147)),
    "퍼플": ("purple", (119, 77, 147)),
}

TOP_KEYWORDS = [
    "반팔티", "반팔", "긴팔티", "긴팔", "티셔츠", "티", "점퍼", "자켓", "재킷",
    "셔츠", "후드티", "후드", "조끼", "패딩", "맨투맨", "니트", "블라우스"
]
BOTTOM_KEYWORDS = [
    "긴바지", "바지", "청바지", "면바지", "반바지", "치마", "트레이닝복", "슬랙스"
]
SHOES_KEYWORDS = [
    "크록스", "운동화", "구두", "슬리퍼", "샌들", "장화", "워커"
]
HAT_KEYWORDS = [
    "모자", "야구모자", "캡모자", "캡", "비니", "벙거지", "챙모자"
]
ACCESSORY_KEYWORDS = [
    "가방", "백팩", "배낭", "지팡이", "우산", "안경", "마스크", "목도리"
]

DEFAULTS = {
    "name": "",
    "gender": "",
    "age": "",
    "height": "",
    "weight": "",
    "body_type": "",
    "top": "",
    "bottom": "",
    "shoes": "",
    "hat": "",
    "accessories": "",
    "hair": "",
    "special_features": "",
}

SKIN = (244, 210, 177)
HAIR = (45, 36, 32)
LINE = (48, 48, 48)
BG = (250, 251, 253)

def safe(v: str) -> str:
    v = str(v or "").strip()
    return v if v else "정보 없음"

def split_chunks(text: str):
    chunks = re.split(r"[,\n/]+", text)
    return [c.strip() for c in chunks if c.strip()]

def find_color(text: str) -> Tuple[str, Tuple[int, int, int]]:
    for k, v in sorted(COLOR_TABLE.items(), key=lambda x: len(x[0]), reverse=True):
        if k in text:
            return v
    return ("unknown", (160, 160, 160))

def detect_body_type(text: str) -> str:
    if "마른" in text or "왜소" in text:
        return "마른 체형"
    if "통통" in text or "비만" in text:
        return "통통한 체형"
    if "보통" in text:
        return "보통 체형"
    return ""

def detect_name(text: str) -> str:
    patterns = [
        r"([가-힣]{2,4})씨",
        r"실종자\s*([가-힣O○]{2,5})",
        r"주민인\s*([가-힣O○]{2,5})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return ""

def contains_any(text: str, keywords):
    return any(k in text for k in keywords)

def extract_clothing_phrase(chunk: str, keywords) -> str:
    if contains_any(chunk, keywords):
        return chunk.strip(" .")
    return ""

def extract_features(text: str) -> Dict[str, str]:
    result = DEFAULTS.copy()
    raw = text.strip()

    result["name"] = detect_name(raw)

    if "여성" in raw or "여 " in raw or "(여" in raw or "여," in raw or "여자" in raw:
        result["gender"] = "여성"
    elif "남성" in raw or "남 " in raw or "(남" in raw or "남," in raw or "남자" in raw:
        result["gender"] = "남성"

    m = re.search(r"(\d{1,3})\s*세", raw)
    if m:
        result["age"] = f"{m.group(1)}세"

    m = re.search(r"(\d{2,3})\s*cm", raw, flags=re.I)
    if m:
        result["height"] = f"{m.group(1)}cm"

    m = re.search(r"(\d{2,3})\s*kg", raw, flags=re.I)
    if m:
        result["weight"] = f"{m.group(1)}kg"

    result["body_type"] = detect_body_type(raw)

    chunks = split_chunks(raw)

    for c in chunks:
        if not result["top"] and contains_any(c, TOP_KEYWORDS):
            result["top"] = extract_clothing_phrase(c, TOP_KEYWORDS)
        if not result["bottom"] and contains_any(c, BOTTOM_KEYWORDS):
            result["bottom"] = extract_clothing_phrase(c, BOTTOM_KEYWORDS)
        if not result["shoes"] and contains_any(c, SHOES_KEYWORDS):
            result["shoes"] = extract_clothing_phrase(c, SHOES_KEYWORDS)
        if not result["hat"] and contains_any(c, HAT_KEYWORDS):
            result["hat"] = extract_clothing_phrase(c, HAT_KEYWORDS)
        if not result["accessories"] and contains_any(c, ACCESSORY_KEYWORDS):
            # 모자는 accessory에서 제외
            if not contains_any(c, HAT_KEYWORDS):
                result["accessories"] = extract_clothing_phrase(c, ACCESSORY_KEYWORDS)

    # 쉼표로 안 분리된 경우 전체 문장 보조 탐색
    if not result["top"]:
        for k in TOP_KEYWORDS:
            idx = raw.find(k)
            if idx >= 0:
                result["top"] = raw[max(0, idx-10):idx+len(k)+2].strip(" ,.")
                break

    if not result["bottom"]:
        for k in BOTTOM_KEYWORDS:
            idx = raw.find(k)
            if idx >= 0:
                result["bottom"] = raw[max(0, idx-10):idx+len(k)+2].strip(" ,.")
                break

    if not result["shoes"]:
        for k in SHOES_KEYWORDS:
            idx = raw.find(k)
            if idx >= 0:
                result["shoes"] = raw[max(0, idx-10):idx+len(k)+2].strip(" ,.")
                break

    if not result["hat"]:
        for k in HAT_KEYWORDS:
            idx = raw.find(k)
            if idx >= 0:
                result["hat"] = raw[max(0, idx-10):idx+len(k)+2].strip(" ,.")
                break

    return result

# -----------------------------
# 조립식 캐릭터 렌더링
# -----------------------------
def body_scale(features: Dict[str, str]) -> float:
    bt = features.get("body_type", "")
    if "마른" in bt:
        return 0.92
    if "통통" in bt:
        return 1.08
    return 1.0

def draw_cap(draw, cx, cy, color):
    # 야구모자
    draw.pieslice((cx-70, cy-40, cx+70, cy+55), start=180, end=360, fill=color, outline=LINE, width=3)
    draw.ellipse((cx+20, cy+20, cx+85, cy+42), fill=color, outline=LINE, width=3)

def draw_beanie(draw, cx, cy, color):
    draw.rounded_rectangle((cx-68, cy-18, cx+68, cy+58), radius=22, fill=color, outline=LINE, width=3)
    draw.rectangle((cx-70, cy+30, cx+70, cy+52), fill=color, outline=LINE, width=3)

def draw_bucket_hat(draw, cx, cy, color):
    draw.polygon([(cx-50, cy-5), (cx+50, cy-5), (cx+72, cy+45), (cx-72, cy+45)], fill=color, outline=LINE)
    draw.rectangle((cx-42, cy-35, cx+42, cy+5), fill=color, outline=LINE, width=3)

def draw_hat(draw, features, cx, head_top):
    hat = features.get("hat", "")
    if not hat:
        return
    _, color = find_color(hat)
    cy = head_top - 5
    if "비니" in hat:
        draw_beanie(draw, cx, cy, color)
    elif "벙거지" in hat or "챙모자" in hat:
        draw_bucket_hat(draw, cx, cy, color)
    else:
        draw_cap(draw, cx, cy, color)

def draw_hair(draw, cx, head_top, head_bottom):
    draw.pieslice((cx-64, head_top-8, cx+64, head_bottom-5), start=180, end=360, fill=HAIR)
    draw.rounded_rectangle((cx-64, head_top+28, cx-48, head_top+96), radius=6, fill=HAIR)
    draw.rounded_rectangle((cx+48, head_top+28, cx+64, head_top+96), radius=6, fill=HAIR)

def draw_face(draw, cx, head_top):
    draw.ellipse((cx-58, head_top, cx+58, head_top+140), fill=SKIN, outline=LINE, width=3)
    # 눈
    draw.line((cx-28, head_top+62, cx-10, head_top+62), fill=LINE, width=3)
    draw.line((cx+10, head_top+62, cx+28, head_top+62), fill=LINE, width=3)
    # 코
    draw.line((cx, head_top+67, cx-4, head_top+90), fill=LINE, width=2)
    draw.line((cx-4, head_top+90, cx+3, head_top+95), fill=LINE, width=2)
    # 입
    draw.arc((cx-20, head_top+95, cx+20, head_top+115), start=10, end=170, fill=LINE, width=2)

def draw_top(draw, features, cx, torso_top, torso_bottom, scale):
    top = features.get("top", "")
    _, color = find_color(top if top else "회색")
    shoulder = int(105 * scale)
    waist = int(85 * scale)

    # 몸통
    draw.polygon([
        (cx-shoulder, torso_top),
        (cx+shoulder, torso_top),
        (cx+waist, torso_bottom),
        (cx-waist, torso_bottom),
    ], fill=color, outline=LINE)

    # 팔
    sleeve_short = ("반팔" in top)
    if sleeve_short:
        arm_y1 = torso_top + 35
        arm_y2 = torso_top + 110
        skin_y1 = arm_y2
        skin_y2 = torso_bottom - 10
    else:
        arm_y1 = torso_top + 20
        arm_y2 = torso_bottom - 12
        skin_y1 = arm_y2
        skin_y2 = arm_y2 + 40

    # 소매/팔
    draw.rounded_rectangle((cx-shoulder-26, arm_y1, cx-shoulder+10, arm_y2), radius=14, fill=color, outline=LINE, width=2)
    draw.rounded_rectangle((cx+shoulder-10, arm_y1, cx+shoulder+26, arm_y2), radius=14, fill=color, outline=LINE, width=2)

    # 노출 팔
    if sleeve_short:
        draw.rounded_rectangle((cx-shoulder-14, skin_y1, cx-shoulder+2, skin_y2), radius=8, fill=SKIN, outline=LINE, width=2)
        draw.rounded_rectangle((cx+shoulder-2, skin_y1, cx+shoulder+14, skin_y2), radius=8, fill=SKIN, outline=LINE, width=2)

    # 목
    draw.rounded_rectangle((cx-18, torso_top-18, cx+18, torso_top+20), radius=10, fill=SKIN, outline=LINE, width=2)

def draw_bottom(draw, features, cx, hip_y, knee_y, ankle_y, scale):
    bottom = features.get("bottom", "")
    _, color = find_color(bottom if bottom else "검정")
    leg_gap = int(18 * scale)
    leg_w = int(48 * scale)

    is_shorts = "반바지" in bottom
    is_skirt = "치마" in bottom

    if is_skirt:
        draw.polygon([
            (cx-90, hip_y), (cx+90, hip_y),
            (cx+58, knee_y-10), (cx-58, knee_y-10)
        ], fill=color, outline=LINE)
        # 종아리
        draw.rounded_rectangle((cx-leg_gap-leg_w, knee_y-10, cx-leg_gap, ankle_y), radius=12, fill=SKIN, outline=LINE, width=2)
        draw.rounded_rectangle((cx+leg_gap, knee_y-10, cx+leg_gap+leg_w, ankle_y), radius=12, fill=SKIN, outline=LINE, width=2)
    elif is_shorts:
        short_end = hip_y + 80
        draw.rounded_rectangle((cx-leg_gap-leg_w, hip_y, cx-leg_gap, short_end), radius=12, fill=color, outline=LINE, width=2)
        draw.rounded_rectangle((cx+leg_gap, hip_y, cx+leg_gap+leg_w, short_end), radius=12, fill=color, outline=LINE, width=2)
        draw.rounded_rectangle((cx-leg_gap-leg_w+8, short_end, cx-leg_gap-8, ankle_y), radius=12, fill=SKIN, outline=LINE, width=2)
        draw.rounded_rectangle((cx+leg_gap+8, short_end, cx+leg_gap+leg_w-8, ankle_y), radius=12, fill=SKIN, outline=LINE, width=2)
    else:
        # 긴바지
        draw.rounded_rectangle((cx-leg_gap-leg_w, hip_y, cx-leg_gap, ankle_y), radius=12, fill=color, outline=LINE, width=2)
        draw.rounded_rectangle((cx+leg_gap, hip_y, cx+leg_gap+leg_w, ankle_y), radius=12, fill=color, outline=LINE, width=2)

def draw_shoes(draw, features, cx, foot_y, scale):
    shoes = features.get("shoes", "")
    _, color = find_color(shoes if shoes else "검정")
    x1 = cx-72
    x2 = cx+12
    w = int(68 * scale)
    h = int(26 * scale)

    if "크록스" in shoes:
        # 앞코가 둥근 크록스
        draw.rounded_rectangle((x1, foot_y, x1+w, foot_y+h), radius=13, fill=color, outline=LINE, width=2)
        draw.rounded_rectangle((x2, foot_y, x2+w, foot_y+h), radius=13, fill=color, outline=LINE, width=2)
        # 구멍
        for sx in [x1+18, x1+32, x1+46]:
            draw.ellipse((sx, foot_y+7, sx+4, foot_y+11), fill=(230,230,230))
        for sx in [x2+18, x2+32, x2+46]:
            draw.ellipse((sx, foot_y+7, sx+4, foot_y+11), fill=(230,230,230))
        # 뒤 스트랩
        draw.arc((x1+4, foot_y-6, x1+w-4, foot_y+h+6), start=190, end=310, fill=LINE, width=2)
        draw.arc((x2+4, foot_y-6, x2+w-4, foot_y+h+6), start=190, end=310, fill=LINE, width=2)
    elif "슬리퍼" in shoes or "샌들" in shoes:
        draw.rounded_rectangle((x1, foot_y+6, x1+w, foot_y+h), radius=7, fill=color, outline=LINE, width=2)
        draw.rounded_rectangle((x2, foot_y+6, x2+w, foot_y+h), radius=7, fill=color, outline=LINE, width=2)
        draw.line((x1+8, foot_y+7, x1+w-8, foot_y+7), fill=LINE, width=3)
        draw.line((x2+8, foot_y+7, x2+w-8, foot_y+7), fill=LINE, width=3)
    else:
        # 운동화/구두
        draw.rounded_rectangle((x1, foot_y, x1+w, foot_y+h), radius=10, fill=color, outline=LINE, width=2)
        draw.rounded_rectangle((x2, foot_y, x2+w, foot_y+h), radius=10, fill=color, outline=LINE, width=2)
        draw.line((x1+12, foot_y+10, x1+w-12, foot_y+10), fill=(235,235,235), width=2)
        draw.line((x2+12, foot_y+10, x2+w-12, foot_y+10), fill=(235,235,235), width=2)

def draw_accessories(draw, features, cx, torso_top, torso_bottom):
    acc = features.get("accessories", "")
    if not acc:
        return

    _, color = find_color(acc)

    if "가방" in acc or "백팩" in acc or "배낭" in acc:
        # 한쪽 어깨 가방
        draw.line((cx+70, torso_top+20, cx+120, torso_top+110), fill=LINE, width=5)
        draw.rounded_rectangle((cx+95, torso_top+95, cx+160, torso_top+180), radius=12, fill=color, outline=LINE, width=2)
    elif "지팡이" in acc:
        draw.line((cx+128, torso_bottom-20, cx+128, torso_bottom+255), fill=(110, 80, 45), width=6)
        draw.arc((cx+112, torso_bottom-35, cx+145, torso_bottom-4), start=180, end=360, fill=(110, 80, 45), width=5)
    elif "우산" in acc:
        draw.line((cx+130, torso_top+40, cx+130, torso_bottom+260), fill=LINE, width=4)
        draw.arc((cx+112, torso_bottom+247, cx+140, torso_bottom+280), start=0, end=160, fill=LINE, width=4)
    elif "안경" in acc:
        # 얼굴쪽 액세서리는 렌더링 함수 밖에서 처리하지 않고 여기선 생략
        pass

def draw_glasses_if_needed(draw, features, cx, head_top):
    acc = features.get("accessories", "")
    if "안경" in acc:
        draw.ellipse((cx-36, head_top+48, cx-8, head_top+74), outline=LINE, width=3)
        draw.ellipse((cx+8, head_top+48, cx+36, head_top+74), outline=LINE, width=3)
        draw.line((cx-8, head_top+61, cx+8, head_top+61), fill=LINE, width=3)

def render_avatar(features: Dict[str, str]) -> Image.Image:
    W, H = 760, 980
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # 프레임
    draw.rounded_rectangle((20, 20, W-20, H-20), radius=24, outline=(221, 226, 232), width=3, fill=(255, 255, 255))

    scale = body_scale(features)
    cx = W // 2
    head_top = 120
    torso_top = 275
    torso_bottom = 470
    hip_y = 470
    knee_y = 680
    ankle_y = 845
    foot_y = 848

    # 그림자
    draw.ellipse((cx-120, 896, cx+120, 936), fill=(234, 237, 242))

    draw_hair(draw, cx, head_top, head_top+140)
    draw_face(draw, cx, head_top)
    draw_glasses_if_needed(draw, features, cx, head_top)
    draw_hat(draw, features, cx, head_top)
    draw_top(draw, features, cx, torso_top, torso_bottom, scale)
    draw_bottom(draw, features, cx, hip_y, knee_y, ankle_y, scale)
    draw_shoes(draw, features, cx, foot_y, scale)
    draw_accessories(draw, features, cx, torso_top, torso_bottom)

    # 간단한 제목
    title = safe(features.get("name"))
    if title == "정보 없음":
        title = "실종자"
    subtitle = "조립식 인상착의 참고 이미지"
    try:
        font_big = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font_big = None
        font_small = None

    draw.text((38, 38), title, fill=(25, 33, 45), font=font_big)
    draw.text((38, 76), subtitle, fill=(93, 105, 120), font=font_small)

    return img

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧩 FindVision AI - 조립식 인상착의")
st.caption("재난문자에서 인상착의를 추출하고, 모자·상의·하의·신발을 정확하게 반영하는 조립식 참고 이미지를 생성합니다.")

st.info(
    "이 버전은 생성형 그림이 아니라 조립식 렌더링 방식입니다. "
    "그래서 모자 누락, 바지 색상 오류, 신발 종류 오류를 줄이는 데 초점을 맞췄습니다."
)

sample = (
    "실종자 남성 68세, 키 164cm, 58kg, 마른 체형, "
    "빨간색 반팔티, 검은색 긴바지, 검은색 크록스, 회색 모자 착용"
)

message = st.text_area(
    "실종 재난문자 입력",
    value=sample,
    height=140,
)

if st.button("조립식 참고 이미지 만들기", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("실종 재난문자를 입력해 주세요.")
        st.stop()

    features = extract_features(message)

    img = render_avatar(features)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    st.divider()
    st.subheader("조립식 인상착의 참고 이미지")
    st.image(buf.getvalue(), use_container_width=True)

    left, right = st.columns(2)

    info_items = [
        ("이름", features["name"]),
        ("성별", features["gender"]),
        ("나이", features["age"]),
        ("키", features["height"]),
        ("몸무게", features["weight"]),
        ("체형", features["body_type"]),
        ("상의", features["top"]),
        ("하의", features["bottom"]),
        ("신발", features["shoes"]),
        ("모자", features["hat"]),
        ("소지품/액세서리", features["accessories"]),
        ("머리", features["hair"]),
        ("기타 특징", features["special_features"]),
    ]

    with left:
        for label, value in info_items[:7]:
            st.write(f"**{label}**: {safe(value)}")

    with right:
        for label, value in info_items[7:]:
            st.write(f"**{label}**: {safe(value)}")

    st.download_button(
        "이미지 다운로드 (PNG)",
        data=buf.getvalue(),
        file_name="findvision_avatar.png",
        mime="image/png",
        use_container_width=True,
    )

    with st.expander("원문 재난문자"):
        st.write(message)
