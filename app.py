import base64
import json
import os
import random

import requests
import streamlit as st

st.set_page_config(
    page_title="FindVision AI",
    page_icon="🔎",
    layout="wide",
)

TEXT_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

FIELDS = [
    "name",
    "gender",
    "age",
    "height",
    "weight",
    "body_type",
    "top",
    "bottom",
    "shoes",
    "accessories",
    "hair",
    "special_features",
    "image_prompt_en",
]


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def cloudflare_request(model: str, payload: dict) -> dict:
    account_id = get_secret("CLOUDFLARE_ACCOUNT_ID")
    api_token = get_secret("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise RuntimeError(
            "Cloudflare 설정이 없습니다. Streamlit Secrets에 "
            "CLOUDFLARE_ACCOUNT_ID와 CLOUDFLARE_API_TOKEN을 입력해 주세요."
        )

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            f"Cloudflare 응답을 읽을 수 없습니다. HTTP {response.status_code}"
        )

    if not response.ok or not data.get("success", False):
        errors = data.get("errors") or []
        message = errors[0].get("message") if errors else str(data)
        raise RuntimeError(
            f"Cloudflare AI 요청 실패 (HTTP {response.status_code}): {message}"
        )

    return data["result"]


def extract_features(message: str) -> dict:
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in FIELDS},
        "required": FIELDS,
        "additionalProperties": False,
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 대한민국 실종자 재난문자에서 인상착의를 구조화하는 AI다. "
                    "반드시 문자에 실제로 적힌 정보만 사용하고, 없는 정보는 빈 문자열로 둔다. "
                    "얼굴 생김새나 머리 모양처럼 원문에 없는 특징은 추측하지 않는다. "
                    "image_prompt_en에는 이미지 생성을 위한 영어 문장만 작성한다. "
                    "성별, 나이, 키, 몸무게, 복장 색상, 신발과 소지품을 원문과 정확히 맞춘다."
                ),
            },
            {
                "role": "user",
                "content": (
                    "다음 실종 재난문자에서 정보를 추출해줘.\n\n"
                    f"{message}\n\n"
                    "image_prompt_en은 다음 형식처럼 작성해: "
                    "'Korean male, 68 years old, 164 cm tall, 58 kg, "
                    "wearing a red short-sleeve T-shirt, black long pants, "
                    "and black Crocs.'"
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 700,
        "response_format": {
            "type": "json_schema",
            "json_schema": schema,
        },
    }

    result = cloudflare_request(TEXT_MODEL, payload)
    parsed = result.get("response", {})

    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    if not isinstance(parsed, dict):
        raise RuntimeError("AI 분석 결과가 올바른 JSON 형식이 아닙니다.")

    return {key: str(parsed.get(key, "") or "") for key in FIELDS}


def build_portrait_prompt(features: dict) -> str:
    translated = features.get("image_prompt_en", "").strip()
    if not translated:
        translated = "A Korean person based only on the provided description."

    return (
        "Create ONE single centered portrait reference image of this person. "
        f"Person description: {translated} "
        "Show head, shoulders, upper torso, and enough clothing to clearly show the top color. "
        "Neutral front-facing pose, plain white studio background, natural lighting. "
        "Realistic but clearly AI-generated reference portrait, not an official ID photo. "
        "Do not invent distinctive facial details that were not provided. "
        "Do not create text anywhere in the image. "
        "NO letters, NO Korean, NO Chinese, NO Japanese, NO English, NO numbers, "
        "NO labels, NO captions, NO poster, NO infographic, NO watermark."
    )


def generate_portrait(prompt: str) -> bytes:
    result = cloudflare_request(
        IMAGE_MODEL,
        {
            "prompt": prompt,
            "steps": 8,
            "seed": random.randint(1, 999999999),
        },
    )

    image_b64 = result.get("image")
    if not image_b64:
        raise RuntimeError("이미지 데이터가 반환되지 않았습니다.")

    return base64.b64decode(image_b64)


def safe(value: str) -> str:
    value = str(value or "").strip()
    return value if value else "정보 없음"


def description_items(features: dict) -> list[tuple[str, str]]:
    return [
        ("성별", safe(features.get("gender"))),
        ("나이", safe(features.get("age"))),
        ("키", safe(features.get("height"))),
        ("몸무게", safe(features.get("weight"))),
        ("체형", safe(features.get("body_type"))),
        ("상의", safe(features.get("top"))),
        ("하의", safe(features.get("bottom"))),
        ("신발", safe(features.get("shoes"))),
        ("소지품·액세서리", safe(features.get("accessories"))),
        ("머리 특징", safe(features.get("hair"))),
        ("기타 특징", safe(features.get("special_features"))),
    ]


st.markdown(
    """
    <style>
    .title-box {
        text-align: center;
        padding: 0.2rem 0 1rem 0;
    }
    .info-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 10px;
        min-height: 92px;
    }
    .info-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .info-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        word-break: keep-all;
    }
    .notice {
        background: #fff8db;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 FindVision AI")
st.caption("실종 재난문자의 인상착의를 분석하고, 얼굴 참고 이미지와 한국어 인상착의를 함께 표시합니다.")

st.markdown(
    '<div class="notice">생성된 얼굴은 실제 얼굴 복원이나 신원 확인용이 아닙니다. '
    '재난문자에 없는 얼굴 정보는 정확히 알 수 없으므로 참고용으로만 사용해야 합니다.</div>',
    unsafe_allow_html=True,
)

sample = (
    "진천군 주민인 이광표씨(남,68세)를 찾습니다. "
    "164cm,58kg, 빨간색 반팔티, 검정색 긴바지, 검정색 크록스"
)

message = st.text_area(
    "실종 재난문자 입력",
    value=sample,
    height=150,
)

if st.button("AI 분석 및 참고 이미지 생성", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("실종 재난문자 내용을 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("재난문자에서 인상착의를 분석하고 있습니다..."):
            features = extract_features(message.strip())

        prompt = build_portrait_prompt(features)

        with st.spinner("얼굴 참고 이미지를 생성하고 있습니다..."):
            portrait_bytes = generate_portrait(prompt)

        name = safe(features.get("name"))
        if name == "정보 없음":
            name = "실종자"

        st.divider()
        st.markdown(
            f'<div class="title-box"><h1>{name} 인상착의 참고</h1></div>',
            unsafe_allow_html=True,
        )

        left, center, right = st.columns([1.05, 1.35, 1.05], gap="large")
        items = description_items(features)

        with left:
            for label, value in items[:6]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with center:
            st.image(
                portrait_bytes,
                caption="AI 생성 얼굴 참고 이미지",
                use_container_width=True,
            )
            st.info("얼굴 세부 특징이 문자에 없다면 생성된 얼굴은 실제 인물과 다를 수 있습니다.")

        with right:
            for label, value in items[6:]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with st.expander("원문 재난문자"):
            st.write(message)

        with st.expander("이미지 생성용 영어 설명"):
            st.write(features.get("image_prompt_en", ""))

    except Exception as exc:
        st.error(f"오류가 발생했습니다: {exc}")
