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

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )

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
        "properties": {
            key: {"type": "string"} for key in FIELDS
        },
        "required": FIELDS,
        "additionalProperties": False,
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "너는 대한민국 실종자 재난문자에서 인상착의 정보를 "
                    "정확하게 추출하는 AI 보조 시스템이다. "
                    "문자에 실제로 적힌 정보만 사용하고, 없는 내용은 절대로 "
                    "추측하지 말며 빈 문자열로 둔다."
                ),
            },
            {
                "role": "user",
                "content": (
                    "다음 실종 재난문자에서 이름, 성별, 나이, 키, 몸무게, "
                    "체형, 상의, 하의, 신발, 액세서리/소지품, 머리 특징, "
                    "기타 특징을 추출해줘.\n\n"
                    f"{message}"
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 500,
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


def build_image_prompt(features: dict) -> str:
    labels = {
        "gender": "gender",
        "age": "age",
        "height": "height",
        "weight": "weight",
        "body_type": "body type",
        "top": "top clothing",
        "bottom": "bottom clothing",
        "shoes": "shoes",
        "accessories": "accessories or carried items",
        "hair": "hair",
        "special_features": "other distinguishing features",
    }

    facts = []
    for key, label in labels.items():
        value = features.get(key, "").strip()
        if value:
            facts.append(f"{label}: {value}")

    description = "; ".join(facts) if facts else "limited appearance information"

    return (
        "Create a neutral full-body reference illustration for a missing-person "
        "search aid. Use only the explicitly provided description. "
        f"Description: {description}. "
        "Do not invent distinctive facial details that were not provided. "
        "Show the person's clothing, body build, shoes, and accessories clearly. "
        "Simple bright background, front-facing full body, realistic but clearly "
        "illustrative reference style, no text, no name, no phone number, no address."
    )


def generate_image(prompt: str) -> bytes:
    result = cloudflare_request(
        IMAGE_MODEL,
        {
            "prompt": prompt,
            "steps": 4,
            "seed": random.randint(1, 999999999),
        },
    )

    image_b64 = result.get("image")
    if not image_b64:
        raise RuntimeError("이미지 데이터가 반환되지 않았습니다.")

    return base64.b64decode(image_b64)


st.title("🔎 FindVision AI")
st.caption("실종 재난문자의 인상착의를 AI가 구조화하고 참고 이미지로 시각화합니다.")

st.warning(
    "생성 이미지는 실제 얼굴 복원이나 신원 확인용이 아닙니다. "
    "재난문자에 포함된 인상착의를 쉽게 이해하기 위한 참고용 시각화입니다."
)

sample = (
    "진천군 주민인 이광표씨(남,68세)를 찾습니다. "
    "164cm,58kg, 빨간색반팔티, 검정색 긴바지, 검정색크록스"
)

message = st.text_area(
    "실종 재난문자 입력",
    value=sample,
    height=160,
)

if st.button("AI 분석 및 이미지 생성", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("실종 재난문자 내용을 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("Cloudflare AI가 인상착의를 분석하고 있습니다..."):
            features = extract_features(message.strip())

        prompt = build_image_prompt(features)

        with st.spinner("Cloudflare AI가 참고 이미지를 생성하고 있습니다..."):
            image_bytes = generate_image(prompt)

        left, right = st.columns([1, 1])

        with left:
            st.subheader("분석 결과")
            labels = {
                "name": "이름",
                "gender": "성별",
                "age": "나이",
                "height": "키",
                "weight": "몸무게",
                "body_type": "체형",
                "top": "상의",
                "bottom": "하의",
                "shoes": "신발",
                "accessories": "소지품/액세서리",
                "hair": "머리 특징",
                "special_features": "기타 특징",
            }

            for key, label in labels.items():
                value = features.get(key, "").strip()
                st.write(f"**{label}:** {value if value else '정보 없음'}")

        with right:
            st.subheader("참고 이미지")
            st.image(
                image_bytes,
                caption="AI 생성 인상착의 참고 이미지",
                use_container_width=True,
            )

        with st.expander("이미지 생성 프롬프트 보기"):
            st.code(prompt, language="text")

    except Exception as exc:
        st.error(f"오류가 발생했습니다: {exc}")
