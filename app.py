import base64
import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(
    page_title="FindVision AI",
    page_icon="🔎",
    layout="wide",
)

FIELDS = [
    "name",
    "gender",
    "age",
    "height",
    "body_type",
    "top",
    "bottom",
    "shoes",
    "accessories",
    "hair",
    "special_features",
]


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("OPENAI_API_KEY가 없습니다. .env 파일에 API 키를 입력하세요.")
        st.stop()
    return OpenAI(api_key=api_key)


def extract_features(client: OpenAI, message: str) -> dict:
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in FIELDS},
        "required": FIELDS,
        "additionalProperties": False,
    }

    instructions = """
너는 대한민국 실종자 재난문자에서 인상착의 정보를 추출하는 AI 보조 시스템이다.

반드시 다음 규칙을 따른다.
- 문자에 실제로 적힌 정보만 추출한다.
- 문자에 없는 얼굴 특징이나 외모는 추측하지 않는다.
- 알 수 없는 값은 빈 문자열("")로 둔다.
- 나이는 가능하면 '72세', 키는 가능하면 '170cm'처럼 정리한다.
- special_features에는 흉터, 점, 보행 특징, 소지품 등 추가 식별 단서를 정리한다.
"""

    response = client.responses.create(
        model=os.getenv("TEXT_MODEL", "gpt-5-mini"),
        instructions=instructions,
        input=message,
        text={
            "format": {
                "type": "json_schema",
                "name": "missing_person_features",
                "strict": True,
                "schema": schema,
            }
        },
    )

    return json.loads(response.output_text)


def build_image_prompt(features: dict) -> str:
    labels = {
        "gender": "성별",
        "age": "나이",
        "height": "키",
        "body_type": "체형",
        "top": "상의",
        "bottom": "하의",
        "shoes": "신발",
        "accessories": "소지품/액세서리",
        "hair": "머리 특징",
        "special_features": "기타 특징",
    }

    lines = []
    for key, label in labels.items():
        value = str(features.get(key, "") or "").strip()
        if value:
            lines.append(f"- {label}: {value}")

    facts = "\n".join(lines) if lines else "- 문자에 명시된 외형 정보가 거의 없음"

    return f"""
실종자 수색 지원을 위한 인상착의 참고 일러스트를 만든다.

문자에 명시된 정보:
{facts}

규칙:
- 위에 명시된 정보만 충실하게 반영한다.
- 문자에 없는 얼굴 특징을 임의로 특정하지 않는다.
- 실제 얼굴 복원물이 아니라 참고용 시각화라는 느낌의 자연스러운 일러스트로 만든다.
- 머리부터 발끝까지 전신이 보이게 한다.
- 복장 색상, 체형, 소지품이 잘 보이게 한다.
- 배경은 단순하고 밝게 한다.
- 이미지 안에 이름, 전화번호, 주소 등 개인정보 텍스트를 넣지 않는다.
""".strip()


def generate_image(client: OpenAI, prompt: str) -> bytes:
    result = client.images.generate(
        model=os.getenv("IMAGE_MODEL", "gpt-image-1"),
        prompt=prompt,
        size="1024x1024",
        quality="medium",
    )

    if not result.data or not result.data[0].b64_json:
        raise RuntimeError("이미지 생성 결과를 받지 못했습니다.")

    return base64.b64decode(result.data[0].b64_json)


st.title("🔎 FindVision AI")
st.caption("실종 재난문자의 인상착의를 AI가 구조화하고 참고 이미지로 시각화합니다.")

st.warning(
    "이 프로젝트의 생성 이미지는 실제 얼굴을 복원하거나 신원을 확정하는 용도가 아닙니다. "
    "문자에 포함된 인상착의를 이해하기 쉽게 보여주는 참고 자료입니다."
)

sample = (
    "실종자 김OO(남, 72세), 키 170cm, 마른 체형, "
    "흰색 반팔티, 검은색 바지, 회색 모자와 검은 운동화 착용"
)

message = st.text_area(
    "실종 재난문자 입력",
    height=160,
    placeholder=sample,
)

if st.button("AI 분석 및 이미지 생성", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("실종 재난문자 내용을 입력해 주세요.")
        st.stop()

    try:
        client = get_client()

        with st.spinner("문자에서 인상착의를 분석하고 있습니다..."):
            features = extract_features(client, message.strip())

        left, right = st.columns([1, 1])

        with left:
            st.subheader("분석 결과")
            labels = {
                "name": "이름",
                "gender": "성별",
                "age": "나이",
                "height": "키",
                "body_type": "체형",
                "top": "상의",
                "bottom": "하의",
                "shoes": "신발",
                "accessories": "소지품/액세서리",
                "hair": "머리 특징",
                "special_features": "기타 특징",
            }

            for key, label in labels.items():
                value = str(features.get(key, "") or "").strip()
                st.write(f"**{label}:** {value if value else '정보 없음'}")

        prompt = build_image_prompt(features)

        with right:
            st.subheader("참고 이미지")
            with st.spinner("참고 이미지를 생성하고 있습니다..."):
                image_bytes = generate_image(client, prompt)
            st.image(image_bytes, caption="AI 생성 인상착의 참고 이미지", use_container_width=True)

        with st.expander("이미지 생성 프롬프트 보기"):
            st.code(prompt, language="text")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
