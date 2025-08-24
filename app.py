import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ==============================
# LLM呼び出し用の関数
# ==============================
def get_llm_response(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類に基づき、LLMの回答を生成して返す関数
    """
    # 専門家の役割に応じたシステムメッセージ
    if expert_type == "医師":
        system_prompt = "あなたは医学の専門家です。患者にわかりやすく丁寧に説明してください。"
    elif expert_type == "弁護士":
        system_prompt = "あなたは法律の専門家です。法律の観点からわかりやすく助言してください。"
    else:
        system_prompt = "あなたは一般的な知識を持つアシスタントです。丁寧に説明してください。"

    # LLMインスタンス作成
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5
    )

    # メッセージ構成
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # LLM呼び出し
    response = llm.invoke(messages)
    return response.content


# ==============================
# Streamlit UI
# ==============================
def main():
    st.title("専門家アシスタント Chat App")

    # アプリ概要
    st.markdown("""
    ### アプリの概要
    このアプリは、あなたの入力テキストに対して **LLM (LangChain + OpenAI)** を使って回答を生成します。  
    回答は、ラジオボタンで選択した「専門家の種類」に基づいて生成されます。  

    ### 使い方
    1. 下のフォームに質問や相談したい内容を入力してください。  
    2. 専門家の種類を選択してください（例：医師 / 弁護士）。  
    3. 「送信」ボタンを押すと、専門家の視点からの回答が表示されます。  
    """)

    # 入力フォーム
    user_input = st.text_area("質問内容を入力してください:", height=100)

    # 専門家の種類選択
    expert_type = st.radio(
        "専門家の種類を選択してください:",
        ("医師", "弁護士")
    )

    # 送信ボタン
    if st.button("送信"):
        if user_input.strip():
            with st.spinner("LLMに問い合わせ中..."):
                response = get_llm_response(user_input, expert_type)
            st.subheader("回答結果")
            st.write(response)
        else:
            st.warning("テキストを入力してください。")


if __name__ == "__main__":
    main()
